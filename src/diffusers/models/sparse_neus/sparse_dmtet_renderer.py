"""
The codes are heavily borrowed from NeuS
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
import trimesh
from icecream import ic
from diffusers.models.sparse_neus.render_utils import sample_pdf
from diffusers.models.sparse_neus.projector import Projector
from diffusers.models.sparse_neus.torchsparse_utils import sparse_to_dense_channel
from diffusers.models.sparse_neus.fast_renderer import FastRenderer
from diffusers.models.sparse_neus.patch_projector import PatchProjector
from diffusers.models.sparse_neus.rays import gen_rays_between

import pdb
from torchvision.utils import save_image

from diffusers.utils import is_torch_version

from diffusers.models.get3d.uni_rep.rep_3d.dmtet import DMTetGeometry
from diffusers.models.get3d.uni_rep.rep_3d.flexicubes_geometry import FlexiCubesGeometry
from diffusers.models.get3d.uni_rep.camera.perspective_camera import PerspectiveCamera
from diffusers.models.get3d.uni_rep.render.neural_render import NeuralRender

import kaolin as kal
from diffusers.models.get3d.uni_rep import flex_render

# NOTE(lihe): test checkpoint training
def create_custom_forward(module):
    def custom_forward(*inputs):
        return module(*inputs)

    return custom_forward

class SparseDMTetRenderer(nn.Module):
    """
    conditional neus render;
    optimize on normalized world space;
    warped by nn.Module to support DataParallel traning
    """

    def __init__(self,
                 sdf_def_network,
                 rendering_network,
                 learn_bg_color=False,
                 rgb_ch=3,
                 pos_code=None,
                 use_resnetfc=False,
                 direct_use_3d=False,
                 use_3d_prior=False,
                 geo_attn_mask=False,
                 iso_surface='dmtet',
                 device='cuda',
                 render_type='neural_render',  # neural type
                 dmtet_scale=0.7, # 1.8,
                 tet_res=90, # 64,  # Resolution for tetrahedron grid
                 deformation_multiplier=1.0,# 2.0,
                 img_resolution=256, # rendering resolution
                 ):
        super(SparseDMTetRenderer, self).__init__()
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        # network setups
        self.sdf_def_network = sdf_def_network
        self.rendering_network = rendering_network

        self.pos_code = pos_code
        self.rgb_ch = rgb_ch

        self.use_resnetfc = use_resnetfc
        self.direct_use_3d = direct_use_3d
        self.use_3d_prior = use_3d_prior
        self.geo_attn_mask = geo_attn_mask

        self.iso_surface = iso_surface

        self.rendering_projector = Projector()  # used to obtain features for generalized rendering 

        self.h_patch_size = 3 # NOTE(lihe): set directly to 3

        # - fitted rendering or general rendering
        self.if_fitted_rendering = False
        self.v_cont = 0

        self.learn_bg_color = learn_bg_color
        if self.learn_bg_color:
            background = nn.Parameter(
                    torch.from_numpy(np.array([-1.]*rgb_ch)).to(dtype=torch.float32))
            self.register_parameter("background", background)
        
        # new parameters
        self.device = device
        self.dmtet_scale = dmtet_scale # TODO(lihe): check this scale !!!!!!!!
        self.grid_res = tet_res
        self.deformation_multiplier = deformation_multiplier
        self.img_resolution = img_resolution
        if iso_surface == "flexicubes":
            self.deformation_multiplier *= 2

        # Camera defination, we follow the defination from Blender (check the render_shapenet_data/rener_shapenet.py for more details)
        fovy = np.arctan(32 / 2 / 35) * 2
        fovyangle = fovy / np.pi * 180.0
        dmtet_camera = PerspectiveCamera(fovy=fovyangle, device=self.device)

        # Renderer we used.
        dmtet_renderer = NeuralRender(device, camera_model=dmtet_camera)
        
        # Geometry class for DMTet
        if self.iso_surface == 'dmtet':
            self.dmtet_geometry = DMTetGeometry(
                grid_res=self.grid_res, scale=self.dmtet_scale, renderer=dmtet_renderer, render_type=render_type,
                device=self.device)
        elif self.iso_surface == 'flexicubes':
            self.dmtet_geometry = FlexiCubesGeometry(
                grid_res=self.grid_res, scale=self.dmtet_scale, renderer=dmtet_renderer, render_type=render_type,
                device=self.device)

        self.debug_color = True        
    
    def render_mesh(self, mesh_v, mesh_f, cam_mv):
        '''
        Function to render a generated mesh with nvdiffrast
        :param mesh_v: List of vertices for the mesh
        :param mesh_f: List of faces for the mesh
        :param cam_mv:  4x4 rotation matrix
        :return:
        '''
        return_value_list = []
        for i_mesh in range(len(mesh_v)):
            return_value = self.dmtet_geometry.render_mesh(
                mesh_v[i_mesh],
                mesh_f[i_mesh].int(),
                cam_mv[i_mesh],
                resolution=self.img_resolution,
                hierarchical_mask=False
            )
            return_value_list.append(return_value)

        return_keys = return_value_list[0].keys()
        return_value = dict()
        for k in return_keys:
            value = [v[k] for v in return_value_list]
            return_value[k] = value

        mask_list, hard_mask_list = torch.cat(return_value['mask'], dim=0), torch.cat(return_value['hard_mask'], dim=0)
        return mask_list, hard_mask_list, return_value
    
    def generate(self, volume, cam_mv, conditional_valid_mask_volume=None, feature_maps=None,
                 img_wh=[64, 64], color_maps=None, w2cs=None, intrinsics=None, vol_dims=None, 
                 partial_vol_origin=None, # [3]
                 emb=None, t=None, vol_size=None, mesh=None,
                 kal_cameras=None,
                 res=128, query_pts=None,
                 color_gen=None):
        # cam_mv: b, nv, 4, 4
        cam_mv_inv = torch.inverse(cam_mv)
        cam_mv_inv = cam_mv_inv.unsqueeze(0)
        run_n_view = cam_mv_inv.shape[1]
        # TODO(lihe): check if the network has bug
        if mesh is None:
            mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(volume)
            if self.iso_surface == 'dmtet':
                sdf_reg_loss = sdf_reg_loss[0] # sdf ref loss is a tuple (loss, None, None) # TODO(lihe): compute other losses when using flexicubes
        else:
            # render extra view using previous constructed mesh
            # mesh_v, mesh_f = mesh[0], mesh[1]
            sdf = None
            sdf_reg_loss = torch.tensor(0., device=cam_mv.device) if self.iso_surface == 'dmtet' else (torch.tensor(0., device=cam_mv.device), torch.tensor(0., device=cam_mv.device), torch.tensor(0., device=cam_mv.device))

        # NOTE(lihe): debug rendering
        flexicubes_mesh = kal.rep.SurfaceMesh(mesh_v[0], mesh_f[0]) if mesh is None else mesh
        buffers = flex_render.render_mesh(flexicubes_mesh, kal_cameras, [self.img_resolution, self.img_resolution])
        # buffers = flex_render.render_mesh(flexicubes_mesh, kal_cameras, [self.img_resolution, self.img_resolution], return_types = ["mask", "depth", "tex_pos"])
        buffers['sdf_reg_loss'] = sdf_reg_loss
        buffers['sdf'] = sdf
        buffers['color_fine'] = torch.zeros(run_n_view*res*res, 3).cuda()
        buffers['flex_mesh'] = flexicubes_mesh
        # return buffers

        # render color
        low_cameras = kal.render.camera.Camera.from_args(view_matrix=cam_mv_inv[0], # [8, 4, 4]
                                                             fov=0.8575560450553894, # 30 * np.pi / 180,
                                                             width=res, height=res,
                                                             device='cuda')
        low_buffers = flex_render.render_mesh(flexicubes_mesh, low_cameras, [res, res], return_types = ["mask", "tex_pos"])
        tex_pos_mv = low_buffers['tex_pos']
        hard_mask_mv = low_buffers['mask']
        
        # Render the mesh into 2D image (get 3d position of each image plane)
        # NOTE(lihe): add view dimension for cam_mv, use for loop to render
        color_list = []
        depth_list = []
        antilias_mask_list = []
        sdf_list = []
        for i in range(run_n_view):
            if not self.debug_color:
                antilias_mask, hard_mask, return_value = self.render_mesh(mesh_v, mesh_f, cam_mv_inv[:, i:i+1, :, :])
                tex_pos = return_value['tex_pos'][0]
                depth = return_value['depth'][0] # NOTE(lihe): the depth of nvdiffrast is negative
                depth = -depth
            else:
                hard_mask = hard_mask_mv[i:i+1]
                tex_pos = tex_pos_mv[i]
            
            if (hard_mask > 0.5).sum() == 0:
                hard_mask[:, 64:80, 64:80, :] = 1.
            valid_mask = hard_mask.view(-1) > 0.5
            tex_pos_valid = tex_pos.view(-1, 3)[valid_mask].detach() # NOTE(lihe): fix bug here, remeber to detach
            assert tex_pos_valid.shape[0] > 0, "tex pos has no points!"
        
            # Querying the texture field to predict the texture feature for each pixel on the image
            # 1. we should first get texture field
            ren_geo_feats, ren_rgb_feats, ren_ray_diff, ren_mask, _, _ = self.rendering_projector.compute(
                    # tex_pos.view(-1, 1, 3),
                    tex_pos_valid.view(-1, 1, 3),
                    # * 3d geometry feature volumes
                    geometryVolume=volume[0],
                    geometryVolumeMask=conditional_valid_mask_volume[0],
                    # * 2d rendering feature maps
                    rendering_feature_maps=feature_maps,
                    color_maps=color_maps,
                    w2cs=w2cs,
                    intrinsics=intrinsics,
                    img_wh=img_wh,
                    query_img_idx=0,  # the index of the N_views dim for rendering
                    query_c2w=cam_mv[i:i+1], # query_c2w, # NOTE(lihe): we render multiple views at once for now
                    # NOTE(lihe): to reduce memory, feed voxel param to generate mask
                    vol_dims=vol_dims,
                    partial_vol_origin=partial_vol_origin,
                    vol_size=vol_size,
                )
            
            # 2. add positional encoding
            if self.pos_code is not None:
                # pos_feats = self.pos_code(tex_pos.view(-1, 3))
                pos_feats = self.pos_code(tex_pos_valid.view(-1, 3))
                pos_feats = pos_feats.view(-1, 1, self.pos_code.d_out)
                if not self.use_resnetfc:
                    ren_geo_feats = torch.cat([ren_geo_feats, pos_feats], dim=-1)
            
            # 3. forward color resnet
            if self.use_resnetfc:
                # NOTE(lihe): [N_views, N_rays, n_samples, 3+c]
                rgb_in = ren_rgb_feats[..., :self.rgb_ch] # [N_views, N_rays, n_samples, 3]
                feats_map = ren_rgb_feats[..., self.rgb_ch:]
                feats_map = feats_map.mean(dim=0) # [N_rays, n_samples, c]

                # NOTE(lihe): test not cat x0 information
                rgb_cat = rgb_in.permute(1,2,3,0) # [N_rays, n_samples, 3, N_views]
                rgb_cat = rgb_cat.reshape([*rgb_cat.shape[:2], -1]) # [N_rays, n_samples, 3*N_views]
                feats_map = torch.cat([feats_map, rgb_cat], dim=-1) # [N_rays, n_samples, c + 3*8]
                
                # z_features = torch.cat([pos_feats.view(-1, self.pos_code.d_out), dirs], dim=-1).unsqueeze(0)# [1, -1, d_out + 3]
                z_features = pos_feats.view(-1, self.pos_code.d_out).unsqueeze(0)# [1, -1, d_out]
                latent = torch.cat([feats_map.view(-1, feats_map.shape[-1]), ren_geo_feats.view(-1, ren_geo_feats.shape[-1])], dim=-1).unsqueeze(0) # [1, n_rays*n_samples, c+3x8 + geo_ch]
                latent = torch.cat([latent, z_features], dim=-1)
                # formulate resnetfc inputs
                # NOTE(lihe):debug
                sampled_color, rendering_valid_mask = self.rendering_network(latent, t, ren_mask,)
            else:
                raise NotImplementedError
            
            color = torch.zeros_like(tex_pos).view(-1, 3) - 1. # background color is -1.
            color[valid_mask] = sampled_color.view(-1, 3)
            color_list.append(color.view(-1, 3))

            if not self.debug_color:
                save_depth = depth.view(1, 1, self.img_resolution, self.img_resolution) * 0.5
                save_image(save_depth, 'debug/rast_depth.png')
                save_antilias_mask = antilias_mask.view(1, 1, self.img_resolution, self.img_resolution)
                save_image(save_antilias_mask, 'debug/rast_antilias_mask.png')
                depth_list.append(depth.view(-1, 1))
                antilias_mask_list.append(antilias_mask)
 
        color = torch.cat(color_list, dim=0) # [nv*res*res, 3]
        if not self.debug_color:
            depth = torch.cat(depth_list, dim=0) # [nv*res*res, 1]
            antilias_mask = torch.cat(antilias_mask_list, dim=0) # [nv*res*res, 1]
        
        buffers['color_fine'] = color

        # color gen
        if color_gen:
            assert query_pts is not None
            ren_geo_feats, ren_rgb_feats, ren_ray_diff, ren_mask, _, _ = self.rendering_projector.compute(
                    # tex_pos.view(-1, 1, 3),
                    query_pts.view(-1, 1, 3),
                    # * 3d geometry feature volumes
                    geometryVolume=volume[0],
                    geometryVolumeMask=conditional_valid_mask_volume[0],
                    # * 2d rendering feature maps
                    rendering_feature_maps=feature_maps,
                    color_maps=color_maps,
                    w2cs=w2cs,
                    intrinsics=intrinsics,
                    img_wh=img_wh,
                    query_img_idx=0,  # the index of the N_views dim for rendering
                    query_c2w=cam_mv[i:i+1], # query_c2w, # NOTE(lihe): we render multiple views at once for now
                    # NOTE(lihe): to reduce memory, feed voxel param to generate mask
                    vol_dims=vol_dims,
                    partial_vol_origin=partial_vol_origin,
                    vol_size=vol_size,
                )
            if self.pos_code is not None:
                # pos_feats = self.pos_code(tex_pos.view(-1, 3))
                pos_feats = self.pos_code(query_pts.view(-1, 3))
                pos_feats = pos_feats.view(-1, 1, self.pos_code.d_out)
                if not self.use_resnetfc:
                    ren_geo_feats = torch.cat([ren_geo_feats, pos_feats], dim=-1)
            if self.use_resnetfc:
                # NOTE(lihe): [N_views, N_rays, n_samples, 3+c]
                rgb_in = ren_rgb_feats[..., :self.rgb_ch] # [N_views, N_rays, n_samples, 3]
                feats_map = ren_rgb_feats[..., self.rgb_ch:]
                feats_map = feats_map.mean(dim=0) # [N_rays, n_samples, c]

                # NOTE(lihe): test not cat x0 information
                rgb_cat = rgb_in.permute(1,2,3,0) # [N_rays, n_samples, 3, N_views]
                rgb_cat = rgb_cat.reshape([*rgb_cat.shape[:2], -1]) # [N_rays, n_samples, 3*N_views]
                feats_map = torch.cat([feats_map, rgb_cat], dim=-1) # [N_rays, n_samples, c + 3*8]
                
                # z_features = torch.cat([pos_feats.view(-1, self.pos_code.d_out), dirs], dim=-1).unsqueeze(0)# [1, -1, d_out + 3]
                z_features = pos_feats.view(-1, self.pos_code.d_out).unsqueeze(0)# [1, -1, d_out]
                latent = torch.cat([feats_map.view(-1, feats_map.shape[-1]), ren_geo_feats.view(-1, ren_geo_feats.shape[-1])], dim=-1).unsqueeze(0) # [1, n_rays*n_samples, c+3x8 + geo_ch]
                latent = torch.cat([latent, z_features], dim=-1)
                # formulate resnetfc inputs
                # NOTE(lihe):debug
                sampled_color, rendering_valid_mask = self.rendering_network(latent, t, ren_mask,)
                query_colors = sampled_color.view(-1, 3)
                buffers['query_colors'] = query_colors
            else:
                raise NotImplementedError

        return buffers


    
    def get_geometry_prediction(self, volume):
        # Step 1: first get the sdf and deformation value for each vertices in the tetrahedon grid.
        sdf, deformation, sdf_reg_loss, weight = self.get_sdf_deformation_prediction(volume)
        v_deformed = self.dmtet_geometry.verts.unsqueeze(dim=0).expand(sdf.shape[0], -1, -1) + deformation
        tets = self.dmtet_geometry.indices
        n_batch = sdf.shape[0]
        v_list = []
        f_list = []
        flexicubes_surface_reg_list = []
        # Step 2: Using marching tet to obtain the mesh
        for i_batch in range(n_batch):
            if self.iso_surface == 'flexicubes':
                verts, faces, flexicubes_surface_reg = self.dmtet_geometry.get_mesh(
                v_deformed[i_batch], sdf[i_batch].squeeze(dim=-1),
                with_uv=False, indices=tets, weight_n=weight[i_batch].squeeze(dim=-1),
                is_training=self.training)     
                flexicubes_surface_reg_list.append(flexicubes_surface_reg)           
            elif self.iso_surface == 'dmtet':
                verts, faces = self.dmtet_geometry.get_mesh(
                v_deformed[i_batch], sdf[i_batch].squeeze(dim=-1),
                with_uv=False, indices=tets)
            v_list.append(verts)
            f_list.append(faces)
        if self.iso_surface == 'flexicubes':
            flexicubes_surface_reg = torch.cat(flexicubes_surface_reg_list).mean()
            flexicubes_weight_reg = (weight ** 2).mean()
        else:
            flexicubes_surface_reg, flexicubes_weight_reg = None, None
        return v_list, f_list, sdf, deformation, v_deformed, (sdf_reg_loss, flexicubes_surface_reg, flexicubes_weight_reg)

    def get_sdf_deformation_prediction(self, volume):
        ###
        weights = None
        # Step 1: first get the sdf and deformation value for each vertices in the tetrahedon grid.
        # init_position = self.dmtet_geometry.verts.unsqueeze(dim=0)
        init_position = self.dmtet_geometry.verts
        if self.iso_surface == 'dmtet':
            outs = self.sdf_def_network.get_sdf_def_prediction(init_position, volume, split_batch=None, flexicubes_indices=None)
            sdf = outs['sdf_pts']
            deformation = outs['def_pts']
        elif self.iso_surface == 'flexicubes':
            outs = self.sdf_def_network.get_sdf_def_prediction(init_position, volume, split_batch=None, flexicubes_indices=self.dmtet_geometry.indices)
            sdf = outs['sdf_pts']
            deformation = outs['def_pts']
            weights = outs['weights_pts']
        
        ###
        # Step 2: Normalize the deformation to avoid the flipped triangles.
        # NOTE(lihe): our AABB box is in [-0.35, +0.35]
        # deformation = 1.0 / (self.grid_res * self.deformation_multiplier) * torch.tanh(deformation)
        deformation = 0.7 / (self.grid_res * self.deformation_multiplier) * torch.tanh(deformation) # TODO(lihe): chech this hparam
        sdf = sdf.unsqueeze(0) # 1, N, 1
        deformation = deformation.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        sdf_reg_loss = torch.zeros(sdf.shape[0], device=sdf.device, dtype=torch.float32)

        ###
        # Step 3: Fix some sdf if we observe empty shape (full positive or full negative)
        if self.iso_surface == 'flexicubes':
            sdf_bxnxnxn = sdf.reshape((sdf.shape[0], self.grid_res + 1, self.grid_res + 1, self.grid_res + 1))
            sdf_less_boundary = sdf_bxnxnxn[:, 1:-1, 1:-1, 1:-1].reshape(sdf.shape[0], -1)
            pos_shape = torch.sum((sdf_less_boundary > 0).int(), dim=-1)
            neg_shape = torch.sum((sdf_less_boundary < 0).int(), dim=-1)
            zero_surface = torch.bitwise_or(pos_shape == 0, neg_shape == 0)
        else:
            pos_shape = torch.sum((sdf.squeeze(dim=-1) > 0).int(), dim=-1)
            neg_shape = torch.sum((sdf.squeeze(dim=-1) < 0).int(), dim=-1)
            zero_surface = torch.bitwise_or(pos_shape == 0, neg_shape == 0)
        if torch.sum(zero_surface).item() > 0:
            update_sdf = torch.zeros_like(sdf[0:1])
            max_sdf = sdf.max()
            min_sdf = sdf.min()
            update_sdf[:, self.dmtet_geometry.center_indices] += (1.0 - min_sdf)  # greater than zero
            update_sdf[:, self.dmtet_geometry.boundary_indices] += (-1 - max_sdf)  # smaller than zero
            new_sdf = torch.zeros_like(sdf)
            for i_batch in range(zero_surface.shape[0]):
                if zero_surface[i_batch]:
                    new_sdf[i_batch:i_batch + 1] += update_sdf
            update_mask = (new_sdf == 0).float()
            # Regulraization here is used to push the sdf to be a different sign (make it not fully positive or fully negative)
            sdf_reg_loss = torch.abs(sdf).mean(dim=-1).mean(dim=-1)
            sdf_reg_loss = sdf_reg_loss * zero_surface.float()
            sdf = sdf * update_mask + new_sdf * (1 - update_mask)
        
        # Step 4: Here we remove the gradient for the bad sdf (full positive or full negative)
        final_sdf = []
        final_def = []
        for i_batch in range(zero_surface.shape[0]):
            if zero_surface[i_batch]:
                final_sdf.append(sdf[i_batch: i_batch + 1].detach())
                final_def.append(deformation[i_batch: i_batch + 1].detach())
            else:
                final_sdf.append(sdf[i_batch: i_batch + 1])
                final_def.append(deformation[i_batch: i_batch + 1])
        sdf = torch.cat(final_sdf, dim=0)
        deformation = torch.cat(final_def, dim=0)
        sdf = torch.cat(final_sdf, dim=0)
        deformation = torch.cat(final_def, dim=0)
        return sdf, deformation, sdf_reg_loss, weights
            

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_variance,
                  conditional_valid_mask_volume=None):
        device, dtype = rays_o.device, rays_o.dtype
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3

        if conditional_valid_mask_volume is not None:
            pts_mask = self.get_pts_mask_for_conditional_volume(pts.view(-1, 3), conditional_valid_mask_volume)
            pts_mask = pts_mask.reshape(batch_size, n_samples)
            pts_mask = pts_mask[:, :-1] * pts_mask[:, 1:]  # [batch_size, n_samples-1]
        else:
            pts_mask = torch.ones([batch_size, n_samples], dtype=dtype).to(pts.device)

        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        dot_val = None
        if self.alpha_type == 'uniform':
            dot_val = torch.ones([batch_size, n_samples - 1], dtype=dtype) * -1.0
        else:
            dot_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
            prev_dot_val = torch.cat([torch.zeros([batch_size, 1], dtype=dtype).to(device), dot_val[:, :-1]], dim=-1)
            dot_val = torch.stack([prev_dot_val, dot_val], dim=-1)
            dot_val, _ = torch.min(dot_val, dim=-1, keepdim=False)
            dot_val = dot_val.clip(-10.0, 0.0) * pts_mask
        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - dot_val * dist * 0.5
        next_esti_sdf = mid_sdf + dot_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_variance)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_variance)
        alpha_sdf = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)

        alpha = alpha_sdf

        # - apply pts_mask
        alpha = pts_mask * alpha

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]).to(device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, lod,
                   sdf_network, gru_fusion,
                   # * related to conditional feature
                   conditional_volume=None,
                   conditional_valid_mask_volume=None,
                   emb=None,
                   ):
        device, dtype = rays_o.device, rays_o.dtype
        z_vals, new_z_vals = z_vals.to(dtype), new_z_vals.to(dtype)
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]

        if conditional_valid_mask_volume is not None:
            pts_mask = self.get_pts_mask_for_conditional_volume(pts.view(-1, 3), conditional_valid_mask_volume)
            pts_mask = pts_mask.reshape(batch_size, n_importance)
            pts_mask_bool = (pts_mask > 0).view(-1)
        else:
            pts_mask = torch.ones([batch_size, n_importance]).to(pts.device)

        new_sdf = torch.ones([batch_size * n_importance, 1], dtype=dtype).to(device) * 100

        if torch.sum(pts_mask) > 1:
            new_outputs = sdf_network.sdf(pts.reshape(-1, 3)[pts_mask_bool], conditional_volume, lod=lod, emb=emb)
            new_sdf[pts_mask_bool] = new_outputs['sdf_pts_scale%d' % lod]  # .reshape(batch_size, n_importance)

        new_sdf = new_sdf.view(batch_size, n_importance).to(pts.dtype)

        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        sdf = torch.cat([sdf, new_sdf], dim=-1)

        z_vals, index = torch.sort(z_vals, dim=-1)
        xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
        index = index.reshape(-1)
        sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    @torch.no_grad()
    def get_pts_mask_for_conditional_volume(self, pts, mask_volume):
        """

        :param pts: [N, 3]
        :param mask_volume: [1, 1, X, Y, Z]
        :return:
        """
        num_pts = pts.shape[0]
        pts = pts.view(1, 1, 1, num_pts, 3)  # - should be in range (-1, 1)

        pts = torch.flip(pts, dims=[-1])

        pts_mask = F.grid_sample(mask_volume, pts, mode='nearest')  # [1, c, 1, 1, num_pts]
        pts_mask = pts_mask.view(-1, num_pts).permute(1, 0).contiguous()  # [num_pts, 1]

        return pts_mask

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    lod,
                    sdf_network,
                    rendering_network,
                    background_alpha=None,  # - no use here
                    background_sampled_color=None,  # - no use here
                    background_rgb=None,  # - no use here
                    alpha_inter_ratio=0.0,
                    # * related to conditional feature
                    conditional_volume=None,
                    conditional_valid_mask_volume=None,
                    # * 2d feature maps
                    feature_maps=None,
                    color_maps=None,
                    w2cs=None,
                    intrinsics=None,
                    img_wh=None,
                    query_c2w=None,  # - used for testing
                    if_general_rendering=True,
                    if_render_with_grad=True,
                    # * used for blending mlp rendering network
                    img_index=None,
                    rays_uv=None,
                    # * used for clear bg and fg
                    bg_num=0,
                    # NOTE(lihe): add voxel param
                    vol_dims=None,
                    partial_vol_origin=None,
                    vol_size=None,
                    pred_density=False,
                    emb=None,
                    t=None,
                    noisy_latents_3d=None,
                    xm=None,
                    options_3d=None,
                    ):
        device = rays_o.device
        N_rays = rays_o.shape[0]
        _, n_samples = z_vals.shape
        dists = z_vals[..., 1:] - z_vals[..., :-1] # n_rays, n_samples - 1
        dists = torch.cat([dists, torch.tensor([sample_dist], dtype=dists.dtype).expand(dists[..., :1].shape).to(device)], -1)

        mid_z_vals = z_vals + dists * 0.5 # n_rays, n_samples
        mid_dists = mid_z_vals[..., 1:] - mid_z_vals[..., :-1]
        if pred_density:
            dist_inf = 1.8 - mid_z_vals[:, -1]
            mid_dists = torch.cat([mid_dists, dist_inf.unsqueeze(-1)], dim=-1) # n_rays, n_samples
        
        # predict background color
        if self.learn_bg_color:
            background_rgb = self.background
        # print("======background rgb======", background_rgb)

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        # NOTE(lihe): debug
        # if self.v_cont <= 7:
        #     np.save(f"debug/points_{self.v_cont}.npy", pts.detach().cpu().numpy())
        # else:
        #     exit()
        # self.v_cont += 1
        # print("=======exit========")
        # exit()

        # * if conditional_volume is restored from sparse volume, need mask for pts
        if conditional_valid_mask_volume is not None:
            # pts_mask = self.get_pts_mask_for_conditional_volume(pts, conditional_valid_mask_volume)
            pts_mask = self.get_pts_mask_for_conditional_volume(pts / 0.35, conditional_valid_mask_volume) # NOTE(lihe): we should scale the pts
            pts_mask = pts_mask.reshape(N_rays, n_samples).float().detach()
            pts_mask_bool = (pts_mask > 0).view(-1)
            # np.save("debug/pts_mask_bool.npy", pts_mask_bool.detach().cpu().numpy())


            if torch.sum(pts_mask_bool.float()) < 1:  # ! when render out image, may meet this problem
                pts_mask_bool[:100] = True

        else:
            pts_mask = torch.ones([N_rays, n_samples]).to(pts.device)

        # pts_valid = pts[pts_mask_bool]
        # sdf_nn_output = sdf_network.sdf(pts[pts_mask_bool], conditional_volume, lod=lod, split_batch=(48+32)*64*64)
        if not pred_density:
            # neus
            sdf_nn_output = sdf_network.sdf(pts[pts_mask_bool], conditional_volume, lod=lod, split_batch=None, emb=emb)

            sdf = torch.ones([N_rays * n_samples, 1]).to(pts.dtype).to(device) * 100
            sdf[pts_mask_bool] = sdf_nn_output['sdf_pts_scale%d' % lod]  # [N_rays*n_samples, 1]
            feature_vector_valid = sdf_nn_output['sdf_features_pts_scale%d' % lod]
            feature_vector = torch.zeros([N_rays * n_samples, feature_vector_valid.shape[1]]).to(pts.dtype).to(device)
            feature_vector[pts_mask_bool] = feature_vector_valid

            # * estimate alpha from sdf
            gradients = torch.zeros([N_rays * n_samples, 3]).to(pts.dtype).to(device)
            gradients[pts_mask_bool] = sdf_network.gradient(
                pts[pts_mask_bool], conditional_volume, lod=lod, emb=emb).squeeze()
            # gradients[pts_mask_bool] = sdf_network.gradient(
            #     pts[pts_mask_bool], conditional_volume, lod=lod).squeeze()
        else:
            # nerf
            sdf_nn_output = sdf_network.sdf(pts[pts_mask_bool], conditional_volume, lod=lod, split_batch=None, emb=emb)
            sigmas = torch.zeros([N_rays * n_samples, 1]).to(pts.dtype).to(device)
            sigmas[pts_mask_bool] = sdf_nn_output['sdf_pts_scale%d' % lod]  # [N_rays*n_samples, 1]
            # print("====debug sigmas sum mean=====", (sigmas <= 0).all())
            sdf = None

        sampled_color_mlp = None
        rendering_valid_mask_mlp = None
        sampled_color_patch = None
        rendering_patch_mask = None

        ######### no use for now ########
        if self.if_fitted_rendering:  # used for fine-tuning
            position_latent = sdf_nn_output['sampled_latent_scale%d' % lod]
            sampled_color_mlp = torch.zeros([N_rays * n_samples, 3]).to(pts.dtype).to(device)
            sampled_color_mlp_mask = torch.zeros([N_rays * n_samples, 1]).to(pts.dtype).to(device)

            # - extract pixel
            pts_pixel_color, pts_pixel_mask = self.patch_projector.pixel_warp(
                pts[pts_mask_bool][:, None, :], color_maps, intrinsics,
                w2cs, img_wh=None)  # [N_rays * n_samples,1, N_views,  3] , [N_rays*n_samples, 1, N_views]
            pts_pixel_color = pts_pixel_color[:, 0, :, :]  # [N_rays * n_samples, N_views,  3]
            pts_pixel_mask = pts_pixel_mask[:, 0, :]  # [N_rays*n_samples, N_views]

            # - extract patch
            if_patch_blending = False if rays_uv is None else True
            pts_patch_color, pts_patch_mask = None, None
            if if_patch_blending:
                pts_patch_color, pts_patch_mask = self.patch_projector.patch_warp(
                    pts.reshape([N_rays, n_samples, 3]),
                    rays_uv, gradients.reshape([N_rays, n_samples, 3]),
                    color_maps,
                    intrinsics[0], intrinsics,
                    query_c2w[0], torch.inverse(w2cs), img_wh=None
                )  # (N_rays, n_samples, N_src, Npx, 3), (N_rays, n_samples, N_src, Npx)
                N_src, Npx = pts_patch_mask.shape[2:]
                pts_patch_color = pts_patch_color.view(N_rays * n_samples, N_src, Npx, 3)[pts_mask_bool]
                pts_patch_mask = pts_patch_mask.view(N_rays * n_samples, N_src, Npx)[pts_mask_bool]

                sampled_color_patch = torch.zeros([N_rays * n_samples, Npx, 3]).to(device)
                sampled_color_patch_mask = torch.zeros([N_rays * n_samples, 1]).to(device)

            sampled_color_mlp_, sampled_color_mlp_mask_, \
            sampled_color_patch_, sampled_color_patch_mask_ = sdf_network.color_blend(
                pts[pts_mask_bool],
                position_latent,
                gradients[pts_mask_bool],
                dirs[pts_mask_bool],
                feature_vector[pts_mask_bool],
                img_index=img_index,
                pts_pixel_color=pts_pixel_color,
                pts_pixel_mask=pts_pixel_mask,
                pts_patch_color=pts_patch_color,
                pts_patch_mask=pts_patch_mask

            )  # [n, 3], [n, 1]
            sampled_color_mlp[pts_mask_bool] = sampled_color_mlp_
            sampled_color_mlp_mask[pts_mask_bool] = sampled_color_mlp_mask_.float()
            sampled_color_mlp = sampled_color_mlp.view(N_rays, n_samples, 3)
            sampled_color_mlp_mask = sampled_color_mlp_mask.view(N_rays, n_samples)
            rendering_valid_mask_mlp = torch.mean(pts_mask * sampled_color_mlp_mask, dim=-1, keepdim=True) > 0.5

            # patch blending
            if if_patch_blending:
                sampled_color_patch[pts_mask_bool] = sampled_color_patch_
                sampled_color_patch_mask[pts_mask_bool] = sampled_color_patch_mask_.float()
                sampled_color_patch = sampled_color_patch.view(N_rays, n_samples, Npx, 3)
                sampled_color_patch_mask = sampled_color_patch_mask.view(N_rays, n_samples)
                rendering_patch_mask = torch.mean(pts_mask * sampled_color_patch_mask, dim=-1,
                                                  keepdim=True) > 0.5  # [N_rays, 1]
            else:
                sampled_color_patch, rendering_patch_mask = None, None
        ######### no use for now ########

        # get attn mask
        if self.geo_attn_mask:
            # pdb.set_trace()
            from diffusers.models.sparse_neus.render_utils import get_attn_mask
            attn_mask = get_attn_mask(pts.view(N_rays, n_samples, 3).detach(), feature_maps.shape[0], w2cs, intrinsics, img_wh, step=1) # N_ray, N_v, H, W
            # print('===attn_mask===', attn_mask.shape)
            # debug_mask = attn_mask.sum(0).unsqueeze(1)
            # debug_mask = attn_mask[2000].unsqueeze(1)
            # from torchvision.utils import save_image
            # save_image(debug_mask.float(), 'debug/attn_mask.png', nrow=4)
            # pdb.set_trace()
        else:
            attn_mask = None

        if if_general_rendering:  # used for general training
            ren_geo_feats, ren_rgb_feats, ren_ray_diff, ren_mask, _, _ = self.rendering_projector.compute(
                pts.view(N_rays, n_samples, 3),
                # * 3d geometry feature volumes
                geometryVolume=conditional_volume[0],
                geometryVolumeMask=conditional_valid_mask_volume[0],
                # * 2d rendering feature maps
                rendering_feature_maps=feature_maps,
                color_maps=color_maps,
                w2cs=w2cs,
                intrinsics=intrinsics,
                img_wh=img_wh,
                query_img_idx=0,  # the index of the N_views dim for rendering
                query_c2w=query_c2w, # NOTE(lihe): we render multiple views at once for now
                # NOTE(lihe): to reduce memory, feed voxel param to generate mask
                vol_dims=vol_dims,
                partial_vol_origin=partial_vol_origin,
                vol_size=vol_size,
            )

            # add positional encoding
            if self.pos_code is not None:
                pos_feats = self.pos_code(pts.view(-1, 3))
                pos_feats = pos_feats.view(N_rays, n_samples, self.pos_code.d_out)
                if not self.use_resnetfc:
                    ren_geo_feats = torch.cat([ren_geo_feats, pos_feats], dim=-1)
            
            # prepare inputs
            if self.use_resnetfc:
                # NOTE(lihe): [N_views, N_rays, n_samples, 3+c]
                rgb_in = ren_rgb_feats[..., :self.rgb_ch] # [N_views, N_rays, n_samples, 3]
                feats_map = ren_rgb_feats[..., self.rgb_ch:]
                feats_map = feats_map.mean(dim=0) # [N_rays, n_samples, c]

                # NOTE(lihe): test not cat x0 information
                rgb_cat = rgb_in.permute(1,2,3,0) # [N_rays, n_samples, 3, N_views]
                rgb_cat = rgb_cat.reshape([*rgb_cat.shape[:2], -1]) # [N_rays, n_samples, 3*N_views]
                feats_map = torch.cat([feats_map, rgb_cat], dim=-1) # [N_rays, n_samples, c + 3*8]
                
                z_features = torch.cat([pos_feats.view(-1, self.pos_code.d_out), dirs], dim=-1).unsqueeze(0)# [1, -1, d_out + 3]
                latent = torch.cat([feats_map.view(-1, feats_map.shape[-1]), ren_geo_feats.view(-1, ren_geo_feats.shape[-1])], dim=-1).unsqueeze(0) # [1, n_rays*n_samples, c+3x8 + geo_ch]
                latent = torch.cat([latent, z_features], dim=-1)
                # formulate resnetfc inputs
                assert t is not None
                if is_torch_version(">=", "1.11.0"):
                    sampled_color, rendering_valid_mask = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(rendering_network), 
                        latent, t, ren_mask,
                        use_reentrant=False,
                    )
                    sampled_color = sampled_color[0].view(N_rays, n_samples, -1)
                else:
                    sampled_color, rendering_valid_mask = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(rendering_network), 
                        latent, t, ren_mask
                    )
                    sampled_color = sampled_color[0].view(N_rays, n_samples, -1)

            # (N_rays, n_samples, 3)
            if if_render_with_grad and not self.use_resnetfc:
                # sampled_color, rendering_valid_mask = rendering_network(
                #     ren_geo_feats, ren_rgb_feats, ren_ray_diff, ren_mask)

                # NOTE(lihe): use pts_mask to reduce memory
                # sampled_color, rendering_valid_mask = rendering_network(
                #     ren_geo_feats, ren_rgb_feats, ren_ray_diff, ren_mask, pts_mask=pts_mask_bool)
    
                if is_torch_version(">=", "1.11.0"):
                    sampled_color, rendering_valid_mask = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(rendering_network), 
                        ren_geo_feats, ren_rgb_feats, ren_ray_diff, ren_mask, pts_mask_bool, emb,
                        use_reentrant=False,
                    )
                else:
                    sampled_color, rendering_valid_mask = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(rendering_network), 
                        ren_geo_feats, ren_rgb_feats, ren_ray_diff, ren_mask, pts_mask_bool,
                        emb,
                    )
                
                # NOTE(lihe): split rays to multiple views to reduce mem
                # NOTE(lihe): bug!
                # rays_batch = ren_geo_feats.shape[0] // 8
                # sampled_color_list, rendering_valid_mask_list = [], []
                # for v_id in range(8):
                #     # print("====before rendering network======", v_id)
                #     # pdb.set_trace()
                #     sampled_color, rendering_valid_mask = rendering_network(
                #         ren_geo_feats[v_id*rays_batch : (v_id+1)*rays_batch], 
                #         ren_rgb_feats[:, v_id*rays_batch : (v_id+1)*rays_batch],
                #         ren_ray_diff[:, v_id*rays_batch : (v_id+1)*rays_batch],
                #         ren_mask[:, v_id*rays_batch : (v_id+1)*rays_batch])
                #     sampled_color_list.append(sampled_color)
                #     rendering_valid_mask_list.append(rendering_valid_mask)
                # sampled_color = torch.cat(sampled_color_list, dim=0)
                # rendering_valid_mask = torch.cat(rendering_valid_mask_list, dim=0)

            # else:
            #     with torch.no_grad():
            #         sampled_color, rendering_valid_mask = rendering_network(
            #             ren_geo_feats, ren_rgb_feats, ren_ray_diff, ren_mask)
        else:
            sampled_color, rendering_valid_mask = None, None

        if not pred_density:
            #### neus
            inv_variance = self.variance_network(feature_vector)[:, :1].clip(1e-6, 1e6)
            
            # print("====gradients mean=====", gradients.mean())

            true_dot_val = (dirs * gradients).sum(-1, keepdim=True)  # * calculate

            iter_cos = -(F.relu(-true_dot_val * 0.5 + 0.5) * (1.0 - alpha_inter_ratio) + F.relu(
                -true_dot_val) * alpha_inter_ratio)  # always non-positive
            
            # print("====iter_cos mean=====", iter_cos.mean())

            iter_cos = iter_cos * pts_mask.view(-1, 1)

            true_estimate_sdf_half_next = sdf + iter_cos.clip(-10.0, 10.0) * dists.reshape(-1, 1) * 0.5
            true_estimate_sdf_half_prev = sdf - iter_cos.clip(-10.0, 10.0) * dists.reshape(-1, 1) * 0.5

            prev_cdf = torch.sigmoid(true_estimate_sdf_half_prev * inv_variance)
            next_cdf = torch.sigmoid(true_estimate_sdf_half_next * inv_variance)

            p = prev_cdf - next_cdf
            c = prev_cdf

            if self.alpha_type == 'div':
                alpha_sdf = ((p + 1e-5) / (c + 1e-5)).reshape(N_rays, n_samples).clip(0.0, 1.0)
            elif self.alpha_type == 'uniform':
                uniform_estimate_sdf_half_next = sdf - dists.reshape(-1, 1) * 0.5
                uniform_estimate_sdf_half_prev = sdf + dists.reshape(-1, 1) * 0.5
                uniform_prev_cdf = torch.sigmoid(uniform_estimate_sdf_half_prev * inv_variance)
                uniform_next_cdf = torch.sigmoid(uniform_estimate_sdf_half_next * inv_variance)
                uniform_alpha = F.relu(
                    (uniform_prev_cdf - uniform_next_cdf + 1e-5) / (uniform_prev_cdf + 1e-5)).reshape(
                    N_rays, n_samples).clip(0.0, 1.0)
                alpha_sdf = uniform_alpha
            else:
                assert False

            alpha = alpha_sdf
        else:
            ### nerf
            alpha = 1 - torch.exp(-mid_dists.view(-1, 1) * torch.relu(sigmas)) # n_rays*n_samples, 1
            alpha = alpha.reshape(N_rays, n_samples)
            # print("====debug alpha sum mean=====", alpha.sum(), alpha.mean())
        
        # NOTE(lihe): save pts and sdf to debug
        # print("====SAVING points for process sdf data====")
        # np.save('debug/save_real_scale_pts.npy', pts.detach().cpu().numpy())
        # np.save('debug/save_real_scale_sdf.npy', sdf.detach().cpu().numpy())

        # NOTE(lihe): test use 3d prior directly
        # if self.direct_use_3d and t[0] >= 700 and self.use_3d_prior:
        # if True and t[0] >= 700 and self.use_3d_prior:
        # if True and self.use_3d_prior:
        if False:
            print(f"=====directly use shap-e at timestep {t[0]}========")
            assert options_3d is not None and xm is not None, "please also set options and renderer when injecting 3d prior."
            from diffusers.models.shap_e.shap_e.models.query import Query
            query = Query(
                position=pts / 0.35,
                direction=None,
            )
            # get params
            with torch.no_grad():
                dist_inf = 1.8 - mid_z_vals[:, -1]
                mid_dists = torch.cat([mid_dists, dist_inf.unsqueeze(-1)], dim=-1) # n_rays, n_samples
                # print("============ render 3d prior =============")
                params = xm.encoder.bottleneck_to_params(
                    noisy_latents_3d[None]
                )
                raw = xm.renderer._query(query=query, params=params, options=options_3d)
                density = raw.density
                # sdf = raw.signed_distance
                sampled_color = raw.channels.view(N_rays, n_samples, -1)
                sampled_color = sampled_color * 2 - 1.

                alpha = 1 - torch.exp(-mid_dists.view(-1, 1) * density) # n_rays*n_samples, 1
                alpha = alpha.reshape(N_rays, n_samples)

        # - apply pts_mask
        alpha = alpha * pts_mask

        # NOTE(lihe): debug
        # print("=====debug... saving alpha...====")
        # print("=====debug alpha mean======", alpha.mean())
        # np.save('debug/alpha.npy', alpha.detach().cpu().numpy())

        # pts_radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(N_rays, n_samples)
        # inside_sphere = (pts_radius < 1.0).float().detach()
        # relax_inside_sphere = (pts_radius < 1.2).float().detach()
        inside_sphere = pts_mask
        relax_inside_sphere = pts_mask

        weights = alpha * torch.cumprod(torch.cat([torch.ones([N_rays, 1]).to(device), 1. - alpha + 1e-7], -1), -1)[:,
                          :-1]  # n_rays, n_samples

        weights_sum = weights.sum(dim=-1, keepdim=True)
        alpha_sum = alpha.sum(dim=-1, keepdim=True)

        if bg_num > 0:
            weights_sum_fg = weights[:, :-bg_num].sum(dim=-1, keepdim=True)
        else:
            weights_sum_fg = weights_sum
        if sampled_color is not None:
            color = (sampled_color * weights[:, :, None]).sum(dim=1)
        else:
            color = None

        if background_rgb is not None and color is not None:
            color = color + background_rgb * (1.0 - weights_sum)

        ###################*  mlp color rendering  #####################
        color_mlp = None
        if sampled_color_mlp is not None:
            color_mlp = (sampled_color_mlp * weights[:, :, None]).sum(dim=1)

        if background_rgb is not None and color_mlp is not None:
            color_mlp = color_mlp + background_rgb * (1.0 - weights_sum)

        ############################ *  patch blending  ################
        blended_color_patch = None
        if sampled_color_patch is not None:
            blended_color_patch = (sampled_color_patch * weights[:, :, None, None]).sum(dim=1)  # [N_rays, Npx, 3]

        ######################################################

        if not pred_density:
            gradient_error = (torch.linalg.norm(gradients.reshape(N_rays, n_samples, 3), ord=2,
                                                dim=-1) - 1.0) ** 2
            # ! the gradient normal should be masked out, the pts out of the bounding box should also be penalized
            gradient_error = (pts_mask * gradient_error).sum() / (
                    (pts_mask).sum() + 1e-5)
        else:
            gradient_error = 0.

        depth = (mid_z_vals * weights[:, :n_samples]).sum(dim=1, keepdim=True)
        # print("====debug sampled color mean====", sampled_color.mean())
        # print("====debug sampled color min max====", sampled_color.min(), sampled_color.max())
        # print("=====debug depth====", depth.sum(), depth.mean())
        # print("====depth=====", depth)
        # print('====weighted sum=====', (weights_sum > 0).sum())
        # print('====weighted max=====', weigh––––ts_sum.max())
        if False:
            save_depth = depth.view(-1, 1, 64, 64) / 2.
            save_weights_sum = weights_sum.view(-1, 1, 64, 64)
            from torchvision.utils import save_image
            save_image(save_depth, 'debug/debug_sample/debug_depth_.png', nrow=4)
            save_image(save_weights_sum, 'debug/debug_sample/debug_weights_.png', nrow=4)
            save_image((save_weights_sum > 0).float(), 'debug/debug_sample/debug_weights_mask.png', nrow=4)

        return {
            'color': color,
            'color_mask': rendering_valid_mask,  # (N_rays, 1)
            'color_mlp': color_mlp,
            'color_mlp_mask': rendering_valid_mask_mlp,
            'sdf': sdf,  # (N_rays, n_samples)
            'depth': depth,  # (N_rays, 1)
            'dists': dists,
            'gradients': gradients.reshape(N_rays, n_samples, 3) if not pred_density else None,
            'variance': 1.0 / inv_variance if not pred_density else None,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'weights_sum': weights_sum,
            'alpha_sum': alpha_sum,
            'alpha_mean': alpha.mean(),
            'cdf': c.reshape(N_rays, n_samples) if not pred_density else None,
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
            'blended_color_patch': blended_color_patch,
            'blended_color_patch_mask': rendering_patch_mask,
            'weights_sum_fg': weights_sum_fg,
            'attn_mask': attn_mask
        }

    def render(self, rays_o, rays_d, near, far, sdf_network, rendering_network,
               perturb_overwrite=-1,
               background_rgb=None,
               alpha_inter_ratio=0.0,
               # * related to conditional feature
               lod=None,
               conditional_volume=None,
               conditional_valid_mask_volume=None,
               # * 2d feature maps
               feature_maps=None,
               color_maps=None,
               w2cs=None,
               intrinsics=None,
               img_wh=None,
               query_c2w=None,  # -used for testing
               if_general_rendering=True,
               if_render_with_grad=True,
               # * used for blending mlp rendering network
               img_index=None,
               rays_uv=None,
               # * importance sample for second lod network
               pre_sample=False,  # no use here
               # * for clear foreground
               bg_ratio=0.0,
               vol_dims=None,
               partial_vol_origin=None,
               vol_size=None,
               pred_density=False,
               emb=None,
               t=None,
               noisy_latents_3d=None,
               xm=None,
               options_3d=None,
               ):
        device = rays_o.device
        N_rays = len(rays_o)
        # sample_dist = 2.0 / self.n_samples
        sample_dist = ((far - near) / self.n_samples).mean().item()
        z_vals = torch.linspace(0.0, 1.0, self.n_samples).to(device)
        z_vals = near + (far - near) * z_vals[None, :]

        bg_num = int(self.n_samples * bg_ratio)

        if z_vals.shape[0] == 1:
            z_vals = z_vals.repeat(N_rays, 1)

        if bg_num > 0:
            z_vals_bg = z_vals[:, self.n_samples - bg_num:]
            z_vals = z_vals[:, :self.n_samples - bg_num]

        n_samples = self.n_samples - bg_num
        perturb = self.perturb

        # - significantly speed up training, for the second lod network
        if pre_sample:
            z_vals = self.sample_z_vals_from_maskVolume(rays_o, rays_d, near, far,
                                                        conditional_valid_mask_volume).to(rays_o.dtype)

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, dtype=z_vals.dtype).to(device)
            z_vals = lower + (upper - lower) * t_rand

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if pred_density:
            assert self.n_importance == 0, "we dont support fine stage sampling for nerf now."
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]

                sdf_outputs = sdf_network.sdf(
                    pts.reshape(-1, 3), conditional_volume, lod=lod, emb=emb)
                # pdb.set_trace()
                sdf = sdf_outputs['sdf_pts_scale%d' % lod].reshape(N_rays, self.n_samples - bg_num)

                n_steps = 4
                for i in range(n_steps):
                    new_z_vals = self.up_sample(rays_o, rays_d, z_vals, sdf, self.n_importance // n_steps,
                                                64 * 2 ** i,
                                                conditional_valid_mask_volume=conditional_valid_mask_volume,
                                                )
                    # print('\nDEBUG 0:', z_vals.dtype, new_z_vals.dtype)
                    z_vals, sdf = self.cat_z_vals(
                        rays_o, rays_d, z_vals, new_z_vals, sdf, lod,
                        sdf_network, gru_fusion=False,
                        conditional_volume=conditional_volume,
                        conditional_valid_mask_volume=conditional_valid_mask_volume,
                        emb=emb,
                    )
                    # print('\nDEBUG 1:', z_vals.dtype)
                del sdf

            n_samples = self.n_samples + self.n_importance

        # Background
        ret_outside = None

        # Render
        if bg_num > 0:
            z_vals = torch.cat([z_vals, z_vals_bg], dim=1)
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    lod,
                                    sdf_network,
                                    rendering_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    alpha_inter_ratio=alpha_inter_ratio,
                                    # * related to conditional feature
                                    conditional_volume=conditional_volume,
                                    conditional_valid_mask_volume=conditional_valid_mask_volume,
                                    # * 2d feature maps
                                    feature_maps=feature_maps,
                                    color_maps=color_maps,
                                    w2cs=w2cs,
                                    intrinsics=intrinsics,
                                    img_wh=img_wh,
                                    query_c2w=query_c2w,
                                    if_general_rendering=if_general_rendering,
                                    if_render_with_grad=if_render_with_grad,
                                    # * used for blending mlp rendering network
                                    img_index=img_index,
                                    rays_uv=rays_uv,
                                    # NOTE(lihe): add voxel param
                                    vol_dims=vol_dims,
                                    partial_vol_origin=partial_vol_origin,
                                    vol_size=vol_size,
                                    pred_density=pred_density,
                                    emb=emb,
                                    t=t,
                                    noisy_latents_3d=noisy_latents_3d,
                                    xm=xm,
                                    options_3d=options_3d,
                                    )

        color_fine = ret_fine['color']

        if self.n_outside > 0:
            color_fine_mask = torch.logical_or(ret_fine['color_mask'], ret_outside['color_mask'])
        else:
            color_fine_mask = ret_fine['color_mask']

        # weights = ret_fine['weights']
        # weights_sum = ret_fine['weights_sum']

        # gradients = ret_fine['gradients']
        # mid_z_vals = ret_fine['mid_z_vals']

        # depth = (mid_z_vals * weights[:, :n_samples]).sum(dim=1, keepdim=True)
        depth = ret_fine['depth']
        # depth_varaince = ((mid_z_vals - depth) ** 2 * weights[:, :n_samples]).sum(dim=-1, keepdim=True)
        # variance = ret_fine['variance'].reshape(N_rays, n_samples).mean(dim=-1, keepdim=True)

        # - randomly sample points from the volume, and maximize the sdf
        # TODO(lihe): move out
        # NOTE(lihe): done
        # pts_random = torch.rand([1024, 3]).float().to(device) * 2 - 1  # normalized to (-1, 1)
        # sdf_random = sdf_network.sdf(pts_random, conditional_volume, lod=lod)['sdf_pts_scale%d' % lod]

        # NOTE(lihe): reformulate the output dict
        result = {
            'color_fine': color_fine,
            'color_fine_mask': color_fine_mask,
            'depth': depth,
            'gradient_error_fine': ret_fine['gradient_error'],
            'sdf': ret_fine['sdf'],
            'attn_mask': ret_fine['attn_mask']
        }

        # result = {
        #     'depth': depth,
        #     'color_fine': color_fine,
        #     'color_fine_mask': color_fine_mask,
        #     'color_outside': ret_outside['color'] if ret_outside is not None else None,
        #     'color_outside_mask': ret_outside['color_mask'] if ret_outside is not None else None,
        #     'color_mlp': ret_fine['color_mlp'],
        #     'color_mlp_mask': ret_fine['color_mlp_mask'],
        #     'variance': variance.mean(),
        #     'cdf_fine': ret_fine['cdf'],
        #     'depth_variance': depth_varaince,
        #     'weights_sum': weights_sum,
        #     'weights_max': torch.max(weights, dim=-1, keepdim=True)[0],
        #     'alpha_sum': ret_fine['alpha_sum'].mean(),
        #     'alpha_mean': ret_fine['alpha_mean'],
        #     'gradients': gradients,
        #     'weights': weights,
        #     'gradient_error_fine': ret_fine['gradient_error'],
        #     'inside_sphere': ret_fine['inside_sphere'],
        #     'sdf': ret_fine['sdf'],
        #     'sdf_random': sdf_random,
        #     'blended_color_patch': ret_fine['blended_color_patch'],
        #     'blended_color_patch_mask': ret_fine['blended_color_patch_mask'],
        #     'weights_sum_fg': ret_fine['weights_sum_fg']
        # }

        return result

    @torch.no_grad()
    def sample_z_vals_from_sdfVolume(self, rays_o, rays_d, near, far, sdf_volume, mask_volume):
        # ? based on sdf to do importance sampling, seems that too biased on pre-estimation
        device = rays_o.device
        N_rays = len(rays_o)
        n_samples = self.n_samples * 2

        z_vals = torch.linspace(0.0, 1.0, n_samples).to(device)
        z_vals = near + (far - near) * z_vals[None, :]

        if z_vals.shape[0] == 1:
            z_vals = z_vals.repeat(N_rays, 1)

        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]

        sdf = self.get_pts_mask_for_conditional_volume(pts.view(-1, 3), sdf_volume).reshape([N_rays, n_samples])

        new_z_vals = self.up_sample(rays_o, rays_d, z_vals, sdf, self.n_samples,
                                    200,
                                    conditional_valid_mask_volume=mask_volume,
                                    )
        return new_z_vals

    @torch.no_grad()
    def sample_z_vals_from_maskVolume(self, rays_o, rays_d, near, far, mask_volume):  # don't use
        device = rays_o.device
        N_rays = len(rays_o)
        n_samples = self.n_samples * 2

        z_vals = torch.linspace(0.0, 1.0, n_samples).to(device)
        z_vals = near + (far - near) * z_vals[None, :]

        if z_vals.shape[0] == 1:
            z_vals = z_vals.repeat(N_rays, 1)

        mid_z_vals = (z_vals[:, 1:] + z_vals[:, :-1]) * 0.5

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]

        pts_mask = self.get_pts_mask_for_conditional_volume(pts.view(-1, 3), mask_volume).reshape(
            [N_rays, n_samples - 1])

        # empty voxel set to 0.1, non-empty voxel set to 1
        weights = torch.where(pts_mask > 0, torch.ones_like(pts_mask).to(device),
                              0.1 * torch.ones_like(pts_mask).to(device))

        # sample more pts in non-empty voxels
        z_samples = sample_pdf(z_vals, weights, self.n_samples, det=True).detach()
        return z_samples

    @torch.no_grad()
    def filter_pts_by_depthmaps(self, coords, pred_depth_maps, proj_matrices,
                                partial_vol_origin, voxel_size,
                                near, far, depth_interval, d_plane_nums):
        """
        Use the pred_depthmaps to remove redundant pts (pruned by sdf, sdf always have two sides, the back side is useless)
        :param coords: [n, 3]  int coords
        :param pred_depth_maps: [N_views, 1, h, w]
        :param proj_matrices: [N_views, 4, 4]
        :param partial_vol_origin: [3]
        :param voxel_size: 1
        :param near: 1
        :param far: 1
        :param depth_interval: 1
        :param d_plane_nums: 1
        :return:
        """
        device = pred_depth_maps.device
        n_views, _, sizeH, sizeW = pred_depth_maps.shape

        if len(partial_vol_origin.shape) == 1:
            partial_vol_origin = partial_vol_origin[None, :]
        pts = coords * voxel_size + partial_vol_origin

        rs_grid = pts.unsqueeze(0).expand(n_views, -1, -1)
        rs_grid = rs_grid.permute(0, 2, 1).contiguous()  # [n_views, 3, n_pts]
        nV = rs_grid.shape[-1]
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).to(device)], dim=1)  # [n_views, 4, n_pts]

        # Project grid
        im_p = proj_matrices @ rs_grid  # - transform world pts to image UV space   # [n_views, 4, n_pts]
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
        im_x = im_x / im_z
        im_y = im_y / im_z

        im_grid = torch.stack([2 * im_x / (sizeW - 1) - 1, 2 * im_y / (sizeH - 1) - 1], dim=-1)

        im_grid = im_grid.view(n_views, 1, -1, 2)
        sampled_depths = torch.nn.functional.grid_sample(pred_depth_maps, im_grid, mode='bilinear',
                                                         padding_mode='zeros',
                                                         align_corners=True)[:, 0, 0, :]  # [n_views, n_pts]
        sampled_depths_valid = (sampled_depths > 0.5 * near).float()
        valid_d_min = (sampled_depths - d_plane_nums * depth_interval).clamp(near.item(),
                                                                             far.item()) * sampled_depths_valid
        valid_d_max = (sampled_depths + d_plane_nums * depth_interval).clamp(near.item(),
                                                                             far.item()) * sampled_depths_valid

        mask = im_grid.abs() <= 1
        mask = mask[:, 0]  # [n_views, n_pts, 2]
        mask = (mask.sum(dim=-1) == 2) & (im_z > valid_d_min) & (im_z < valid_d_max)

        mask = mask.view(n_views, -1)
        mask = mask.permute(1, 0).contiguous()  # [num_pts, nviews]

        mask_final = torch.sum(mask.float(), dim=1, keepdim=False) > 0

        return mask_final

    @torch.no_grad()
    def get_valid_sparse_coords_by_sdf_depthfilter(self, sdf_volume, coords_volume, mask_volume, feature_volume,
                                                   pred_depth_maps, proj_matrices,
                                                   partial_vol_origin, voxel_size,
                                                   near, far, depth_interval, d_plane_nums,
                                                   threshold=0.02, maximum_pts=110000):
        """
        assume batch size == 1, from the first lod to get sparse voxels
        :param sdf_volume: [1, X, Y, Z]
        :param coords_volume: [3, X, Y, Z]
        :param mask_volume: [1, X, Y, Z]
        :param feature_volume: [C, X, Y, Z]
        :param threshold:
        :return:
        """
        device = coords_volume.device
        _, dX, dY, dZ = coords_volume.shape

        def prune(sdf_pts, coords_pts, mask_volume, threshold):
            occupancy_mask = (torch.abs(sdf_pts) < threshold).squeeze(1)  # [num_pts]
            valid_coords = coords_pts[occupancy_mask]

            # - filter backside surface by depth maps
            mask_filtered = self.filter_pts_by_depthmaps(valid_coords, pred_depth_maps, proj_matrices,
                                                         partial_vol_origin, voxel_size,
                                                         near, far, depth_interval, d_plane_nums)
            valid_coords = valid_coords[mask_filtered]

            # - dilate
            occupancy_mask = sparse_to_dense_channel(valid_coords, 1, [dX, dY, dZ], 1, 0, device)  # [dX, dY, dZ, 1]

            # - dilate
            occupancy_mask = occupancy_mask.float()
            occupancy_mask = occupancy_mask.view(1, 1, dX, dY, dZ)
            occupancy_mask = F.avg_pool3d(occupancy_mask, kernel_size=7, stride=1, padding=3)
            occupancy_mask = occupancy_mask.view(-1, 1) > 0

            final_mask = torch.logical_and(mask_volume, occupancy_mask)[:, 0]  # [num_pts]

            return final_mask, torch.sum(final_mask.float())

        C, dX, dY, dZ = feature_volume.shape
        sdf_volume = sdf_volume.permute(1, 2, 3, 0).contiguous().view(-1, 1)
        coords_volume = coords_volume.permute(1, 2, 3, 0).contiguous().view(-1, 3)
        mask_volume = mask_volume.permute(1, 2, 3, 0).contiguous().view(-1, 1)
        feature_volume = feature_volume.permute(1, 2, 3, 0).contiguous().view(-1, C)

        # - for check
        # sdf_volume = torch.rand_like(sdf_volume).float().to(sdf_volume.device) * 0.02

        final_mask, valid_num = prune(sdf_volume, coords_volume, mask_volume, threshold)

        while (valid_num > maximum_pts) and (threshold > 0.003):
            threshold = threshold - 0.002
            final_mask, valid_num = prune(sdf_volume, coords_volume, mask_volume, threshold)

        valid_coords = coords_volume[final_mask]  # [N, 3]
        valid_feature = feature_volume[final_mask]  # [N, C]

        valid_coords = torch.cat([torch.ones([valid_coords.shape[0], 1]).to(valid_coords.device) * 0,
                                  valid_coords], dim=1)  # [N, 4], append batch idx

        # ! if the valid_num is still larger than maximum_pts, sample part of pts
        if valid_num > maximum_pts:
            valid_num = valid_num.long()
            occupancy = torch.ones([valid_num]).to(device) > 0
            choice = np.random.choice(valid_num.cpu().numpy(), valid_num.cpu().numpy() - maximum_pts,
                                      replace=False)
            ind = torch.nonzero(occupancy).to(device)
            occupancy[ind[choice]] = False
            valid_coords = valid_coords[occupancy]
            valid_feature = valid_feature[occupancy]

            # print(threshold, "randomly sample to save memory")

        return valid_coords, valid_feature

    @torch.no_grad()
    def get_valid_sparse_coords_by_sdf(self, sdf_volume, coords_volume, mask_volume, feature_volume, threshold=0.02,
                                       maximum_pts=110000):
        """
        assume batch size == 1, from the first lod to get sparse voxels
        :param sdf_volume: [num_pts, 1]
        :param coords_volume: [3, X, Y, Z]
        :param mask_volume: [1, X, Y, Z]
        :param feature_volume: [C, X, Y, Z]
        :param threshold:
        :return:
        """

        def prune(sdf_volume, mask_volume, threshold):
            occupancy_mask = torch.abs(sdf_volume) < threshold  # [num_pts, 1]

            # - dilate
            occupancy_mask = occupancy_mask.float()
            occupancy_mask = occupancy_mask.view(1, 1, dX, dY, dZ)
            occupancy_mask = F.avg_pool3d(occupancy_mask, kernel_size=7, stride=1, padding=3)
            occupancy_mask = occupancy_mask.view(-1, 1) > 0

            final_mask = torch.logical_and(mask_volume, occupancy_mask)[:, 0]  # [num_pts]

            return final_mask, torch.sum(final_mask.float())

        C, dX, dY, dZ = feature_volume.shape
        coords_volume = coords_volume.permute(1, 2, 3, 0).contiguous().view(-1, 3)
        mask_volume = mask_volume.permute(1, 2, 3, 0).contiguous().view(-1, 1)
        feature_volume = feature_volume.permute(1, 2, 3, 0).contiguous().view(-1, C)

        final_mask, valid_num = prune(sdf_volume, mask_volume, threshold)

        while (valid_num > maximum_pts) and (threshold > 0.003):
            threshold = threshold - 0.002
            final_mask, valid_num = prune(sdf_volume, mask_volume, threshold)

        valid_coords = coords_volume[final_mask]  # [N, 3]
        valid_feature = feature_volume[final_mask]  # [N, C]

        valid_coords = torch.cat([torch.ones([valid_coords.shape[0], 1]).to(valid_coords.device) * 0,
                                  valid_coords], dim=1)  # [N, 4], append batch idx

        # ! if the valid_num is still larger than maximum_pts, sample part of pts
        if valid_num > maximum_pts:
            device = sdf_volume.device
            valid_num = valid_num.long()
            occupancy = torch.ones([valid_num]).to(device) > 0
            choice = np.random.choice(valid_num.cpu().numpy(), valid_num.cpu().numpy() - maximum_pts,
                                      replace=False)
            ind = torch.nonzero(occupancy).to(device)
            occupancy[ind[choice]] = False
            valid_coords = valid_coords[occupancy]
            valid_feature = valid_feature[occupancy]

            # print(threshold, "randomly sample to save memory")

        return valid_coords, valid_feature

    @torch.no_grad()
    def extract_fields(self, bound_min, bound_max, resolution, query_func, device,
                       # * related to conditional feature
                       **kwargs
                       ):
        N = 64
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)

                        # ! attention, the query function is different for extract geometry and fields
                        output = query_func(pts, **kwargs)
                        sdf = output['sdf_pts_scale%d' % kwargs['lod']].reshape(len(xs), len(ys),
                                                                                len(zs)).detach().cpu().numpy()

                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = -1 * sdf
        return u

    @torch.no_grad()
    def extract_geometry(self, u, bound_min, bound_max, resolution, threshold, device, occupancy_mask=None,
                         # * 3d feature volume
                         **kwargs
                         ):
        # logging.info('threshold: {}'.format(threshold))

        # u = self.extract_fields(bound_min, bound_max, resolution,
        #                         lambda pts, **kwargs: sdf_network.sdf(pts, **kwargs),
        #                         # - sdf need to be multiplied by -1
        #                         device,
        #                         # * 3d feature volume
        #                         **kwargs
        #                         )
        if occupancy_mask is not None:
            dX, dY, dZ = occupancy_mask.shape
            empty_mask = 1 - occupancy_mask
            empty_mask = empty_mask.view(1, 1, dX, dY, dZ)
            # - dilation
            # empty_mask = F.avg_pool3d(empty_mask, kernel_size=7, stride=1, padding=3)
            empty_mask = F.interpolate(empty_mask, [resolution, resolution, resolution], mode='nearest')
            empty_mask = empty_mask.view(resolution, resolution, resolution).cpu().numpy() > 0
            u[empty_mask] = -100
            del empty_mask

        vertices, triangles = mcubes.marching_cubes(u, threshold)
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        return vertices, triangles, u

    @torch.no_grad()
    def extract_depth_maps(self, sdf_network, con_volume, intrinsics, c2ws, H, W, near, far):
        """
        extract depth maps from the density volume
        :param con_volume: [1, 1+C, dX, dY, dZ]  can by con_volume or sdf_volume
        :param c2ws: [B, 4, 4]
        :param H:
        :param W:
        :param near:
        :param far:
        :return:
        """
        device = con_volume.device
        batch_size = intrinsics.shape[0]

        with torch.no_grad():
            ys, xs = torch.meshgrid(torch.linspace(0, H - 1, H),
                                    torch.linspace(0, W - 1, W))  # pytorch's meshgrid has indexing='ij'
            p = torch.stack([xs, ys, torch.ones_like(ys)], dim=-1)  # H, W, 3

            intrinsics_inv = torch.inverse(intrinsics)

            p = p.view(-1, 3).float().to(device)  # N_rays, 3
            p = torch.matmul(intrinsics_inv[:, None, :3, :3], p[:, :, None]).squeeze()  # Batch, N_rays, 3
            rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # Batch, N_rays, 3
            rays_v = torch.matmul(c2ws[:, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # Batch, N_rays, 3
            rays_o = c2ws[:, None, :3, 3].expand(rays_v.shape)  # Batch, N_rays, 3
            rays_d = rays_v

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        ################## - sphere tracer to extract depth maps               ######################
        depth_masks_sphere, depth_maps_sphere = self.ray_tracer.extract_depth_maps(
            rays_o, rays_d,
            near[None, :].repeat(rays_o.shape[0], 1),
            far[None, :].repeat(rays_o.shape[0], 1),
            sdf_network, con_volume
        )

        depth_maps = depth_maps_sphere.view(batch_size, 1, H, W)
        depth_masks = depth_masks_sphere.view(batch_size, 1, H, W)

        depth_maps = torch.where(depth_masks, depth_maps,
                                 torch.zeros_like(depth_masks.float()).to(device))  # fill invalid pixels by 0

        return depth_maps, depth_masks
