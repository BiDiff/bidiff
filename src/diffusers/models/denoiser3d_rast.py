import torch
# ! amazing!!!! autograd.grad with set_detect_anomaly(True) will cause memory leak
# ! https://github.com/pytorch/pytorch/issues/51349
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from inplace_abn import InPlaceABN
from diffusers.models.sparse_neus.rendering_network import GeneralRenderingNetwork
from diffusers.models.sparse_neus.conv_modules import ConvBnReLU
# mvrast
from diffusers.models.sparse_neus.latent_volume_network import LatentVolumeNetwork
from diffusers.models.sparse_neus.sparse_dmtet_renderer import SparseDMTetRenderer
import pdb
from torchvision.utils import save_image
import trimesh
import os
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from PIL import Image
import imageio
from typing import Any, Callable, Dict, Optional
import kaolin as kal
from diffusers.models.get3d.uni_rep import flex_render
from diffusers.models.shap_e.shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

DEBUG_ID=0

def beta_linear_log_snr(t):
    return -torch.log(torch.special.expm1(1e-4 + 10 * (t ** 2)))

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

# Regulrarization loss for dmtet
def sdf_reg_loss_batch(sdf, all_edges):
    sdf_f1x6x2 = sdf[:, all_edges.reshape(-1)].reshape(sdf.shape[0], -1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()) + \
               torch.nn.functional.binary_cross_entropy_with_logits(
                   sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff

def uncond_guide_model_x0(
    model: Callable[..., torch.Tensor], scale: float
) -> Callable[..., torch.Tensor]:
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2] # [1, C]
        C = half.shape[1]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs) # [2, C] -> [2, 2C]
        eps, rest = model_out[:, :C], model_out[:, C:] # [2, C], [2, C]
        cond_eps, uncond_eps = torch.chunk(eps, 2, dim=0) # [2, C] -> [1, C], [1, C]
        half_eps = uncond_eps + scale * (cond_eps - uncond_eps) # [1, C]
        return half_eps # [2, C]
    return model_fn

def render_latents(xm, latent, render_mode = 'nerf', size = 64, device=None, name='train'):
    cameras = create_pan_cameras(size, device)
    images = decode_latent_images(xm, latent[0], cameras, rendering_mode=render_mode)
    # display(gif_widget(images))
    images[0].save(f'./debug/{name}_latent_denoise_img.gif', save_all=True, append_images=images[1:], duration=100, loop=0)

class DepthLoss(nn.Module):
    def __init__(self, type='l1'):
        super(DepthLoss, self).__init__()
        self.type = type

    def forward(self, depth_pred, depth_gt, mask):
        mask_d = (depth_gt > 0).float()

        mask = mask * mask_d

        mask_sum = mask.sum() + 1e-5

        depth_error = (depth_pred - depth_gt) * mask
        depth_loss = F.l1_loss(depth_error, torch.zeros_like(depth_error).to(depth_error.device),
                               reduction='sum') / mask_sum

        return depth_loss

# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars, device=None):
        if isinstance(vars, list):
            return [wrapper(x, device) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x, device) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v, device) for k, v in vars.items()}
        else:
            return func(vars, device)

    return wrapper

@make_recursive_func
def numpy2tensor(vars, device='cpu'):
    if not isinstance(vars, torch.Tensor) and vars is not None :
        return torch.tensor(vars, device=device)
    elif isinstance(vars, torch.Tensor):
        return vars
    elif vars is None:
        return vars
    else:
        raise NotImplementedError("invalid input type {} for float2tensor".format(type(vars)))

def ratio_threshold(depth1, depth2, threshold):
    """
    Computes the percentage of pixels for which the ratio of the two depth maps is less than a given threshold.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns:
        percentage of pixels with ratio less than the threshold

    """
    assert (threshold > 0.)
    assert (np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 >= 0) & (depth2 >= 0)))
    log_diff = np.log(depth1) - np.log(depth2)
    num_pixels = float(log_diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return float(np.sum(np.absolute(log_diff) < np.log(threshold))) / num_pixels

def compute_depth_errors(depth_pred, depth_gt, valid_mask):
    """
    Computes different distance measures between two depth maps.

    depth_pred:           depth map prediction
    depth_gt:             depth map ground truth
    distances_to_compute: which distances to compute

    Returns:
        a dictionary with computed distances, and the number of valid pixels

    """
    depth_pred = depth_pred[valid_mask]
    depth_gt = depth_gt[valid_mask]
    num_valid = np.sum(valid_mask)

    distances_to_compute = ['l1',
                            'l1_inverse',
                            'scale_invariant',
                            'abs_relative',
                            'sq_relative',
                            'avg_log10',
                            'rmse_log',
                            'rmse',
                            'ratio_threshold_1.25',
                            'ratio_threshold_1.5625',
                            'ratio_threshold_1.953125']

    results = {'num_valid': num_valid}
    for dist in distances_to_compute:
        if dist.startswith('ratio_threshold'):
            threshold = float(dist.split('_')[-1])
            results[dist] = ratio_threshold(depth_pred, depth_gt, threshold)
        else:
            results[dist] = globals()[dist](depth_pred, depth_gt)

    return results

# feature net for Stable Diffusion
class SDFeatureNet(nn.Module):
    """
    intergrate pyramid sd feats
    """
    def __init__(self, dims=[2816, 2816, 2112, 1408, 704], out_chs=[8, 8, 4, 4], scales=[8, 4, 2, 1, 1]):
        super(SDFeatureNet, self).__init__()
        self.dims = dims
        self.out_chs = out_chs
        self.scales = scales
        map_layers = []
        for dim in dims:
            layer = nn.Conv2d(dim, 32, 1)
            map_layers.append(layer)
        self.map_layers = nn.ModuleList(map_layers)
        smooth_layers = []
        for dim in out_chs:
            layer = nn.Conv2d(32, dim, 3, padding=1)
            smooth_layers.append(layer)
        self.smooth_layers = nn.ModuleList(smooth_layers)
    
    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2,
                             mode="bilinear", align_corners=True) + y
    
    def forward(self, feats,):
        out_list = []
        for i, feat in enumerate(feats):
            if i==0:
                out = self.map_layers[i](feat)
                out_list.append(self.smooth_layers[i](out))
            else:
                if feat.shape[-1] != out.shape[-1]:
                    out = self._upsample_add(out, self.map_layers[i](feat))
                else:
                    out = out + self.map_layers[i](feat) # the last layer
                if i == len(feats) - 1:
                    out_list.append(out) # 32
                else:
                    out_list.append(self.smooth_layers[i](out))
        return out_list
    
    def obtain_pyramid_feature_maps(self, imgs):
        """
        get feature maps of all conditional images
        :param imgs:
        :return:
        """
        pyramid_feature_maps = self(imgs)

        # * the pyramid features are very important, if only use the coarst features, hard to optimize

        feats_list = []
        for i, feats in enumerate(pyramid_feature_maps):
            if self.scales[i] != 1:
                feats_list.append(F.interpolate(feats, scale_factor=self.scales[i], mode='bilinear', align_corners=True))
            else:
                feats_list.append(feats)
        fused_feature_maps = torch.cat(
            feats_list, dim=1)

        return fused_feature_maps # (B, 56, H, W)
            
# feature net for DeepFloyd
class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """

    def __init__(self, norm_act=InPlaceABN, img_ch=3, use_featurenet_view_embed=False, num_views=8, view_embed_dim=32):
        super(FeatureNet, self).__init__()
        self.img_ch = img_ch
        self.use_featurenet_view_embed = use_featurenet_view_embed
        if use_featurenet_view_embed:
            self.view_embed_dim = view_embed_dim
            self.num_views = num_views
            self.view_embed = nn.Parameter(torch.randn(num_views, view_embed_dim, 1, 1))
            img_ch = img_ch + view_embed_dim

        self.conv0 = nn.Sequential(
            ConvBnReLU(img_ch, 8, 3, 1, 1, norm_act=norm_act),
            ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))

        self.conv1 = nn.Sequential(
            ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
            ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
            ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))

        self.conv2 = nn.Sequential(
            ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
            ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
            ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        # to reduce channel size of the outputs from FPN
        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 8, 3, padding=1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2,
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        bv, c, h, w = x.shape
        assert c == self.img_ch
        if self.use_featurenet_view_embed:
            view_embed = self.view_embed.to(dtype=x.dtype).repeat(bv//self.num_views, 1, h, w)
            x = torch.cat((x, view_embed), dim=1)

        conv0 = self.conv0(x)  # (B, 8, H, W)
        conv1 = self.conv1(conv0)  # (B, 16, H//2, W//2)
        conv2 = self.conv2(conv1)  # (B, 32, H//4, W//4)
        feat2 = self.toplayer(conv2)  # (B, 32, H//4, W//4)
        feat1 = self._upsample_add(feat2, self.lat1(conv1))  # (B, 32, H//2, W//2)
        feat0 = self._upsample_add(feat1, self.lat0(conv0))  # (B, 32, H, W)

        # reduce output channels
        feat1 = self.smooth1(feat1)  # (B, 16, H//2, W//2)
        feat0 = self.smooth0(feat0)  # (B, 8, H, W)

        return [feat2, feat1, feat0]  # coarser to finer features
    
    def obtain_pyramid_feature_maps(self, imgs):
        """
        get feature maps of all conditional images
        :param imgs:
        :return:
        """
        pyramid_feature_maps = self(imgs)

        # * the pyramid features are very important, if only use the coarst features, hard to optimize
        fused_feature_maps = torch.cat([
            F.interpolate(pyramid_feature_maps[0], scale_factor=4, mode='bilinear', align_corners=True),
            F.interpolate(pyramid_feature_maps[1], scale_factor=2, mode='bilinear', align_corners=True),
            pyramid_feature_maps[2]
        ], dim=1)

        return fused_feature_maps # (B, 56, H, W)

class Denoiser3DV2Rast(nn.Module):
    """
    3D Denoiser v2 using Differentiable Rasterization
    """
    def __init__(self, ch=56, res=64, num_views=8, time_embed_ch=512, use_viewdirs=False, img_ch=3,
                 voxel_size=0.7/95., vol_dims=[96, 96, 96], hidden_dim=128, cost_type='variance_mean', # sparsesdfnet
                 d_pyramid_feature_compress=16, regnet_d_out=16, num_sdf_layers=4, multires=6, # 
                 init_val=0.2, # variance
                 in_geometry_feat_ch=16, in_rendering_feat_ch=56, anti_alias_pooling=True, # rendering network
                 n_samples=48, n_importance=32, n_outside=0, perturb=1.0, alpha_type='div',
                 partial_vol_origin=[-0.35, -0.35, -0.35], scale=0.35, pred_density=False,
                 add_temb=True, # True, 
                 temb_channels=320, encoder_hid_dim=4096,
                 regress_rgb=False, foundation_model='if', learn_bg_color=False,
                 pos_enc=False, debug_sd_feat=False, blend_x0=False,
                 extra_view_num=0, disable_in_color_loss=False,
                 abandon_sdf_x0=False,
                 debug_regress=False,
                 use_resnetfc=False,
                 use_3d_prior=False, 
                 device=None,
                 model_type='text300M',
                 direct_use_3d=False,
                 lazy_3d=False,
                 lazy_t=None,
                 new_sdf_arc=False,
                 sdf_gen=False,
                 voxel_cond=False,
                 use_featurenet_view_embed=False,
                 iso_surface='flexicubes', # 'flexicubes' or 'dmtet',
                 tet_res=64,
                 img_resolution=1024,
                 input_res=64,
                 render_res=512,
                 ) -> None:
        super(Denoiser3DV2Rast, self).__init__()
        # 1. build pyramid featurenet, we now only implement one stage
        self.featurenet = FeatureNet(img_ch=img_ch * 2 if not abandon_sdf_x0 else img_ch, use_featurenet_view_embed=use_featurenet_view_embed)
        # self.fuse_layer = nn.Conv2d(ch*2 + img_ch*2 if foundation_model=='if' else ch*2 + img_ch, ch, 1) # if pred mean and var
        # self.fuse_layer = nn.Conv2d(ch + img_ch, ch, 1) # if pred mean and var

        if pos_enc:
            from diffusers.models.sparse_neus.pos_encoder import PositionalEncoding
            self.code = PositionalEncoding(num_freqs=6, d_in=3, freq_factor=1.5, include_input=True)
        else:
            self.code = None
        # abandon_sdf_x0
        self.abandon_sdf_x0 = abandon_sdf_x0
        self.debug_regress = debug_regress
        
        # 2. build latent volume (previous build sdf network)
        print(f"======== Initialize Differentiable Mesh Representation: {iso_surface} ========")
        self.sdf_def_network = LatentVolumeNetwork(ch_in=ch, voxel_size = voxel_size, vol_dims = vol_dims, 
                                            hidden_dim = hidden_dim, cost_type = cost_type, d_pyramid_feature_compress = d_pyramid_feature_compress, 
                                            regnet_d_out = regnet_d_out, num_sdf_layers = num_sdf_layers, multires = multires,
                                            add_temb=add_temb, # NOTE: we need to add temb 
                                            temb_channels=temb_channels,
                                            use_3d_prior=use_3d_prior,
                                            new_sdf_arc=new_sdf_arc,
                                            sdf_gen=sdf_gen,
                                            voxel_cond=voxel_cond,
                                            iso_surface=iso_surface,)
        if use_resnetfc:
            from .resnetfc import ResnetFC
            self.rendering_network = ResnetFC(d_out=img_ch, d_in=self.code.d_out, # NOTE(lihe): not use diretions # self.code.d_out + 3,
                                              n_blocks=3, 
                                              d_latent=img_ch * 8 + in_rendering_feat_ch + in_geometry_feat_ch,
                                              d_hidden=64, time_embed_ch=64)
        else:
            self.rendering_network = GeneralRenderingNetwork(in_geometry_feat_ch=in_geometry_feat_ch, 
                                                            # NOTE(lihe): only feed predicted x0 to rendering network
                                                            in_rendering_feat_ch=in_rendering_feat_ch,
                                                            anti_alias_pooling=anti_alias_pooling, 
                                                            add_temb=add_temb, # TODO(lihe): ablate this
                                                            in_rgb_ch=img_ch,
                                                            regress_rgb=regress_rgb,
                                                            pos_enc_dim=0 if not pos_enc else self.code.d_out,
                                                            debug_regress=debug_regress
                                                            )
        # self.rendering_network.enable_gradient_checkpointing()
        # 3. build rasterize renderer
        self.iso_surface = iso_surface
        self.img_resolution = img_resolution
        self.dmtet_renderer = SparseDMTetRenderer(
            self.sdf_def_network,
            self.rendering_network,
            pos_code=self.code,
            use_resnetfc=use_resnetfc,
            iso_surface=iso_surface, # 'flexicubes' by default
            img_resolution=img_resolution,
            tet_res=tet_res,
        )
        # NOTE(lihe): add time embed
        self.add_temb = add_temb
        if add_temb:
            self.time_proj = Timesteps(temb_channels, True, 0)
            self.time_embedding = TimestepEmbedding(
                        in_channels=temb_channels,
                        time_embed_dim=temb_channels,
                        act_fn="gelu",
                        post_act_fn=None,
                        cond_proj_dim=None,
                        )
            self.add_embedding = TextTimeEmbedding(
                            encoder_hid_dim, temb_channels, num_heads=16,
                        )
        
        # NOTE(lihe): add 3d prior
        if use_3d_prior:
            from diffusers.models.shap_e.shap_e.models.download import load_config, load_model
            from diffusers.models.shap_e.shap_e.diffusion.gaussian_diffusion import diffusion_from_config
            from diffusers.models.shap_e.shap_e.util.collections import AttrDict
            assert device is not None
            self.ddpm_3d = diffusion_from_config(load_config('diffusion'))
            self.xm = load_model('transmitter', device=device)
            self.xm.requires_grad_(False)
            self.options_3d = AttrDict(rendering_mode="nerf", render_with_direction=False)
        else:
            self.ddpm_3d = None
            self.xm = None
            self.options_3d = None
        
        self.use_3d_prior = use_3d_prior
        self.partial_vol_origin = partial_vol_origin
        self.voxel_size = voxel_size
        self.vol_dims = vol_dims
        self.scale = scale

        query_pts = np.load('cache/grid_batch.npy')
        query_pts = torch.from_numpy(query_pts).float()
        self.query_pts = query_pts
        self.log_snr = beta_linear_log_snr
        self.log_snr_to_alpha_sigma = log_snr_to_alpha_sigma
        
        # loss weight
        self.fg_bg_weight = 0.
        self.anneal_start = 0
        self.anneal_end = 25000
        self.sdf_sparse_weight = 0.02
        self.sdf_igr_weight = 0.1
        self.sdf_decay_param = 100
        self.pred_density = pred_density

        self.regress_rgb = regress_rgb
        self.img_ch = img_ch
        self.foundation_model = foundation_model
        self.learn_bg_color = learn_bg_color
        self.pos_enc = pos_enc
        self.debug_sd_feat = debug_sd_feat
        if self.learn_bg_color:
            assert self.foundation_model == 'sd', 'only latent diffusion need to regress color'
        self.blend_x0 = blend_x0
        self.extra_view_num = extra_view_num
        self.disable_in_color_loss = disable_in_color_loss
        self.use_resnetfc = use_resnetfc
        self.model_type = model_type
        self.lazy_3d = lazy_3d
        self.lazy_t = lazy_t
        if self.lazy_3d:
            assert lazy_t is not None
        self.sdf_gen = sdf_gen
        self.voxel_cond = voxel_cond

        self.input_res = input_res
        assert input_res in [64, 128, 256, 512], "input res should be in [64, 128, 256, 512]"
        
        self.render_res = render_res

    @torch.no_grad()
    def validate_mesh(self, density_or_sdf_network, func_extract_geometry, world_space=True, resolution=128,
                      threshold=0.0, mode='val',
                      # * 3d feature volume
                      conditional_volume=None, lod=None, occupancy_mask=None,
                      bound_min=[-1, -1, -1], bound_max=[1, 1, 1], meta='', iter_step=0, scale_mat=None,
                      trans_mat=None, emb=None, save_path=None
                      ):

        bound_min = torch.tensor(bound_min, dtype=torch.float32)
        bound_max = torch.tensor(bound_max, dtype=torch.float32)

        vertices, triangles, fields = func_extract_geometry(
            density_or_sdf_network,
            bound_min, bound_max, resolution=resolution,
            threshold=threshold, device=conditional_volume.device,
            # * 3d feature volume
            conditional_volume=conditional_volume, lod=lod,
            occupancy_mask=occupancy_mask,
            emb=emb,
        )

        if scale_mat is not None:
            scale_mat_np = scale_mat.cpu().numpy()
            vertices = vertices * scale_mat_np[0][0, 0] + scale_mat_np[0][:3, 3][None]

        if trans_mat is not None:
            trans_mat_np = trans_mat.cpu().numpy()
            vertices_homo = np.concatenate([vertices, np.ones_like(vertices[:, :1])], axis=1)
            vertices = np.matmul(trans_mat_np, vertices_homo[:, :, None])[:, :3, 0]

        mesh = trimesh.Trimesh(vertices, triangles)
        if save_path is None:
            save_path = os.path.join('debug', 'meshes_' + mode, 'mesh_save.ply')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # mesh.export(os.path.join('debug', 'meshes_' + mode,
        #                          'mesh_{:0>8d}_{}_lod{:0>1d}.ply'.format(iter_step, meta, lod)))
        mesh.export(save_path)
    
    def cal_losses_rgb(self, buffers, iter_step, sample_rays):
        # only compute color loss for input views
        true_rgb = sample_rays['mv_images'][0]# nv, 3, h, w
        true_depth = sample_rays['mv_depths'][0] # nv, h, w
        object_mask = true_depth.view(-1) > 0
        c, res = true_rgb.shape[1], true_rgb.shape[-1]
        true_rgb = true_rgb.permute(0,2,3,1).reshape([-1, c]) # nv*h*w, 3
        pred_color = buffers['color_fine']
        
        color_loss = ((pred_color - true_rgb) * object_mask.unsqueeze(1)).abs().mean()
        # visualization during training
        if iter_step % 1000 == 0:
            save_gt_color = (true_rgb + 1) / 2
            save_gt_color = save_gt_color.view(-1, res, res, 3).permute(0, 3, 1, 2)
            save_pred_color = (pred_color + 1) / 2
            save_pred_color = save_pred_color.view(-1, res, res, 3).permute(0, 3, 1, 2)
            save_color = torch.cat([save_gt_color, save_pred_color], dim=0)
            save_image(save_color, f'debug/rast_train_color_given_{DEBUG_ID}.png', nrow=4)
        
        losses = {'color_loss': color_loss.detach().item()}
        return color_loss, losses
    
    def cal_losses_rast_flexi(self, buffers, target, iter_step, sample_rays=None):
        prefix = ''
        mask_loss = (buffers['mask'] - target['mask']).abs().mean()
        depth_loss = (((((buffers['depth'] - (target['depth']))* target['mask'])**2).sum(-1)+1e-8)).sqrt().mean() * 10
        sdf_reg_loss = buffers['sdf_reg_loss']
        sdf = buffers['sdf']
        if self.iso_surface == 'flexicubes':
            edge_sdf_reg_loss, flexicubes_surface_reg, flexicubes_weights_reg = sdf_reg_loss
            sdf_reg_loss = edge_sdf_reg_loss + 0.5 * flexicubes_surface_reg + 0.1 * flexicubes_weights_reg
        if sdf is not None:
            sdf_reg_loss_entropy = sdf_reg_loss_batch(sdf, self.dmtet_renderer.dmtet_geometry.all_edges).mean() * 0.01
        else:
            sdf_reg_loss_entropy = 0.
        
        loss = mask_loss + depth_loss + sdf_reg_loss + sdf_reg_loss_entropy

        # visualization
        if iter_step % 1000 == 0:
            gt_mask = target['mask'].view(-1, 1, self.img_resolution, self.img_resolution)
            gt_depth = target['depth'].view(-1, 4, self.img_resolution, self.img_resolution)[:, :3, :, :]
            pred_mask = buffers['mask'].view(-1, 1, self.img_resolution, self.img_resolution)
            pred_depth = buffers['depth'].view(-1, 4, self.img_resolution, self.img_resolution)[:, :3, :, :]
            save_image(torch.cat([gt_mask, pred_mask], dim=0), f'debug/rast_train_mask{DEBUG_ID}.png', nrow=4)
            
        if buffers.get('pred_normal', None) is not None:
            pred_norm = buffers['pred_normal']
            gt_norm = buffers['gt_normal']
            # norm_loss = (((((pred_norm - gt_norm)* target['mask'])**2).sum(-1)+1e-8)).sqrt().mean() * 10
            norm_loss = ((pred_norm - gt_norm) * target['mask']).abs().mean()
            loss = loss + norm_loss
        else:
            norm_loss = 0.
        
        if buffers.get('color_fine', None) is not None:
            true_rgb = sample_rays['extra_mv_images'][0, :self.extra_view_num]# nv, 3, h, w
            true_depth = sample_rays['extra_mv_depths'][0, :self.extra_view_num] # nv, h, w
            object_mask = true_depth.view(-1) > 0
            c, res = true_rgb.shape[1], true_rgb.shape[-1]
            true_rgb = true_rgb.permute(0,2,3,1).reshape([-1, c]) # nv*h*w, 3
            pred_color = buffers['color_fine']
            # color_loss = ((((pred_color - true_rgb) * object_mask.unsqueeze(1))**2).sum(-1)+1e-8).sqrt().mean() * 10
            color_loss = ((pred_color - true_rgb) * object_mask.unsqueeze(1)).abs().mean()
            loss = loss + color_loss
            if iter_step % 1000 == 0:
                save_gt_color = (true_rgb + 1) / 2
                save_gt_color = save_gt_color.view(-1, res, res, 3).permute(0, 3, 1, 2)
                save_pred_color = (pred_color + 1) / 2
                save_pred_color = save_pred_color.view(-1, res, res, 3).permute(0, 3, 1, 2)
                save_color = torch.cat([save_gt_color, save_pred_color], dim=0)
                save_image(save_color, f'debug/rast_train_color{DEBUG_ID}.png', nrow=4)
        else:
            color_loss = 0.
        
        losses = {
            prefix + "mv_loss": loss.detach().item(),
            prefix + "color_loss": color_loss.detach().item() if isinstance(color_loss, torch.Tensor) else color_loss,
            prefix + "dep_loss": depth_loss.detach().item() if isinstance(depth_loss, torch.Tensor) else depth_loss,
            prefix + "sdf_reg_loss": sdf_reg_loss.detach().item() if isinstance(sdf_reg_loss, torch.Tensor) else sdf_reg_loss,
            prefix + "sdf_reg_loss_entropy": sdf_reg_loss_entropy.detach().item() if isinstance(sdf_reg_loss_entropy, torch.Tensor) else sdf_reg_loss_entropy,
            prefix + "mask_loss": mask_loss.detach().item() if isinstance(mask_loss, torch.Tensor) else mask_loss,
            prefix + "norm_loss": norm_loss.detach().item() if isinstance(norm_loss, torch.Tensor) else norm_loss,
        }

        return loss, losses
    
    def cal_losses_rast(self, render_out, sample_rays, iter_step=-1, 
                        bs_id=0, vis_iter=200, 
                        noisy_input=None, extra_view=False,):
        
        if render_out == {}:
            return 0., {}
        
        prefix = '' if not extra_view else 'e_'
        if self.foundation_model == 'if':
            true_rgb = sample_rays['mv_images'][bs_id] if not extra_view else sample_rays['extra_mv_images'][bs_id, :self.extra_view_num]# nv, 3, h, w
        else:
            true_rgb = sample_rays['latent_images'][bs_id] if not extra_view else sample_rays['extra_latent_images'][bs_id, :self.extra_view_num]# nv, 3, h, w
        c, res = true_rgb.shape[1], true_rgb.shape[-1]
        true_rgb = true_rgb.permute(0,2,3,1).reshape([-1, c]) # nv*h*w, 3

        if 'mv_depths' in sample_rays.keys():
            # true_depth = sample_rays['rays_depth'][0]
            true_depth = sample_rays['mv_depths'][bs_id] if not extra_view else sample_rays['extra_mv_depths'][bs_id, :self.extra_view_num] # nv, h, w
            true_depth = true_depth.reshape([-1, 1]) * 2. # nv*h*w, 1 # NOTE(lihe): rescale depth to canonical space
        else:
            true_depth = None
        
        # color loss
        color_fine = render_out.get('color_fine', None)
        depth_pred = render_out.get('depth', None)

        if color_fine is not None:
            if self.foundation_model == 'sd':
                assert not self.learn_bg_color
                object_mask = true_depth.view(-1) > 0
                true_rgb[~object_mask] = -1.
                # color_error = (color_fine[object_mask] - true_rgb[object_mask])
                color_error = (color_fine - true_rgb)
            else:
                object_mask = None
                color_error = (color_fine - true_rgb)
            if not extra_view and self.disable_in_color_loss:
                color_fine_loss = torch.tensor(0., device=color_error.device) # TODO(lihe): ablate this
            else:
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error).to(color_error.device),
                                            reduction='mean')
        else:
            color_fine_loss = 0.
        
        # save pred imgs
        if iter_step % vis_iter == 0 and color_fine is not None:
            with torch.no_grad():
                print(prefix + "======saving gt and perd imgs======" )
                gt_imgs = (true_rgb + 1) / 2. # (nv*h*w, 3)
                gt_imgs = gt_imgs.view(-1, res, res, c).permute(0, 3, 1, 2)
                pred_imgs = (color_fine + 1) / 2.
                pred_imgs = pred_imgs.view(-1, res, res, c).permute(0, 3, 1, 2)
                if not extra_view:
                    noisy_input = noisy_input[bs_id]
                    noisy_input = (noisy_input + 1) / 2.
                    noisy_imgs = noisy_input.view(*pred_imgs.shape)
                    save_imgs = torch.cat([gt_imgs, noisy_imgs, pred_imgs], dim=0)
                else:
                    save_imgs = torch.cat([gt_imgs, pred_imgs], dim=0)
                # cat noisy input
                save_image(save_imgs[:, :3], 'debug/rast_train_imgs.png' if not extra_view else 'debug/rast_train_imgs_extra.png', nrow=4 if not extra_view else self.extra_view_num)

        # depth loss
        if depth_pred is not None:
            depth_error = depth_pred - true_depth
            depth_loss = F.l1_loss(depth_error,
                                       torch.zeros_like(depth_error).to(depth_error.device),
                                       reduction='mean')
            # save pred imgs
            if iter_step % vis_iter == 0:
                with torch.no_grad():
                    print(prefix + "======saving gt and perd deps======")
                    gt_dep = true_depth * 0.5 # (nv*h*w, 1)
                    gt_dep = gt_dep.view(-1, 1, res, res)
                    gt_dep_mask = gt_dep
                    pred_dep = depth_pred * 0.5
                    pred_dep = pred_dep.view(-1, 1, res, res)
                    pred_dep_mask = pred_dep
                    save_imgs = torch.cat([gt_dep, pred_dep, gt_dep_mask, pred_dep_mask], dim=0)
                    save_image(save_imgs, 'debug/rast_train_deps.png' if not extra_view else 'debug/rast_train_deps_extra.png', nrow=4 if not extra_view else self.extra_view_num)
        else:
            depth_loss = 0.
        
        ### antilias_mask
        if render_out.get('antilias_mask', None) is not None:
            antilias_mask = render_out['antilias_mask'].view(-1, 1, res, res)
            gt_antilias_mask = (true_depth > 0).float().view(-1, 1, res, res)
            save_imgs = torch.cat([gt_antilias_mask, antilias_mask], dim=0)
            save_image(save_imgs, 'debug/rast_train_mask.png',  nrow=4 if not extra_view else self.extra_view_num)
            mask_error = antilias_mask.view(-1) - gt_antilias_mask.view(-1)
            mask_loss = F.l1_loss(mask_error,
                                       torch.zeros_like(mask_error).to(mask_error.device),
                                       reduction='mean')
        else:
            mask_loss = 0.
        
        # reg loss
        sdf_reg_loss = render_out['sdf_reg_loss']
        if self.iso_surface == 'flexicubes':
            edge_sdf_reg_loss, flexicubes_surface_reg, flexicubes_weights_reg = sdf_reg_loss
            sdf_reg_loss = edge_sdf_reg_loss + 0.5 * flexicubes_surface_reg + 0.1 * flexicubes_weights_reg
        sdf = render_out['sdf']

        if sdf is not None:
            sdf_reg_loss_entropy = sdf_reg_loss_batch(sdf, self.dmtet_renderer.dmtet_geometry.all_edges).mean() * 0.01
        else:
            sdf_reg_loss_entropy = 0.
        
        loss = color_fine_loss + depth_loss + sdf_reg_loss + sdf_reg_loss_entropy + mask_loss
        
        losses = {
            prefix + "mv_loss": loss.detach().item(),
            prefix + "color_loss": color_fine_loss.detach().item() if isinstance(color_fine_loss, torch.Tensor) else color_fine_loss,
            prefix + "dep_loss": depth_loss.detach().item() if isinstance(depth_loss, torch.Tensor) else depth_loss,
            prefix + "sdf_reg_loss": sdf_reg_loss.detach().item() if isinstance(sdf_reg_loss, torch.Tensor) else sdf_reg_loss,
            prefix + "sdf_reg_loss_entropy": sdf_reg_loss_entropy.detach().item() if isinstance(sdf_reg_loss_entropy, torch.Tensor) else sdf_reg_loss_entropy,
            prefix + "mask_loss": mask_loss.detach().item() if isinstance(mask_loss, torch.Tensor) else mask_loss,
        }

        return loss, losses
    
    def cal_losses_sdf(self, render_out, sample_rays, iter_step=-1, lod=0, bs_id=0, vis_iter=200, noisy_input=None, extra_view=False):

        # loss weight schedule; the regularization terms should be added in later training stage
        def get_weight(iter_step, weight):
            assert lod == 0, "we now only support lod==0"
            if lod == 1:
                anneal_start = self.anneal_end if lod == 0 else self.anneal_end_lod1
                anneal_end = self.anneal_end if lod == 0 else self.anneal_end_lod1
                anneal_end = anneal_end * 2
            else:
                anneal_start = self.anneal_start if lod == 0 else self.anneal_start_lod1
                anneal_end = self.anneal_end if lod == 0 else self.anneal_end_lod1
                anneal_end = anneal_end * 2

            if iter_step < 0:
                return weight

            if anneal_end == 0.0:
                return weight
            elif iter_step < anneal_start:
                return 0.0
            else:
                return np.min(
                    [1.0,
                     (iter_step - anneal_start) / (anneal_end - anneal_start)]) * weight
        
        prefix = '' if not extra_view else 'e_'

        rays_o = sample_rays['rays_o'][bs_id] if not extra_view else sample_rays['extra_rays_o'][bs_id]
        rays_d = sample_rays['rays_d'][bs_id] if not extra_view else sample_rays['extra_rays_o'][bs_id]
        if self.foundation_model == 'if':
            true_rgb = sample_rays['mv_images'][bs_id] if not extra_view else sample_rays['extra_mv_images'][bs_id, :self.extra_view_num]# nv, 3, h, w
        else:
            true_rgb = sample_rays['latent_images'][bs_id] if not extra_view else sample_rays['extra_latent_images'][bs_id, :self.extra_view_num]# nv, 3, h, w

        c, res = true_rgb.shape[1], true_rgb.shape[-1]
        true_rgb = true_rgb.permute(0,2,3,1).reshape([-1, c]) # nv*h*w, 3

        if 'mv_depths' in sample_rays.keys():
            true_depth = sample_rays['mv_depths'][bs_id] if not extra_view else sample_rays['extra_mv_depths'][bs_id, :self.extra_view_num] # nv, h, w
            true_depth = true_depth.reshape([-1, 1]) * 2. # nv*h*w, 1 # NOTE(lihe): rescale depth to canonical space
        else:
            true_depth = None

        # TODO(lihe): add rays mask in dataset
        if 'rays_mask' in sample_rays:
            mask = sample_rays['rays_mask'][bs_id]
        else:
            mask = torch.ones(rays_o.shape[0]).to(rays_o.device)

        color_fine = render_out.get('color_fine', None)
        color_fine_mask = render_out.get('color_fine_mask', None)
        depth_pred = render_out.get('depth', None)

        gradient_error_fine = render_out.get('gradient_error_fine', torch.tensor(0., device=rays_o.device))

        # * color generated by mlp
        color_mlp = render_out.get('color_mlp', None)
        color_mlp_mask = render_out.get('color_mlp_mask', None)

        if color_fine is not None:
            # Color loss
            color_mask = color_fine_mask if color_fine_mask is not None else mask
            color_mask = color_mask[..., 0]
            # color_error = (color_fine[color_mask].fill_(-1.) - true_rgb[color_mask])
            if self.foundation_model == 'sd':
                assert not self.learn_bg_color
                object_mask = true_depth.view(-1) > 0
                true_rgb[~object_mask] = -1.
                # color_error = (color_fine[object_mask] - true_rgb[object_mask])
                color_error = (color_fine[color_mask] - true_rgb[color_mask])
            else:
                object_mask = None
                color_error = (color_fine[color_mask] - true_rgb[color_mask])
            if not extra_view and self.disable_in_color_loss:
                color_fine_loss = torch.tensor(0., device=color_error.device) # NOTE(lihe):debug
            else:
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error).to(color_error.device),
                                            reduction='mean')
            # save pred imgs
            if iter_step % vis_iter == 0:
                with torch.no_grad():
                    print(prefix + "======saving gt and perd imgs======" )
                    # save color masks
                    save_color_mask = color_mask.view(-1, 1, res, res).float()
                    save_image(save_color_mask, 'debug/color_masks.png' if not extra_view else 'debug/color_masks_extra.png', nrow=4 if not extra_view else self.extra_view_num)

                    gt_imgs = (true_rgb + 1) / 2. # (nv*h*w, 3)
                    gt_imgs = gt_imgs.view(-1, res, res, c).permute(0, 3, 1, 2)
                    gt_imgs_mask = gt_imgs * save_color_mask
                    pred_imgs = (color_fine + 1) / 2.
                    pred_imgs = pred_imgs.view(-1, res, res, c).permute(0, 3, 1, 2)
                    pred_imgs_mask = pred_imgs * save_color_mask
                    if not extra_view:
                        noisy_input = noisy_input[bs_id]
                        noisy_input = (noisy_input + 1) / 2.
                        noisy_imgs = noisy_input.view(*pred_imgs.shape)
                        save_imgs = torch.cat([gt_imgs, noisy_imgs, pred_imgs, gt_imgs_mask, pred_imgs_mask], dim=0)
                    else:
                        save_imgs = torch.cat([gt_imgs, pred_imgs, gt_imgs_mask, pred_imgs_mask], dim=0)
                    # cat noisy input
                    save_image(save_imgs[:, :3], 'debug/train_imgs.png' if not extra_view else 'debug/train_imgs_extra.png', nrow=4 if not extra_view else self.extra_view_num)
        else:
            color_fine_loss = torch.tensor(0., device=rays_o.device)

        if color_mlp is not None:
            # Color loss
            color_mlp_mask = color_mlp_mask[..., 0]
            color_error_mlp = (color_mlp[color_mlp_mask] - true_rgb[color_mlp_mask])
            color_mlp_loss = F.l1_loss(color_error_mlp,
                                       torch.zeros_like(color_error_mlp).to(color_error_mlp.device),
                                       reduction='mean')
        else:
            color_mlp_loss = torch.tensor(0., device=rays_o.device)

        if depth_pred is not None:
            depth_error = depth_pred[color_mask] - true_depth[color_mask]
            # depth_error = depth_pred[color_mask].fill_(0.) - true_depth[color_mask] # NOTE(lihe): debug!!!
            depth_loss = F.l1_loss(depth_error,
                                       torch.zeros_like(depth_error).to(depth_error.device),
                                       reduction='mean')
            # save pred imgs
            if iter_step % vis_iter == 0:
                with torch.no_grad():
                    print(prefix + "======saving gt and perd deps======")
                    save_color_mask = color_mask.view(-1, 1, res, res).float()
                    gt_dep = true_depth * 0.5 # (nv*h*w, 1)
                    gt_dep = gt_dep.view(-1, 1, res, res)
                    gt_dep_mask = gt_dep * save_color_mask
                    pred_dep = depth_pred * 0.5
                    pred_dep = pred_dep.view(-1, 1, res, res)
                    pred_dep_mask = pred_dep * save_color_mask
                    save_imgs = torch.cat([gt_dep, pred_dep, gt_dep_mask, pred_dep_mask], dim=0)
                    save_image(save_imgs, 'debug/train_deps.png' if not extra_view else 'debug/train_deps_extra.png', nrow=4 if not extra_view else self.extra_view_num)
        else:
            depth_loss = torch.tensor(0., device=rays_o.device)

        if not self.pred_density:
            if not extra_view:
                sparse_loss_1 = torch.exp(
                    -1 * torch.abs(render_out['sdf_random']) * self.sdf_decay_param).mean()  # - should equal
                sdf = render_out.get('sdf', None)
                if sdf is not None:
                    sparse_loss_2 = torch.exp(-1 * torch.abs(sdf) * self.sdf_decay_param).mean()
                else:
                    sparse_loss_2 = 0.
                sparse_loss = (sparse_loss_1 + sparse_loss_2) / 2
            else:
                sparse_loss = torch.tensor(0., device=rays_o.device)

            # Eikonal loss
            gradient_error_loss = gradient_error_fine
        else:
            sparse_loss = torch.tensor(0., device=rays_o.device)
            gradient_error_loss = torch.tensor(0., device=rays_o.device)

        # Mask loss, optional
        background_loss = 0.0
        fg_bg_loss = 0.0
        if self.fg_bg_weight > 0 and torch.mean((mask < 0.5).to(torch.float32)) > 0.02:
            weights_sum_fg = render_out['weights_sum_fg']
            fg_bg_error = (weights_sum_fg - mask)[mask < 0.5]
            fg_bg_loss = F.l1_loss(fg_bg_error,
                                   torch.zeros_like(fg_bg_error).to(fg_bg_error.device),
                                   reduction='mean')

        fg_bg_weight = 0.0 if iter_step < 50000 else get_weight(iter_step, self.fg_bg_weight)

        loss = color_fine_loss + color_mlp_loss + depth_loss +\
               sparse_loss * get_weight(iter_step, self.sdf_sparse_weight) + \
               fg_bg_loss * fg_bg_weight + \
               gradient_error_loss * self.sdf_igr_weight  # ! gradient_error_loss need a mask
        
        losses = {
            prefix + "mv_loss": loss.detach().item(),
            prefix + "color_loss": color_fine_loss.detach().item(),
            prefix + "dep_loss": depth_loss.detach().item(),
            prefix + "sp_loss": (sparse_loss * get_weight(iter_step, self.sdf_sparse_weight)).detach().item() if not self.pred_density and not extra_view else 0.,
            prefix + "grad_loss": (gradient_error_loss * self.sdf_igr_weight).detach().item() if not self.pred_density else 0.
        }
        return loss, losses

    
    def forward(self, feats, t, query_ids, sample_rays, depth=None, return_sigma=False, debug=False, voxels=None, vis_iter=200,
                unet=None, encoder_hidden_states=None, model_3d=None, dpm_solver_scheduler=None,
                noisy_sdf=None, mesh_save_path=None, only_train=None, background_rgb=-1,
                pyramid_feats=None, conditional_features_lod0_cache=[], new_batch=False, feats_64=None,
                pred_clean_sdf=None):
        """
        Args:
            feats: [B, 8, C, res, res]
            t: [B] timesteps
            query_ids: [B] (0-7) indicates which view is the query view
            sample_rays: dict('rays_o': [B, 8*num_rays, 3], 'rays_d': [B, 8*num_rays, 3], 'near': [], 'far': [])
            depth: [B, 8, res, res] if use GT depth correspondence
        """
        feats_high = sample_rays.get('mv_images_high', None) # b, 8, h, w, 3
        if feats_high is not None:
            b, num_views, c, res_org, _ = feats.shape
            feats = feats_high
            print(f"==== Replace feats with {res_org} res to feats_high with {feats.shape[3]} res ====")
        b, num_views, c, res, _ = feats.shape
        if feats_high is None:
            res_org = self.render_res

        # embed timesteps
        if self.add_temb:
            t_ = t.view([b, num_views])[:, 0] # [b,]
            t_emb = self.time_proj(t_)
            t_emb = t_emb.to(dtype=feats.dtype)
            emb = self.time_embedding(t_emb) # [B, 320]
            assert emb.shape[0] == b
            encoder_hidden_states_ = encoder_hidden_states.view([b, num_views, *encoder_hidden_states.shape[1:]])[:, 0]
            aug_emb = self.add_embedding(encoder_hidden_states_)
            emb = emb + aug_emb # TODO(lihe): ablate text embedding
        else:
            t_ = None
            emb = None

        rays_o = sample_rays['rays_o'] # (b, nv*w*h, 3)
        rays_d = sample_rays['rays_d'] # (b, nv*w*h, 3)
        near, far = sample_rays['near_fars'][..., :1], sample_rays['near_fars'][..., 1:] # (b, 2)
        w2cs = sample_rays['w2cs'] # (b, nv, 4, 4)
        c2ws = sample_rays['c2ws'] # (b, nv, 4, 4)
        intrinsics = sample_rays['K'] # (b, 3, 3)
        proj_mats = sample_rays['affine_mat'] # (b, nv, 4, 4)
        alpha_inter_ratio_lod0 = sample_rays['alpha_inter_ratio_lod0'] # float
        iter_step = sample_rays['step'] # int

        # decode 3d prior
        if self.use_3d_prior:
            latents_3d = sample_rays['latents_3d']
            if self.lazy_3d:
                t_3d = torch.where(t_ > self.lazy_t, torch.zeros_like(t_) + self.lazy_t, t_)
            else:
                t_3d = t_

            if self.training:
                noisy_latents_3d = self.ddpm_3d.q_sample(latents_3d, t_3d)
            else:
                noisy_latents_3d = latents_3d # latent3d is noisy latents during inference
            # self.options_3d.nerf_level = "coarse"
            self.options_3d.nerf_level = "fine"
        else:
            noisy_latents_3d = None
        
        # also generate sdf
        if self.sdf_gen:
            noise_scheduler = sample_rays['noise_scheduler']
            # NOTE(lihe): the sdf data is not [-1, 1], figure out if this is a bug
            gt_sdf = sample_rays['gt_sdf']
            # assert gt_sdf.shape[0] == 1, 'gt sdf bs is not 1'
            TSDF_VALUE = 0.001
            occupancy_high = torch.where(torch.abs(gt_sdf) < TSDF_VALUE, torch.ones_like(gt_sdf), torch.zeros_like(gt_sdf))
            occupancy = 2 * occupancy_high - 1 # [0,1] -> [-1, 1]
            occupancy = occupancy.view(b, -1, 1)
            raw_t = t.view([b, num_views])[:, 0] # [b,]
            noise_level = self.log_snr(raw_t / 1000)
            noise_level = noise_level.view(b, 1, 1)
            alpha, sigma = self.log_snr_to_alpha_sigma(noise_level)
            noise = torch.randn_like(occupancy)
            if self.training:
                noisy_sdf = alpha * occupancy + sigma * noise # [b, N, 1]
            else:
                assert noisy_sdf is not None
        else:
            noisy_sdf = None
        
        if self.voxel_cond:
            voxels = sample_rays['voxels'] # [b, n, 3]
        else:
            voxels = None

        # extra view
        if self.extra_view_num > 0:
            extra_rays_o = sample_rays['extra_rays_o'] # (b, nv*w*h, 3)
            extra_rays_d = sample_rays['extra_rays_d'] # (b, nv*w*h, 3)
            extra_near, extra_far = sample_rays['extra_near_fars'][..., :1], sample_rays['extra_near_fars'][..., 1:] # (b, 2)
            extra_w2cs = sample_rays['extra_w2cs'] # (b, nv, 4, 4)
            extra_c2ws = sample_rays['extra_c2ws'] # (b, nv, 4, 4)
            extra_intrinsics = sample_rays['extra_K'] # (b, 3, 3)
            extra_proj_mats = sample_rays['extra_affine_mat'] # (b, nv, 4, 4)
        
        # 1. get volume feats
        # NOTE(optional): use pre-trained SD to generate multi-view features
        with torch.no_grad():
            assert res == self.input_res, '{} vs {}'.format(res, self.input_res)
            if feats_64 is None:
                assert res == 64
                feats_64 = feats
            model_pred = unet(feats_64.view(b*num_views, -1, 64, 64), t,
                            encoder_hidden_states=encoder_hidden_states,
                            return_feats=False).sample
        # NOTE(optional): use the predicted clean x0 as additional feature
        if self.debug_sd_feat:
            with torch.no_grad():
                ## NOTE: we rewrite the step func
                noise_scheduler = sample_rays['noise_scheduler'] # NOTE: use noise scheduler to transfer noise to clean x0
                pred_x0 = noise_scheduler.step_w_device(
                        model_pred, t, feats_64.view(b*num_views, -1, 64, 64), device=feats_64.device
                    )
                pred_x0_64 = pred_x0
                if res > 64:
                    pred_x0 = F.interpolate(pred_x0, size=(res, res), mode='bilinear', align_corners=False, antialias=True)

                if not self.abandon_sdf_x0:
                    pyramid_feats = pred_x0
                else:
                    pyramid_feats = None
                if (self.training and iter_step % vis_iter == 0):
                    save_x0 = torch.cat([feats.view(b*num_views, -1, res, res), pred_x0], dim=0)
                    save_x0 = (save_x0 + 1) /2 
                    save_image(save_x0[:, :3], 'debug/train_x0.png', nrow=4)
                    save_x0_64 = torch.cat([feats_64.view(b*num_views, -1, 64, 64), pred_x0_64], dim=0)
                    save_x0_64 = (save_x0_64 + 1) /2
                    save_image(save_x0_64[:, :3], 'debug/train_x0_64.png', nrow=4)

        pyramid_feats = torch.cat([feats.view(b*num_views, c, res, res), pyramid_feats], dim=1) if not self.abandon_sdf_x0 else feats.view(b*num_views, c, res, res) # we do not feed x0 into the featurenet
        pyramid_feats = self.featurenet.obtain_pyramid_feature_maps(pyramid_feats)
        pyramid_feats = pyramid_feats.view(b, num_views, *pyramid_feats.shape[1:]) # (B, 8, 56, H, W)
                
        partial_vol_origin = torch.tensor(self.partial_vol_origin).unsqueeze(0).to(feats.device) # [-1., -1., -1] (1, 3)

        # NOTE(lihe): use image300M or text300M
        noisy_latents_3d_prev = None
        if self.model_type == 'image300M' and model_3d is not None and self.use_3d_prior:
            # NOTE(lihe): implement lazy training && sampling strategy
            with torch.no_grad():
                # split input data
                guidance_scale = 3.0
                batch_one = 1 if self.training else 2
                assert t_3d.shape[0] == batch_one, "we only support bs=1(train)/2(test) using 3d prior."
                if t_3d[0] == self.lazy_t: # NOTE(lihe): debug
                    cond_image = sample_rays['mv_images'][:, 5] # b, 8, c, h, w
                else:
                    cond_image = pred_x0.view(b, num_views, *pred_x0.shape[1:])[:, 5] # now we directly use one specific view
                cond_image = (cond_image + 1.) / 2.
                embeddings = model_3d.wrapped.clip.model.get_image_grid_features(cond_image.detach())
                g_model = uncond_guide_model_x0(model_3d, guidance_scale)

                if guidance_scale > 1:
                    latents_input = torch.cat([noisy_latents_3d]*2)
                    t_batch = torch.cat([t_3d]*2)
                else:
                    latents_input = noisy_latents_3d
                    t_batch = t_3d
                model_kwargs_3d = dict(embeddings=embeddings)
                if guidance_scale != 1.0 and guidance_scale != 0.0:
                    for k, v in model_kwargs_3d.copy().items():
                        model_kwargs_3d[k] = torch.cat([v, torch.zeros_like(v)], dim=0)
                output = g_model(latents_input, t_batch, **model_kwargs_3d)
                output = output.clamp(-1, 1) # clean latents
                if not self.training:
                    if t_3d[0] < self.lazy_t:
                        noisy_latents_3d = dpm_solver_scheduler.scale_model_input(noisy_latents_3d, t_3d[0]) # NOTE(lihe): debug
                        noisy_latents_3d_prev = dpm_solver_scheduler.step(
                            output, t_3d[0], noisy_latents_3d, return_dict=False
                        )[0]
                    else:
                        noisy_latents_3d_prev = noisy_latents_3d # TODO(lihe): when t >= lazy_t, dont update noisy latent
                noisy_latents_3d = output
                if (self.training and iter_step % vis_iter == 0):
                    np.save('debug/img300M_latents.npy', noisy_latents_3d.detach().cpu().numpy())
                    save_image(cond_image, 'debug/train_latent_cond_image.png', nrow=4)
                    print(f"====== training, save img300M results at timestep t_ : {t_[0]}, t_3d: {t_3d[0]} =====")
                    render_latents(self.xm, noisy_latents_3d, device=noisy_latents_3d.device)
                # if not self.training:
                #     save_image(cond_image, 'debug/eval_latent_cond_image.png', nrow=4)
                #     print(f"====== testing, save img300M results at timestep t_ : {t_[0]}, t_3d: {t_3d[0]} =====")
                #     render_latents(self.xm, noisy_latents_3d, device=noisy_latents_3d.device, name='eval')

        loss_all = 0.
        render_color = []
        render_dep = []
        pred_clean_sdf = []
        all_attn_mask = []
        for bs_id in range(b):
            with torch.autocast('cuda', enabled=False):
                conditional_features_lod0 = self.sdf_def_network.get_conditional_volume(
                        feature_maps=pyramid_feats[bs_id:bs_id+1].float(), # (1, 8, 56, H, W)
                        partial_vol_origin=partial_vol_origin.float(), # (1, 3)
                        proj_mats=proj_mats[bs_id:bs_id+1].float(), # (1, 4, 4)
                        sizeH=res,
                        sizeW=res,
                        lod=0,
                        debug_images=sample_rays['mv_images'],
                        emb=emb[bs_id:bs_id+1].float() if emb is not None else None,
                        noisy_latents_3d=noisy_latents_3d[bs_id:bs_id+1].float() if noisy_latents_3d is not None else None,
                        xm=self.xm,
                        options_3d=self.options_3d,
                        noisy_sdf=noisy_sdf[bs_id] if self.sdf_gen else None,
                        voxels=voxels[bs_id] if voxels is not None else None
                    )
            for k, v in conditional_features_lod0.items():
                conditional_features_lod0[k] = v.to(pyramid_feats.dtype)
            
            con_volume = conditional_features_lod0['dense_volume'] # (1, C, 96, 96, 96)
            con_valid_mask_volume = conditional_features_lod0['valid_mask_volume'] # (1, 1, 96, 96, 96)
            coords = conditional_features_lod0['coords']  # [1,3,wX,wY,wZ]

            mesh = None
            render_out = {}
            # NOTE(lihe): input view rendering with rasterization
            cameras = flex_render.get_random_camera_batch(self.extra_view_num, iter_res=[self.img_resolution, self.img_resolution], device=con_volume.device)
            render_out = self.dmtet_renderer.generate(volume=con_volume, cam_mv=c2ws[bs_id], # [8, 4, 4], query camera
                                                    conditional_valid_mask_volume=con_valid_mask_volume,
                                                    feature_maps=pyramid_feats[bs_id],
                                                    img_wh=[res, res],
                                                    color_maps=feats[bs_id] if not self.blend_x0 else pred_x0.view(*feats.shape)[bs_id], # NOTE(lihe): blend x0
                                                    w2cs=w2cs[bs_id],
                                                    intrinsics=intrinsics[bs_id],
                                                    vol_dims=torch.tensor(self.vol_dims).to(feats.device),
                                                    partial_vol_origin=partial_vol_origin[0], # [3]
                                                    vol_size=self.voxel_size,
                                                    emb=emb[bs_id:bs_id+1] if emb is not None else None,
                                                    t=t_[bs_id:bs_id+1],
                                                    mesh=None, # NOTE: feed previous constructed mesh
                                                    kal_cameras=cameras,
                                                    res=self.render_res,
                                                    )
            # mesh = render_out['mesh']
            # NOTE: compute sdf loss
            mesh_rast = render_out['flex_mesh']
            if self.sdf_gen:
                query_pts = self.query_pts.to(con_volume.device)
                pred_sdf = flex_render.compute_sdf(query_pts, mesh_rast.vertices, mesh_rast.faces)
                if self.training:
                    pred_sdf_masked = pred_sdf.view(-1)
                    pred_sdf_masked = pred_sdf_masked[occupancy_high.view(-1) > 0]
                    gt_sdf_masked = gt_sdf.view(-1)
                    gt_sdf_masked = gt_sdf_masked[occupancy_high.view(-1) > 0]
                    sdf_denoise_loss = torch.nn.functional.mse_loss(pred_sdf_masked, gt_sdf_masked) * 2e3
                else:
                    pred_sdf = (pred_sdf.abs() < 0.001).float()
                    pred_sdf = pred_sdf.view(-1, 1)
                pred_clean_sdf.append(pred_sdf)

            if self.training and iter_step % 78 == 0:
                mesh_debug_save_path = os.path.join('debug', 'mesh_rast_train.ply')
                mesh_v, mesh_f = mesh_rast.vertices.detach().cpu().numpy(), mesh_rast.faces.detach().cpu().numpy()
                mesh_save = trimesh.Trimesh(vertices=mesh_v, faces=mesh_f)
                mesh_save.export(mesh_debug_save_path)
                print(f"Save mesh to {mesh_debug_save_path}")
            elif not self.training and t_[0] < 50:
                if mesh_save_path is None:
                    mesh_save_path = os.path.join('debug', 'mesh_rast_sample.ply')
                mesh_v, mesh_f = mesh_rast.vertices.detach().cpu().numpy(), mesh_rast.faces.detach().cpu().numpy()
                mesh_save = trimesh.Trimesh(vertices=mesh_v, faces=mesh_f)
                mesh_save.export(mesh_save_path)
                
            # extra view rendering
            if self.extra_view_num > 0 and self.training:
                # random choose x views
                extra_view_num = self.extra_view_num
                gt_mesh = sample_rays['gt_mesh'][bs_id]
                cameras = flex_render.get_random_camera_batch(self.extra_view_num, iter_res=[self.img_resolution, self.img_resolution], device=con_volume.device)
                target = flex_render.render_mesh(gt_mesh, cameras, [self.img_resolution, self.img_resolution])
                
                extra_render_out = self.dmtet_renderer.generate(volume=con_volume, cam_mv=extra_c2ws[bs_id][:extra_view_num], # [8, 4, 4], query camera
                                                    conditional_valid_mask_volume=con_valid_mask_volume,
                                                    feature_maps=pyramid_feats[bs_id],
                                                    img_wh=[res, res],
                                                    color_maps=feats[bs_id] if not self.blend_x0 else pred_x0.view(*feats.shape)[bs_id], # NOTE(lihe): blend x0
                                                    w2cs=w2cs[bs_id],
                                                    intrinsics=intrinsics[bs_id],
                                                    vol_dims=torch.tensor(self.vol_dims).to(feats.device),
                                                    partial_vol_origin=partial_vol_origin[0], # [3]
                                                    vol_size=self.voxel_size,
                                                    emb=emb[bs_id:bs_id+1] if emb is not None else None,
                                                    t=t_[bs_id:bs_id+1] if t_ is not None else None,
                                                    mesh=mesh, # NOTE: feed previous constructed mesh
                                                    kal_cameras=cameras,
                                                    res=self.render_res,
                                                    )
                flex_mesh = extra_render_out['flex_mesh']
                gt = flex_render.render_mesh(gt_mesh, cameras, [self.img_resolution, self.img_resolution], return_types=["normals"])
                pred = flex_render.render_mesh(flex_mesh, cameras, [self.img_resolution, self.img_resolution], return_types=["normals"])
                extra_render_out['pred_normal'] = pred['normals']
                extra_render_out['gt_normal'] = gt['normals']
                if iter_step % vis_iter == 0:
                    render_normal_gt = (gt['normals'] + 1) / 2.
                    render_normal_gt = render_normal_gt.permute(0, 3, 1, 2)
                    render_normal_pred = (pred['normals'] + 1) / 2.
                    render_normal_pred = render_normal_pred.permute(0, 3, 1, 2)
                    save_image(torch.cat([render_normal_gt, render_normal_pred], dim=0), f'debug/rast_normals{DEBUG_ID}.png', nrow=4)
                        
            # compute loss
            if self.training:
                loss, losses = self.cal_losses_rgb(render_out, iter_step, sample_rays)
                if self.extra_view_num > 0:
                    extra_loss, extra_losses = self.cal_losses_rast_flexi(extra_render_out, target, iter_step, sample_rays)
                    loss = loss + extra_loss
                    losses.update(extra_losses)
                if self.sdf_gen:
                    loss = loss + sdf_denoise_loss
                    losses.update({'sdf_loss':sdf_denoise_loss.item()})
            else:
                loss = 0.
                losses = {}
                
            loss_all = loss_all + loss

            # get rendered imgs
            color_fine = render_out['color_fine'] # (nv*w*h, 3)
            color_fine = color_fine.view(num_views, res_org, res_org, self.img_ch)
            color_fine = color_fine.permute(0, 3, 1, 2) # (nv, 3, h, w)
            render_color.append(color_fine)
        
        # average 
        loss_all = loss_all / b
        out_color = torch.stack(render_color, dim=0) # (b, nv, 3, h, w)
        
        if self.sdf_gen and not self.training:
            pred_clean_sdf = torch.stack(pred_clean_sdf) # [2, N, 1]
        else:
            pred_clean_sdf = None
            
        return out_color, loss_all, losses, pred_x0, noisy_latents_3d, noisy_latents_3d_prev, pred_clean_sdf
        
        

    
