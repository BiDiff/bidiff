import os
import trimesh

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from einops import rearrange

from transformers import ViTImageProcessor, ViTModel
from transformers import AutoModel
from diffusers.models.resnetfc import ResnetFC
from diffusers.models.embeddings import (
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)

from diffusers.models.sparse_neus.sparse_sdf_network import SparseSdfNetwork
from diffusers.models.sparse_neus.fields import SingleVarianceNetwork
from diffusers.models.sparse_neus.rendering_network import GeneralRenderingNetwork
from diffusers.models.sparse_neus.sparse_neus_renderer import SparseNeuSRenderer
from diffusers.models.sparse_neus.pos_encoder import PositionalEncoding
from diffusers.models.openaimodel.openaimodel import build_nerf_encoder, timestep_embedding
from diffusers.models.openaimodel.code_test import DEBUG_SAVE_ID


class FusionNetwork(nn.Module):
    def __init__(self, in_channels=768, post_process_channels=[96, 192, 384, 768], hidden_channels=96, out_channels=56):
        super().__init__()
        self.in_channels = in_channels
        self.post_process_channels = post_process_channels
        self.hidden_channels = hidden_channels
        self.projection_layers = nn.ModuleList()
        for post_process_channel in post_process_channels:
            self.projection_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, post_process_channel, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                )
            )
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                    in_channels=post_process_channels[0], out_channels=post_process_channels[0], kernel_size=8, stride=8, padding=0
                ),
            nn.ConvTranspose2d(
                in_channels=post_process_channels[1], out_channels=post_process_channels[1], kernel_size=4, stride=4, padding=0
            ),
            nn.ConvTranspose2d(
                in_channels=post_process_channels[2], out_channels=post_process_channels[2], kernel_size=2, stride=2, padding=0
            ),
            nn.Identity(),
        ])
        self.convs = nn.ModuleList()
        for post_process_channel in post_process_channels:
            self.convs.append(nn.Conv2d(post_process_channel, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.fusion_layers1 = nn.ModuleList()
        for _ in range(len(self.convs)):
            self.fusion_layers1.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                )
            )
        self.fusion_layers2 = nn.ModuleList()
        for _ in range(len(self.convs)):
            self.fusion_layers2.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                )
            )
        self.out_layer = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        assert len(x) == len(self.fusion_layers1)
        x = [projection_layer(x[i]) for i, projection_layer in enumerate(self.projection_layers)]
        x = [resize_layer(x[i]) for i, resize_layer in enumerate(self.resize_layers)]
        x = [conv(x[i]) for i, conv in enumerate(self.convs)]
        out = x[-1]
        for i in range(len(self.fusion_layers1)):
            out = out + self.fusion_layers1[i](x[-1-i]) # ensure relu(replace=False)
            out = out + self.fusion_layers2[i](out)
            out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.out_layer(out)
        return out
        

class MultiviewEncoder(nn.Module):
    """
    Multiview to volume encoder.
    """
    def __init__(
            self,  
            out_channels, 
            pretrained_image_encoder='facebook/dino-vitb16',
            freeze_pretrained_image_encoder=True,
            pretrained_encoder_out_channels=768,
            pretrained_encoder_down_scale=16,
            use_fusion_layers=False,
            ):
        super().__init__()
        self.out_channels = out_channels
        self.pretrained_encoder_out_channels = pretrained_encoder_out_channels
        self.pretrained_encoder_down_scale = pretrained_encoder_down_scale

        if isinstance(pretrained_image_encoder, str):
            self.image_encoder = self._build_pretrained_model(pretrained_image_encoder)
        elif isinstance(pretrained_image_encoder, nn.Module):
            self.image_encoder = pretrained_image_encoder
        else:
            raise NotImplementedError

        self.freeze_pretrained_image_encoder = freeze_pretrained_image_encoder
        if freeze_pretrained_image_encoder:
            self._freezed_image_encoder()
        self.use_fusion_layers = use_fusion_layers
        if not use_fusion_layers:
            self.out_layers = nn.Linear(pretrained_encoder_out_channels, out_channels)
            self.out_size = 32
        else:
            self.out_layers = FusionNetwork(out_channels=out_channels)
            self.out_size = 512
    
    def _build_pretrained_model(self, pretrained_model_name, proxy_error_retries=3, proxy_error_cooldown=5):
        import requests
        try:
            model = AutoModel.from_pretrained(pretrained_model_name, add_pooling_layer=False, output_hidden_states=True)
            return model
        except requests.exceptions.ProxyError as err:
            if proxy_error_retries > 0:
                print(f"Huggingface ProxyError: Retrying in {proxy_error_cooldown} seconds...")
                import time
                time.sleep(proxy_error_cooldown)
                return MultiviewEncoder._build_pretrained_model(pretrained_model_name, proxy_error_retries - 1, proxy_error_cooldown)
            else:
                raise err
    
    def _freezed_image_encoder(self):
        print(f"======== Freezing ImageEncoder ========")
        self.image_encoder.eval()
        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = False

    def forward(self, image):
        """
        Args:
            image: (b, v, c_in, h_in, w_in) ~ [-1, 1]
        Returns:
            image_feats: (b, v, c_out, h, w)
        """
        b, nv, c, h, w = image.shape
        if self.freeze_pretrained_image_encoder:
            with torch.no_grad():
                image_feats = self.image_encoder(
                    pixel_values=image.view(b * nv, c, h, w), 
                    interpolate_pos_encoding=True
                )
        else:
            image_feats = self.image_encoder(
                pixel_values=image.view(b * nv, c, h, w), 
                interpolate_pos_encoding=True
            )
        if self.use_fusion_layers:
            image_feats = image_feats.hidden_states
            image_feats = [image_feats[i][:, 1:, :] for i in [2, 5, 8, -1]]
            bv, hw, c_hidden = image_feats[0].shape
            h_out, w_out = h // self.pretrained_encoder_down_scale, w // self.pretrained_encoder_down_scale
            assert hw == h_out * w_out
            image_feats = [rearrange(o, 'b (h w) c -> b c h w', h=h_out, w=w_out) for o in image_feats]
            image_feats = rearrange(self.out_layers(image_feats), '(b v) c h w -> b v c h w', b=b, v=nv)
        else:
            image_feats = image_feats.last_hidden_state[:, 1:, :] # (b * nv, h*w=h_in/16**2, c) dinovitb16
            image_feats = rearrange(self.out_layers(image_feats), '(b v) (h w) c -> b v c h w', b=b, v=nv, h=32, w=32)
        return image_feats


class MMVAE_MultiviewVolumeNeus(nn.Module):
    """
    Multimodal VAE.
    """
    def __init__(
            self, 
            num_views, 
            image_channels,
            image_feat_channels,
            pretrained_image_encoder='facebook/dino-vitb16',
            freeze_pretrained_image_encoder=True,
            pretrained_encoder_out_channels=768,
            voxel_size=0.7/95., vol_dims=[96, 96, 96], hidden_dim=128, cost_type='variance_mean', # sparsesdfnet
            d_pyramid_feature_compress=16, regnet_d_out=16, num_sdf_layers=4, multires=6, # 
            init_val=0.2, # variance
            in_geometry_feat_ch=16, in_rendering_feat_ch=56, anti_alias_pooling=True, # rendering network
            n_samples=48, n_importance=32, n_outside=0, perturb=1.0, alpha_type='div',
            partial_vol_origin=[-0.35, -0.35, -0.35], scale=0.35, pred_density=False,
            use_sd_feat=True, add_temb=True, temb_channels=320, encoder_hid_dim=4096,
            regress_rgb=False, foundation_model='if', learn_bg_color=False, use_all_feat=False,
            pos_enc=False, debug_sd_feat=False, blend_x0=False,
            extra_view_num=0, disable_in_color_loss=False,
            abandon_sdf_x0=False,
            debug_regress=False,
            use_resnetfc=False,
            use_3d_prior=False, 
            device=None,
            use_diff_featuernet=False,
            model_type='text300M',
            direct_use_3d=False,
            lazy_3d=False,
            lazy_t=None,
            new_sdf_arc=False,
            sdf_gen=False,
            voxel_cond=False,
            use_featurenet_view_embed=False,
            geo_attn_mask=False,
            use_dino_feat=False,
    ):
        super().__init__()
        self.image_encoder = MultiviewEncoder(
            out_channels=image_feat_channels,
            pretrained_image_encoder=pretrained_image_encoder,
            freeze_pretrained_image_encoder=freeze_pretrained_image_encoder,
            pretrained_encoder_out_channels=pretrained_encoder_out_channels,
        )

        self.code = PositionalEncoding(num_freqs=6, d_in=3, freq_factor=1.5, include_input=True)
        self.sdf_network = SparseSdfNetwork(
            lod=0, ch_in=image_feat_channels, voxel_size = voxel_size, vol_dims = vol_dims, 
            hidden_dim = hidden_dim, cost_type = cost_type, d_pyramid_feature_compress = d_pyramid_feature_compress, 
            regnet_d_out = regnet_d_out, num_sdf_layers = num_sdf_layers, multires = multires,
            add_temb=add_temb, 
            temb_channels=temb_channels,
            use_3d_prior=use_3d_prior,
            new_sdf_arc=new_sdf_arc,
            sdf_gen=sdf_gen,
            voxel_cond=voxel_cond,
        )
        self.variance_network = SingleVarianceNetwork(init_val=init_val)
        self.rendering_network = ResnetFC(
            d_out=image_channels, d_in=self.code.d_out + 3,
            n_blocks=3, 
            d_latent=image_channels * 8 + in_rendering_feat_ch + in_geometry_feat_ch,
            d_hidden=64, time_embed_ch=64)
        self.sdf_renderer = SparseNeuSRenderer(
            None, # render_network_outside = None
            self.sdf_network,
            self.variance_network,
            self.rendering_network,
            n_samples,
            n_importance,
            n_outside,
            perturb,
            alpha_type=alpha_type,
            conf=None,
            learn_bg_color=learn_bg_color,
            rgb_ch=image_channels,
            pos_code=self.code,
            use_resnetfc=True,
            direct_use_3d=direct_use_3d,
            use_3d_prior=use_3d_prior,
            geo_attn_mask=False)
        
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
        
        self.use_3d_prior = use_3d_prior
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
        
        self.num_views = num_views
        self.img_ch = image_channels
        self.partial_vol_origin = partial_vol_origin
        self.voxel_size = voxel_size
        self.vol_dims = vol_dims
        self.scale = scale
        self.fg_bg_weight = 0.
        self.anneal_start = 0
        self.anneal_end = 25000
        self.sdf_sparse_weight = 0.02
        self.sdf_igr_weight = 0.1
        self.sdf_decay_param = 100
        self.pred_density = pred_density
        self.use_sd_feat = use_sd_feat
        self.regress_rgb = regress_rgb
        self.foundation_model = foundation_model
        self.learn_bg_color = learn_bg_color
        self.pos_enc = pos_enc
        self.use_all_feat = use_all_feat
        self.debug_sd_feat = debug_sd_feat
        if self.learn_bg_color:
            assert self.foundation_model == 'sd', 'only latent diffusion need to regress color'
        self.blend_x0 = blend_x0
        self.extra_view_num = extra_view_num
        self.disable_in_color_loss = disable_in_color_loss
        self.use_resnetfc = use_resnetfc
        self.model_type = model_type
        self.lazy_3d = lazy_3d
        # self.lazy_t = 600 # 700
        self.lazy_t = lazy_t
        if self.lazy_3d:
            assert lazy_t is not None
        self.sdf_gen = sdf_gen
        self.voxel_cond = voxel_cond
        self.geo_attn_mask = geo_attn_mask
    
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
            save_path = os.path.join('debug', 'meshes_' + mode, 'mesh{}.ply'.format(DEBUG_SAVE_ID))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # mesh.export(os.path.join('debug', 'meshes_' + mode,
        #                          'mesh_{:0>8d}_{}_lod{:0>1d}.ply'.format(iter_step, meta, lod)))
        mesh.export(save_path)
    
    def cal_losses_sdf(self, render_out, sample_rays, iter_step=-1, lod=0, bs_id=0, vis_iter=78, noisy_input=None, extra_view=False):

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
        # true_rgb = sample_rays['rays_color'][0]
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

        # TODO(lihe): add rays mask in dataset
        if 'rays_mask' in sample_rays:
            mask = sample_rays['rays_mask'][bs_id]
        else:
            mask = torch.ones(rays_o.shape[0]).to(rays_o.device)
        # mask = sample_rays['rays_mask'][0]

        color_fine = render_out.get('color_fine', None)
        color_fine_mask = render_out.get('color_fine_mask', None)
        depth_pred = render_out.get('depth', None)

        # NOTE(lihe): not use now
        # variance = render_out['variance']
        # cdf_fine = render_out['cdf_fine']
        # weight_sum = render_out['weights_sum']

        gradient_error_fine = render_out.get('gradient_error_fine', torch.tensor(0., device=rays_o.device))

        # * color generated by mlp
        color_mlp = render_out.get('color_mlp', None)
        color_mlp_mask = render_out.get('color_mlp_mask', None)

        if color_fine is not None:
            # Color loss
            color_mask = color_fine_mask if color_fine_mask is not None else mask
            color_mask = color_mask[..., 0]
            # print("===color mask===", color_mask.shape, color_mask.dtype)
            # print("====color fine===", color_fine.shape)
            # print("===debug color fine[mask]==", color_fine[color_mask].shape, color_fine[color_mask])
            # debug_color = color_fine.new_zeros(*color_fine.shape)
            # debug_dep = true_depth.new_zeros(*true_depth.shape)
            # debug_gt = true_rgb.new_zeros(*color_fine.shape)
            # debug_color[color_mask] = color_fine[color_mask]
            # debug_gt[color_mask] = true_rgb[color_mask]
            # debug_dep[color_mask] = true_depth[color_mask]

            # debug_color = debug_color.view(-1, res, res, c).permute(0, 3, 1, 2)
            # debug_gt = debug_gt.view(-1, res, res, c).permute(0, 3, 1, 2)
            # debug_color = (debug_color + 1.) / 2.
            # debug_gt = (debug_gt + 1.) / 2.
            # debug_imgs = torch.cat([debug_gt, debug_color], dim=0)
            # debug_dep = debug_dep.view(-1, 1, res, res) * 0.5
            # save_image(debug_imgs[:, :3], 'debug/train_debug_imgs.png' if not extra_view else 'debug/train_debug_imgs_extra.png', nrow=4 if not extra_view else self.extra_view_num)
            # save_image(debug_dep, 'debug/train_debug_dep.png' if not extra_view else 'debug/train_debug_dep_extra.png', nrow=4 if not extra_view else self.extra_view_num)
            ###
            
            # color_error = (color_fine[color_mask].fill_(-1.) - true_rgb[color_mask]) #NOTE(lihe): debug !!!
            if self.foundation_model == 'sd':
                assert not self.learn_bg_color
                object_mask = true_depth.view(-1) > 0
                true_rgb[~object_mask] = -1.
                # color_error = (color_fine[object_mask] - true_rgb[object_mask])
                color_error = (color_fine[color_mask] - true_rgb[color_mask])
            else:
                object_mask = None
                color_error = (color_fine[color_mask] - true_rgb[color_mask])
            # color_error = (color_fine - true_rgb)
            if not extra_view and self.disable_in_color_loss:
                color_fine_loss = torch.tensor(0., device=color_error.device) # NOTE(lihe):debug
            else:
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error).to(color_error.device),
                                            reduction='mean')
                # color_fine_loss = torch.tensor(0., device=color_error.device) # NOTE(lihe):debug
                # print("====debug color loss====", color_fine_loss)
                # exit()
            
            
            # NOTE(lihe): debug, remember to comment the following lines
            # color_fine_loss = color_fine_loss * 0. # NOTE(lihe):debug
            # save pred imgs
            if iter_step % vis_iter == 0:
                with torch.no_grad():
                    print(prefix + "======saving gt and perd imgs======" )
                    # save color masks
                    # save_color_mask = color_mask.view(-1, 1, res, res).float() if object_mask is None else object_mask.view(-1, 1, res, res).float()
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
            
                
            # psnr = 20.0 * torch.log10(
            #     1.0 / (((color_fine[color_mask] - true_rgb[color_mask]) ** 2).mean() / (3.0)).sqrt())
        else:
            color_fine_loss = torch.tensor(0., device=rays_o.device)
            psnr = torch.tensor(0., device=rays_o.device)

        if color_mlp is not None:
            # Color loss
            color_mlp_mask = color_mlp_mask[..., 0]
            color_error_mlp = (color_mlp[color_mlp_mask] - true_rgb[color_mlp_mask])
            color_mlp_loss = F.l1_loss(color_error_mlp,
                                       torch.zeros_like(color_error_mlp).to(color_error_mlp.device),
                                       reduction='mean')

            psnr_mlp = 20.0 * torch.log10(
                1.0 / (((color_mlp[color_mlp_mask] - true_rgb[color_mlp_mask]) ** 2).mean() / (3.0)).sqrt())
        else:
            color_mlp_loss = torch.tensor(0., device=rays_o.device)
            psnr_mlp = torch.tensor(0., device=rays_o.device)

        # depth loss is only used for inference, not included in total loss
        if depth_pred is not None:
            # depth_loss = self.depth_criterion(depth_pred, true_depth, mask)
            # depth_error = depth_pred - true_depth # NOTE(lihe): debug
            depth_error = depth_pred[color_mask] - true_depth[color_mask]
            # depth_error = depth_pred[color_mask].fill_(0.) - true_depth[color_mask] # NOTE(lihe): debug!!!
            depth_loss = F.l1_loss(depth_error,
                                       torch.zeros_like(depth_error).to(depth_error.device),
                                       reduction='mean')
            # print(prefix + "===dep loss: ", depth_loss)
            # NOTE(lihe): debug
            # depth_loss = depth_loss * 0.
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
                    save_image(save_imgs, 'debug/train_deps{}.png'.format(DEBUG_SAVE_ID) if not extra_view else 'debug/train_deps_extra{}.png'.format(DEBUG_SAVE_ID), nrow=4 if not extra_view else self.extra_view_num)
            # depth evaluation
            # NOTE(lihe): we dont evaluate depth to save memory now
            # depth_statis = compute_depth_errors(depth_pred.detach().cpu().numpy(), true_depth.cpu().numpy(),
            #                                     mask.cpu().numpy() > 0)
            # depth_statis = numpy2tensor(depth_statis, device=rays_o.device)
            depth_statis = None
        else:
            depth_loss = torch.tensor(0., device=rays_o.device)
            depth_statis = None

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

            # sdf_mean = torch.abs(sdf).mean()
            # sparseness_1 = (torch.abs(sdf) < 0.01).to(torch.float32).mean()
            # sparseness_2 = (torch.abs(sdf) < 0.02).to(torch.float32).mean()

            # Eikonal loss
            gradient_error_loss = gradient_error_fine
        else:
            sparse_loss = torch.tensor(0., device=rays_o.device)
            gradient_error_loss = torch.tensor(0., device=rays_o.device)

        # Mask loss, optional
        # The images of DTU dataset contain large black regions (0 rgb values),
        # can use this data prior to make fg more clean
        background_loss = 0.0
        fg_bg_loss = 0.0
        if self.fg_bg_weight > 0 and torch.mean((mask < 0.5).to(torch.float32)) > 0.02:
            weights_sum_fg = render_out['weights_sum_fg']
            fg_bg_error = (weights_sum_fg - mask)[mask < 0.5]
            fg_bg_loss = F.l1_loss(fg_bg_error,
                                   torch.zeros_like(fg_bg_error).to(fg_bg_error.device),
                                   reduction='mean')

        # ! the first 50k, don't use bg constraint
        fg_bg_weight = 0.0 if iter_step < 50000 else get_weight(iter_step, self.fg_bg_weight)

        loss = color_fine_loss + color_mlp_loss + depth_loss +\
               sparse_loss * get_weight(iter_step, self.sdf_sparse_weight) + \
               fg_bg_loss * fg_bg_weight + \
               gradient_error_loss * self.sdf_igr_weight  # ! gradient_error_loss need a mask
        
        # losses = {}
        losses = {
            prefix + "mv_loss": loss.detach().item(),
            prefix + "color_loss": color_fine_loss.detach().item(),
            prefix + "dep_loss": depth_loss.detach().item(),
            prefix + "sp_loss": (sparse_loss * get_weight(iter_step, self.sdf_sparse_weight)).detach().item() if not self.pred_density and not extra_view else 0.,
            prefix + "grad_loss": (gradient_error_loss * self.sdf_igr_weight).detach().item() if not self.pred_density else 0.
        }

        # losses = {
        #     "loss": loss,
        #     "depth_loss": depth_loss,
        #     "color_fine_loss": color_fine_loss,
        #     "color_mlp_loss": color_mlp_loss,
        #     "gradient_error_loss": gradient_error_loss,
        #     "background_loss": background_loss,
        #     "sparse_loss": sparse_loss,
        #     "sparseness_1": sparseness_1,
        #     "sparseness_2": sparseness_2,
        #     "sdf_mean": sdf_mean,
        #     "psnr": psnr,
        #     "psnr_mlp": psnr_mlp,
        #     "weights_sum": render_out['weights_sum'],
        #     "weights_sum_fg": render_out['weights_sum_fg'],
        #     "alpha_sum": render_out['alpha_sum'],
        #     "variance": render_out['variance'],
        #     "sparse_weight": get_weight(iter_step, self.sdf_sparse_weight),
        #     "fg_bg_weight": fg_bg_weight,
        # }

        # losses = numpy2tensor(losses, device=rays_o.device)
        # return loss, losses, depth_statis
        return loss, losses
    
    def forward(self, feats, t, sample_rays, voxels=None,  unet=None, 
                encoder_hidden_states=None, model_3d=None, dpm_solver_scheduler=None,
                noisy_sdf=None, mesh_save_path=None, only_train=None, background_rgb=-1,
                pyramid_feats=None, conditional_features_lod0_cache=[]):
        """
        Args:
            feats: (b, v, c_in, h_in, w_in) ~ [-1, 1]
            t: (b) timesteps
            sample_rays: dict('rays_o': [B, 8*num_rays, 3], 'rays_d': [B, 8*num_rays, 3], 'near': [], 'far': [])
        """
        b, num_views, c, res, _ = feats.shape

        # embed timesteps
        if self.add_temb:
            t_ = t.view([b, num_views])[:, 0] # [b,]
            t_emb = self.time_proj(t_)
            t_emb = t_emb.to(dtype=feats.dtype)
            emb = self.time_embedding(t_emb) # [B, 320]
            assert emb.shape[0] == b
            encoder_hidden_states_ = encoder_hidden_states.view([b, num_views, *encoder_hidden_states.shape[1:]])[:, 0]
            aug_emb = self.add_embedding(encoder_hidden_states_)
            emb = emb + aug_emb # NOTE(lihe): remove text embeding to debug
        else:
            emb = None

        rays_o = sample_rays['rays_o'] # (b, nv*w*h, 3)
        rays_d = sample_rays['rays_d'] # (b, nv*w*h, 3)
        near, far = sample_rays['near_fars'][..., :1], sample_rays['near_fars'][..., 1:] # (b, 2)
        w2cs = sample_rays['w2cs'] # (b, nv, 4, 4)
        c2ws = sample_rays['c2ws'] # (b, nv, 4, 4)
        intrinsics = sample_rays['K'] # (b, 3, 3)
        proj_mats = sample_rays['affine_mat'] # (b, nv, 4, 4)
        alpha_inter_ratio_lod0 = sample_rays['alpha_inter_ratio_lod0'] # float
        # alpha_inter_ratio_lod0 = 1. # NOTE(lihe): debug !!!!!!!!!!!!!
        iter_step = sample_rays['step'] # int

        # decode 3d prior
        if self.use_3d_prior:
            latents_3d = sample_rays['latents_3d']
            # add noise
            # print("===latents 3d====", latents_3d.shape)
            # t_[0] = 500 # NOTE: debug !!!
            # print("===latents timsteps====", t_.shape, t_)
            if self.lazy_3d:
                t_3d = torch.where(t_ > self.lazy_t, torch.zeros_like(t_) + self.lazy_t, t_)
            else:
                t_3d = t_

            if self.training:
                noisy_latents_3d = self.ddpm_3d.q_sample(latents_3d, t_3d)
            else:
                noisy_latents_3d = latents_3d # latent3d is noisy latents during inference
            # self.options.nerf_level = "fine"
            # self.options_3d.nerf_level = "coarse"
            self.options_3d.nerf_level = "fine"
        else:
            noisy_latents_3d = None
        
        # also generate sdf
        if self.sdf_gen:
            noise_scheduler = sample_rays['noise_scheduler']
            # NOTE(lihe): the sdf data is not [-1, 1], figure out if this is a bug
            gt_sdf = sample_rays['gt_sdf'][..., -1:]
            sdf_query_pts = sample_rays['gt_sdf'][..., :-1]
            noise = torch.randn_like(gt_sdf)
            if self.training:
                noisy_sdf = noise_scheduler.add_noise(gt_sdf, noise, t_) # [b, N, 1]
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