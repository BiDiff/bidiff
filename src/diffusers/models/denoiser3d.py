import torch

# ! amazing!!!! autograd.grad with set_detect_anomaly(True) will cause memory leak
# ! https://github.com/pytorch/pytorch/issues/51349
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from inplace_abn import InPlaceABN
from diffusers.models.sparse_neus.sparse_sdf_network import SparseSdfNetwork
from diffusers.models.sparse_neus.fields import SingleVarianceNetwork
from diffusers.models.sparse_neus.rendering_network import GeneralRenderingNetwork
from diffusers.models.sparse_neus.sparse_neus_renderer import SparseNeuSRenderer
from diffusers.models.sparse_neus.conv_modules import ConvBnReLU
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

from diffusers.models.openaimodel.openaimodel import build_nerf_encoder, timestep_embedding
from diffusers.models.openaimodel.code_test import DEBUG_SAVE_ID
from diffusers.models.sparse_neus.image_encoder import DinoWrapper, LiftTransformer
from .mmdiffusion3d import MultiviewEncoder

from PIL import Image
import imageio
from typing import Any, Callable, Dict, Optional

#NOTE(lihe): dpm solver sampling 
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

from diffusers.models.shap_e.shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
def render_latents(xm, latent, render_mode = 'nerf', size = 64, device=None, name='train'):
    cameras = create_pan_cameras(size, device)
    images = decode_latent_images(xm, latent[0], cameras, rendering_mode=render_mode)
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

class Denoiser3D(nn.Module):
    """
    3D Denoiser using NeuS
    """
    def __init__(self, ch=56, res=64, num_views=8, time_embed_ch=512, use_viewdirs=False, img_ch=3, # converter
                 voxel_size=0.7/95., vol_dims=[96, 96, 96], hidden_dim=128, cost_type='variance_mean', # sparsesdfnet
                 d_pyramid_feature_compress=16, regnet_d_out=16, num_sdf_layers=4, multires=6, # 
                 init_val=0.2, # variance
                 in_geometry_feat_ch=16, in_rendering_feat_ch=56, anti_alias_pooling=True, # rendering network
                 n_samples=48, n_importance=32, n_outside=0, perturb=1.0, alpha_type='div',
                 partial_vol_origin=[-0.35, -0.35, -0.35], scale=0.35, pred_density=False,
                 use_sd_feat=True, add_temb=True, temb_channels=320, encoder_hid_dim=4096,
                 cond_dep=False, # if use pred depth as another condition signal
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
                 input_res=64,
                 render_res=64,
                 ) -> None:
        super(Denoiser3D, self).__init__()
        # 1. build pyramid featurenet, we now only implement one stage
        if use_sd_feat:
            self.sd_featurenet = SDFeatureNet() # TODO(lihe): support Stable Diffusion, currently only support IF
        else:
            self.featurenet = FeatureNet(img_ch=img_ch)
        if use_all_feat:
            self.use_diff_featuernet = use_diff_featuernet
            self.use_dino_feat = use_dino_feat
            if self.use_diff_featuernet:
                self.diff_featurenet, self.nerf_time_embed, self.nerf_encoder_param = build_nerf_encoder()
            elif self.use_dino_feat:
                self.dino_featurenet = MultiviewEncoder(
                    out_channels=ch, pretrained_image_encoder='facebook/dino-vitb16', freeze_pretrained_image_encoder=True)
                # self.lift_transformer = LiftTransformer(
                #     inner_dim=512, image_feat_dim=768, camera_embed_dim=32,
                #     trans_feat_low_res=32, trans_feat_high_res=64, trans_feat_dim=ch,
                #     num_layers=4, num_heads=8, lift_mode='triplane', num_views=num_views)
            else:
                self.featurenet = FeatureNet(img_ch=img_ch * 2 if not abandon_sdf_x0 else img_ch, use_featurenet_view_embed=use_featurenet_view_embed)
            self.sd_featurenet = None
            # self.fuse_layer = nn.Conv2d(ch*2 + img_ch*2 if foundation_model=='if' else ch*2 + img_ch, ch, 1) # if pred mean and var
            # self.fuse_layer = nn.Conv2d(ch + img_ch, ch, 1) # if pred mean and var

        if use_all_feat:
            assert use_sd_feat, "please set use_sd_feat to True since we already use it when using all features"
        if use_dino_feat:
            assert not use_diff_featuernet
        # pos encoder if needed
        if pos_enc:
            from diffusers.models.sparse_neus.pos_encoder import PositionalEncoding
            self.code = PositionalEncoding(num_freqs=6, d_in=3, freq_factor=1.5, include_input=True)
        else:
            self.code = None
        # abandon_sdf_x0
        self.abandon_sdf_x0 = abandon_sdf_x0
        self.debug_regress = debug_regress
        
        # 2. build sdf network
        self.sdf_network = SparseSdfNetwork(lod=0, ch_in=ch, voxel_size = voxel_size, vol_dims = vol_dims, 
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
        if use_resnetfc:
            from .resnetfc import ResnetFC
            self.rendering_network = ResnetFC(d_out=img_ch, d_in=self.code.d_out + 3,
                                              n_blocks=3, 
                                              d_latent=img_ch * 8 + in_rendering_feat_ch + in_geometry_feat_ch,
                                            #   d_latent=in_rendering_feat_ch + in_geometry_feat_ch,
                                              d_hidden=64, time_embed_ch=64)
        else:
            self.rendering_network = GeneralRenderingNetwork(in_geometry_feat_ch=in_geometry_feat_ch, 
                                                            #NOTE(lihe): only feed predicted x0 to rendering network
                                                            in_rendering_feat_ch=in_rendering_feat_ch,
                                                            anti_alias_pooling=anti_alias_pooling, 
                                                            add_temb=add_temb, # NOTE(lihe): ablate
                                                            in_rgb_ch=img_ch,
                                                            regress_rgb=regress_rgb,
                                                            pos_enc_dim=0 if not pos_enc else self.code.d_out,
                                                            debug_regress=debug_regress
                                                            )
        # self.rendering_network.enable_gradient_checkpointing()
        # 3. build renderer
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
            rgb_ch=img_ch,
            pos_code=self.code,
            use_resnetfc=use_resnetfc,
            direct_use_3d=direct_use_3d,
            use_3d_prior=use_3d_prior,
            geo_attn_mask=geo_attn_mask)
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
        # loss weight adapted from original sparseneus
        self.fg_bg_weight = 0.
        self.anneal_start = 0
        self.anneal_end = 25000
        self.sdf_sparse_weight = 0.02
        self.sdf_igr_weight = 0.1
        self.sdf_decay_param = 100
        self.pred_density = pred_density
        self.use_sd_feat = use_sd_feat
        self.cond_dep = cond_dep
        self.regress_rgb = regress_rgb
        self.img_ch = img_ch
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
            depth_error = depth_pred[color_mask] - true_depth[color_mask]
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
                    save_image(save_imgs, 'debug/train_deps{}.png'.format(DEBUG_SAVE_ID) if not extra_view else 'debug/train_deps_extra{}.png'.format(DEBUG_SAVE_ID), nrow=4 if not extra_view else self.extra_view_num)
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
        return loss, losses

    
    def forward(self, feats, t, query_ids, sample_rays, depth=None, return_sigma=False, debug=False, voxels=None, vis_iter=200,
                unet=None, encoder_hidden_states=None, model_3d=None, dpm_solver_scheduler=None,
                noisy_sdf=None, mesh_save_path=None, only_train=None, background_rgb=-1,
                pyramid_feats=None, conditional_features_lod0_cache=[], new_batch=False, feats_64=None):
        """
        Args:
            feats: [B, 8, C, res, res]
            t: [B] timesteps
            query_ids: [B] (0-7) indicates which view is the query view
            sample_rays: dict('rays_o': [B, 8*num_rays, 3], 'rays_d': [B, 8*num_rays, 3], 'near': [], 'far': [])
            depth: [B, 8, res, res] if use GT depth correspondence
        """
        # TODO(lihe):
        # [x] add timesteps into sparseneus
        # [x] figure out alpha_inter_ratio
        # [x] add positional encoding for rgb prediction
        # [x] add positional encoding for sdf prediction
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
        
        # 1. get volume feats
        # NOTE(lihe): use pre-trained SD to generate multi-view features
        if not (only_train in ['recon_net'] and not self.training and pyramid_feats is not None):
            if self.use_sd_feat:
                conditional_features_lod0_cache = []
            if self.use_sd_feat or self.use_dino_feat:
                if only_train in ['recon_net']:
                    model_pred = feats.view(b*num_views, -1, res, res)
                    if self.debug_sd_feat:
                        pred_x0 = model_pred
                    if not self.abandon_sdf_x0:
                        pyramid_feats = pred_x0
                    else:
                        pyramid_feats = None
                else:
                    with torch.no_grad():
                        # model_pred, pyramid_sd_feats = unet(feats.view(b*num_views, -1, res, res), t,
                        #                 encoder_hidden_states=encoder_hidden_states,
                        #                 return_feats=True)
                        model_pred = unet(feats.view(b*num_views, -1, res, res), t,
                                        encoder_hidden_states=encoder_hidden_states,
                                        return_feats=False).sample
                    # NOTE(lihe): we only use the predicted clean x0 as additional feature
                    if self.debug_sd_feat:
                        ## NOTE(lihe): rewrite step func
                        noise_scheduler = sample_rays['noise_scheduler'] # NOTE(lihe): use noise scheduler to transfer noise to clean x0
                        pred_x0 = noise_scheduler.step_w_device(
                                model_pred, t, feats.view(b*num_views, -1, res, res), device=feats.device
                            )
                        if not self.abandon_sdf_x0:
                            pyramid_feats = pred_x0
                        else:
                            pyramid_feats = None
                        if (self.training and iter_step % vis_iter == 0):
                            with torch.no_grad():
                                save_x0 = torch.cat([feats.view(b*num_views, -1, res, res), pred_x0], dim=0)
                                save_x0 = (save_x0 + 1) /2 
                                save_image(save_x0[:, :3], 'debug/train_x0.png', nrow=4)
                        elif not self.training:
                            save_x0 = torch.cat([feats.view(b*num_views, -1, res, res), pred_x0], dim=0)
                            save_x0 = (save_x0 + 1) /2
                            os.makedirs('debug/sample_x0{}'.format(DEBUG_SAVE_ID), exist_ok=True)
                            save_image(save_x0[:, :3], 'debug/sample_x0{}/step_x0_{}.png'.format(DEBUG_SAVE_ID, t[0]), nrow=4)
            else:
                pyramid_feats = self.featurenet.obtain_pyramid_feature_maps(feats.view(b*num_views, c, res, res)) # (B*8, 56, H, W)
            
            if self.use_all_feat:
                if self.use_diff_featuernet:
                    nerf_t_emb = timestep_embedding(t, self.nerf_encoder_param.model_channels, repeat_only=False)
                    nerf_t_emb = self.nerf_time_embed(nerf_t_emb)
                    h = feats.view(b * num_views, c, res, res)
                    for j, module in enumerate(self.diff_featurenet):
                        h = module(h, nerf_t_emb, encoder_hidden_states)
                        if not self.training and j == 1:
                            fea_c = h.shape[1]
                            save_pyramid_feats = torch.zeros((b*num_views, 3, res, res), dtype=torch.float32, device=h.device)
                            save_pyramid_feats[:, 0] = h[:, :fea_c//3].sum(dim=1)
                            save_pyramid_feats[:, 1] = h[:, fea_c//3:fea_c//3*2].sum(dim=1)
                            save_pyramid_feats[:, 2] = h[:, fea_c//3*2:].sum(dim=1)
                            save_pyramid_feats = torch.sigmoid(save_pyramid_feats)
                            os.makedirs('debug/pyramid_feats_layer{}'.format(DEBUG_SAVE_ID), exist_ok=True)
                            save_image(save_pyramid_feats, 'debug/pyramid_feats_layer{}/step_{}.png'.format(DEBUG_SAVE_ID, t[0]), nrow=4)
                    pyramid_feats = h
                else:
                    pyramid_feats = torch.cat([feats.view(b*num_views, c, res, res), pyramid_feats], dim=1) if not self.abandon_sdf_x0 else feats.view(b*num_views, c, res, res)
                    if self.use_dino_feat:
                        with torch.no_grad():
                            dino_input = F.interpolate(feats.view(b * num_views, c, res, res), size=(512, 512), mode='bilinear', align_corners=False)
                        _out_size = self.dino_featurenet.out_size
                        pyramid_feats = self.dino_featurenet(dino_input.view(b, num_views, c, 512, 512)).view(b*num_views, 56, _out_size, _out_size) # (b*num_views, c=768)
                        pyramid_feats = F.interpolate(pyramid_feats, size=(res, res), mode='bilinear', align_corners=False) # (bv, 56, 64, 64)
                        # dino_lift_feats = self.lift_transformer(
                        #     dino_feats.view(b, num_views, *dino_feats.shape[-2:])) # (b*num_views, c=4*4+1, 768)
                    else:
                        pyramid_feats = self.featurenet.obtain_pyramid_feature_maps(pyramid_feats)


                if not self.training:
                    fea_c = pyramid_feats.shape[1]
                    save_pyramid_feats = torch.zeros((b*num_views, 3, res, res), dtype=torch.float32, device=pyramid_feats.device)
                    save_pyramid_feats[:, 0] = pyramid_feats[:, :fea_c//3].sum(dim=1)
                    save_pyramid_feats[:, 1] = pyramid_feats[:, fea_c//3:fea_c//3*2].sum(dim=1)
                    save_pyramid_feats[:, 2] = pyramid_feats[:, fea_c//3*2:].sum(dim=1)
                    save_pyramid_feats = torch.sigmoid(save_pyramid_feats)
                    os.makedirs('debug/pyramid_feats{}'.format(DEBUG_SAVE_ID), exist_ok=True)
                    save_image(save_pyramid_feats, 'debug/pyramid_feats{}/step_{}.png'.format(DEBUG_SAVE_ID, t[0]), nrow=4)

            pyramid_feats = pyramid_feats.view(b, num_views, *pyramid_feats.shape[1:]) # (B, 8, 56, H, W)
        
        else:
            model_pred = feats.view(b*num_views, -1, res, res)
            if self.debug_sd_feat:
                pred_x0 = model_pred
                
        partial_vol_origin = torch.tensor(self.partial_vol_origin).unsqueeze(0).to(feats.device) # [-1., -1., -1] (1, 3)

        # NOTE(lihe): explore image300M
        noisy_latents_3d_prev = None
        if self.model_type == 'image300M' and model_3d is not None and self.use_3d_prior:
            # NOTE(lihe): implement lazy training && sampling strategy
            with torch.no_grad():
                # split input data
                guidance_scale = 3.0
                batch_one = 1 if self.training else 2
                assert t_3d.shape[0] == batch_one, "we only support bs=1(train)/2(test) using 3d prior."
                if t_3d[0] == self.lazy_t:
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
                if not self.training:
                    save_image(cond_image, 'debug/eval_latent_cond_image.png', nrow=4)
                    print(f"====== testing, save img300M results at timestep t_ : {t_[0]}, t_3d: {t_3d[0]} =====")
                    render_latents(self.xm, noisy_latents_3d, device=noisy_latents_3d.device, name='eval')

        loss_all = 0.
        render_color = []
        render_dep = []
        pred_clean_sdf = []
        all_attn_mask = []
        for bs_id in range(b):
            if only_train in ['recon_net'] and not self.training and len(conditional_features_lod0_cache) == b: # 
                conditional_features_lod0 = conditional_features_lod0_cache[bs_id]
            else:
                with torch.autocast('cuda', enabled=False):
                    conditional_features_lod0 = self.sdf_network.get_conditional_volume(
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
                if only_train in ['recon_net'] and not self.training and len(conditional_features_lod0_cache) < b: # 
                    conditional_features_lod0_cache.append(conditional_features_lod0)
                
            con_volume_lod0 = conditional_features_lod0['dense_volume_scale0'] # (1, C, 96, 96, 96)
            con_valid_mask_volume_lod0 = conditional_features_lod0['valid_mask_volume_scale0'] # (1, 1, 96, 96, 96)
            coords_lod0 = conditional_features_lod0['coords_scale0']  # [1,3,wX,wY,wZ]

            # NOTE(lihe) iteratively render every view to reduce memory
            render_out = {}
            if only_train not in ['recon_net']:
                color_fine_list, color_fine_mask_list, depth_list, sdf_list = [], [], [], []
                attn_mask_list = []
                gradient_error_fine = 0.
                for v_id in range(num_views):
                    start = v_id * res * res
                    end = (v_id + 1) * res * res
                    out = self.sdf_renderer.render(
                        rays_o[bs_id][start:end], rays_d[bs_id][start:end], near[bs_id], far[bs_id],
                        self.sdf_network,
                        self.rendering_network,
                        background_rgb=background_rgb, # NOTE(lihe): fixed background bug
                        alpha_inter_ratio=alpha_inter_ratio_lod0,
                        # * related to conditional feature
                        lod=0,
                        conditional_volume=con_volume_lod0,
                        conditional_valid_mask_volume=con_valid_mask_volume_lod0,
                        # * 2d feature maps
                        feature_maps=pyramid_feats[bs_id],
                        color_maps=feats[bs_id] if not self.blend_x0 else pred_x0.view(*feats.shape)[bs_id], # NOTE(lihe): blend x0
                        w2cs=w2cs[bs_id],
                        intrinsics=intrinsics[bs_id],
                        img_wh=[res, res],
                        query_c2w=c2ws[bs_id][v_id:v_id+1], # [1, 4, 4]
                        if_general_rendering=True,
                        if_render_with_grad=True,
                        vol_dims=torch.tensor(self.vol_dims).to(feats.device),
                        partial_vol_origin=partial_vol_origin[0], # [3]
                        vol_size=self.voxel_size,
                        pred_density=self.pred_density, # NOTE(lihe):!!!!!!!!!!!!!!!!!
                        emb=emb[bs_id:bs_id+1] if emb is not None else None,
                        t=t_[bs_id:bs_id+1],
                        noisy_latents_3d=noisy_latents_3d[bs_id:bs_id+1] if noisy_latents_3d is not None else None,
                        xm=self.xm,
                        options_3d=self.options_3d,
                    )
                    # NOTE(lihe): support learn bg
                    if self.learn_bg_color:
                        out['color_fine'][~out['color_fine_mask'][..., 0]] = self.sdf_renderer.background
                    
                    if self.geo_attn_mask:
                        attn_mask_list.append(out['attn_mask']) # N_ray, N_v, H, W

                    color_fine_list.append(out['color_fine'])
                    color_fine_mask_list.append(out['color_fine_mask'])
                    depth_list.append(out['depth'])
                    gradient_error_fine = gradient_error_fine + out['gradient_error_fine']
                    sdf_list.append(out['sdf'])
                
                # aggregate
                render_out['color_fine'] = torch.cat(color_fine_list, dim=0)
                render_out['color_fine_mask'] = torch.cat(color_fine_mask_list, dim=0)
                render_out['depth'] = torch.cat(depth_list, dim=0)
                render_out['gradient_error_fine'] = gradient_error_fine / num_views
                render_out['sdf'] = torch.cat(sdf_list, dim=0) if not self.pred_density else None
                render_out['color_mlp'] = None
                render_out['color_mlp_mask'] = None
                if self.geo_attn_mask:
                    render_out['attn_mask'] = torch.stack(attn_mask_list, dim=0) # nv, nr, nv, h, w

            #### extra view rendering
            if self.extra_view_num > 0 and (self.training or only_train in ['recon_net']):
                # random choose x views
                extra_render_out = {}
                extra_color_fine_list, extra_color_fine_mask_list, extra_depth_list, extra_sdf_list = [], [], [], []
                extra_gradient_error_fine = 0.
                if not self.training and only_train in ['recon_net']:
                    extra_view_num = extra_rays_o.shape[1] // res // res
                else:
                    extra_view_num = self.extra_view_num
                for v_id in range(extra_view_num):
                    start = v_id * res * res
                    end = (v_id + 1) * res * res
                    extra_out = self.sdf_renderer.render(
                        extra_rays_o[bs_id][start:end], extra_rays_d[bs_id][start:end], extra_near[bs_id], extra_far[bs_id],
                        self.sdf_network,
                        self.rendering_network,
                        background_rgb=background_rgb, # NOTE(lihe): fixed background bug
                        alpha_inter_ratio=alpha_inter_ratio_lod0,
                        # * related to conditional feature
                        lod=0,
                        conditional_volume=con_volume_lod0,
                        conditional_valid_mask_volume=con_valid_mask_volume_lod0,
                        # * 2d feature maps
                        feature_maps=pyramid_feats[bs_id],
                        color_maps=feats[bs_id] if not self.blend_x0 else pred_x0.view(*feats.shape)[bs_id], # NOTE(lihe): blend x0
                        w2cs=w2cs[bs_id], # NOTE(lihe): should be input 8 view
                        intrinsics=intrinsics[bs_id],
                        img_wh=[res, res],
                        query_c2w=extra_c2ws[bs_id][v_id:v_id+1], # [1, 4, 4]
                        if_general_rendering=True,
                        if_render_with_grad=True,
                        vol_dims=torch.tensor(self.vol_dims).to(feats.device),
                        partial_vol_origin=partial_vol_origin[0], # [3]
                        vol_size=self.voxel_size,
                        pred_density=self.pred_density, # NOTE(lihe):!!!!!!!!!!!!!!!!!
                        emb=emb[bs_id:bs_id+1] if emb is not None else None,
                        t=t_[bs_id:bs_id+1],
                        noisy_latents_3d=noisy_latents_3d[bs_id:bs_id+1] if noisy_latents_3d is not None else None,
                        xm=self.xm,
                        options_3d=self.options_3d,
                    )
                    # NOTE(lihe): support learn bg
                    if self.learn_bg_color:
                        extra_out['color_fine'][~extra_out['color_fine_mask'][..., 0]] = self.sdf_renderer.background
                    extra_color_fine_list.append(extra_out['color_fine'])
                    extra_color_fine_mask_list.append(extra_out['color_fine_mask'])
                    extra_depth_list.append(extra_out['depth'])
                    extra_gradient_error_fine = extra_gradient_error_fine + extra_out['gradient_error_fine']
                    # extra_sdf_list.append(extra_out['sdf'])

                extra_render_out['color_fine'] = torch.cat(extra_color_fine_list, dim=0)
                extra_render_out['color_fine_mask'] = torch.cat(extra_color_fine_mask_list, dim=0)
                extra_render_out['depth'] = torch.cat(extra_depth_list, dim=0)
                extra_render_out['gradient_error_fine'] = extra_gradient_error_fine / self.extra_view_num
                # extra_render_out['sdf'] = torch.cat(extra_sdf_list, dim=0) if not self.pred_density else None
                extra_render_out['color_mlp'] = None
                extra_render_out['color_mlp_mask'] = None
            
            # predict sdf gen
            if self.sdf_gen:
                denoised_sdf = self.sdf_network.sdf(sdf_query_pts[bs_id], con_volume_lod0, lod=0, split_batch=None, 
                                                     emb=emb[bs_id:bs_id+1] if emb is not None else None,)
                denoised_sdf = denoised_sdf['sdf_pts_scale0']
                if self.training:
                    sdf_denoise_loss = F.l1_loss(denoised_sdf, gt_sdf[bs_id], reduction='mean')
                else:
                    pred_clean_sdf.append(denoised_sdf) # N,1
                if (self.training and iter_step % vis_iter == 0):
                    print(f"====Savomg predicted sdf at timestep {t_[0]}=====")
                    np.save('debug/gt_sdf.npy', gt_sdf[bs_id].detach().cpu().numpy())
                    np.save('debug/pred_sdf.npy', denoised_sdf.detach().cpu().numpy())
                    np.save('debug/query_pts.npy', sdf_query_pts[bs_id].detach().cpu().numpy())
                    np.save('debug/voxel.npy', voxels[bs_id].detach().cpu().numpy())
                    np.save('debug/noisy_sdf.npy', noisy_sdf[bs_id].detach().cpu().numpy())
                
            if only_train not in ['recon_net'] and not self.training:
                depth_pred = render_out['depth']
                pred_dep = depth_pred * 0.5
                pred_dep = pred_dep.view(-1, 1, res, res)
                os.makedirs('debug/test_deps{}'.format(DEBUG_SAVE_ID), exist_ok=True)
                save_image(pred_dep, 'debug/test_deps{}/{}.png'.format(DEBUG_SAVE_ID, t[0]) , nrow=4)

            if only_train in ['recon_net'] and not self.training and mesh_save_path is not None:
                extra_color_fine = extra_render_out['color_fine']
                pred_imgs = (extra_color_fine + 1) / 2.
                save_imgs = pred_imgs.view(-1, res, res, c).permute(0, 3, 1, 2)
                recon_save_dir = os.path.dirname(mesh_save_path)
                os.makedirs(recon_save_dir, exist_ok=True)
                if save_imgs.shape[0] > 48:
                    save_frames = []
                    for j, save_img in enumerate(save_imgs[:, :3]):
                        img = save_img.mul(255).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                        save_frames.append(img)
                    imageio.mimsave(mesh_save_path[:-4] + '.gif', save_frames, 'GIF', duration=40)
                    imageio.mimsave(mesh_save_path[:-4] + '.mp4', save_frames, fps=25)
                else:
                    recon_save_path = mesh_save_path[:-4] + '.png'
                    save_image(save_imgs[:, :3], recon_save_path, nrow=4)

            # NOTE(lihe): move out from renderer since we iteratively render multiple views
            if not self.pred_density:
                pts_random = torch.rand([1024, 3]).float().to(con_volume_lod0.device) * 2 - 1  # normalized to (-1, 1)
                pts_random = pts_random * 0.35 # rescale
                sdf_random = self.sdf_network.sdf(pts_random, con_volume_lod0, lod=0, emb=emb)['sdf_pts_scale%d' % 0]
                render_out['sdf_random'] = sdf_random

            # compute loss
            if self.training:
                loss_lod0, losses = self.cal_losses_sdf(render_out, sample_rays, iter_step, lod=0,
                                                                                bs_id=bs_id, vis_iter=vis_iter,
                                                                                noisy_input=feats)

                if self.extra_view_num > 0:
                    extra_loss_lod0, extra_losses = self.cal_losses_sdf(extra_render_out, sample_rays, iter_step, lod=0,
                                                                                bs_id=bs_id, vis_iter=vis_iter,
                                                                                noisy_input=feats, extra_view=True)
                    loss_lod0 = loss_lod0 + extra_loss_lod0
                    losses.update(extra_losses)
                if self.sdf_gen:
                    loss_lod0 = loss_lod0 + sdf_denoise_loss
                    losses.update({'a_sdf_loss':sdf_denoise_loss.item()})
            else:
                loss_lod0 = 0.
                losses = {}
                
            loss_all = loss_all + loss_lod0

            if only_train not in ['recon_net']:
                # get rendered imgs
                color_fine = render_out['color_fine'] # (nv*w*h, 3)
                color_fine = color_fine.view(num_views, res, res, self.img_ch)
                color_fine = color_fine.permute(0, 3, 1, 2) # (nv, 3, h, w)
                render_color.append(color_fine)

                # get rendered depth
                if self.cond_dep:
                    depth = render_out['depth'] # (nv*w*h, 3)
                    depth = depth.view(num_views, 1, res, res)
                    render_dep.append(depth) # (nv, 1, h, w)
                
                if self.geo_attn_mask:
                    attn_mask = render_out['attn_mask'] # (nv, nr, nv, h, w)
                    all_attn_mask.append(attn_mask)
            else:
                color_fine = extra_render_out['color_fine']
                color_fine = color_fine.view(extra_view_num, res, res, self.img_ch)
                color_fine = color_fine.permute(0, 3, 1, 2) # (nv, 3, h, w)
                render_color.append(color_fine)
                if self.cond_dep:
                    depth = extra_render_out['depth'] # (nv*w*h, 3)
                    depth = depth.view(extra_view_num, 1, res, res)
                    render_dep.append(depth) # (nv, 1, h, w)

            # extract mesh
            # if (self.training and iter_step % vis_iter == 0) or (not self.training and t[bs_id] == 0): # TODO(lihe): check the last t
            if not self.training and t[bs_id] <= 60 and not (only_train in ['recon_net'] and mesh_save_path is None): # TODO(lihe): check the last t, we dont save mesh to save memory during training.
                print("======saving mesh========")
                torch.cuda.empty_cache()
                self.validate_mesh(self.sdf_network,
                                self.sdf_renderer.extract_geometry,
                                conditional_volume=con_volume_lod0, lod=0,
                                threshold=0,
                                # occupancy_mask=con_valid_mask_volume_lod0[0, 0],
                                mode='train' if self.training else 'test', bound_min=[-0.36, -0.36, -0.36], bound_max=[0.36, 0.36, 0.36],
                                meta='',
                                iter_step=iter_step, scale_mat=None,
                                trans_mat=None,
                                emb=emb,
                                save_path=mesh_save_path)
                torch.cuda.empty_cache()

        
        # average 
        loss_all = loss_all / b
        render_color = torch.stack(render_color, dim=0) # (b, nv, 3, h, w)
        if self.cond_dep:
            render_dep = torch.stack(render_dep, dim=0) # (b, nv, 3, h, w)
        
        if self.sdf_gen and not self.training:
            pred_clean_sdf = torch.stack(pred_clean_sdf) # [2, N, 1]
        else:
            pred_clean_sdf = None
        
        if self.geo_attn_mask:
            geo_attn_mask = torch.stack(all_attn_mask) # [b, nv, nr, nv, h, w]
        else:
            geo_attn_mask = None
        
        feature_volume_save_path = sample_rays.get('feature_volume_save_path', None)
        if not self.training and len(conditional_features_lod0_cache) == b and feature_volume_save_path is not None and feature_volume_save_path != '':
            print('Save conditional features')
            save_dict = {}
            for i, save_data in enumerate(conditional_features_lod0_cache):
                save_dict[i] = save_data
            if new_batch:
                save_dict['init_noise'] = feats
            os.makedirs(feature_volume_save_path, exist_ok=True)
            torch.save(save_dict, os.path.join(feature_volume_save_path, 'conditional_features_lod0_{}.pth'.format(t[0])))
        if not self.training and self.sdf_network.mesh_cond and t[0] < 800:
            self.sdf_network.mesh_sdf_flag = True

        if only_train in ['recon_net'] and not self.training and pyramid_feats is not None and len(conditional_features_lod0_cache) == b:
            return render_color, loss_all, losses, pred_x0, noisy_latents_3d, noisy_latents_3d_prev, \
                   pred_clean_sdf, geo_attn_mask, pyramid_feats, conditional_features_lod0_cache
            
        if self.cond_dep:
            return render_color, render_dep, loss_all, losses, pred_x0, noisy_latents_3d, noisy_latents_3d_prev, pred_clean_sdf, geo_attn_mask
        else:   
            return render_color, loss_all, losses, pred_x0, noisy_latents_3d, noisy_latents_3d_prev, pred_clean_sdf, geo_attn_mask
        
        

    
