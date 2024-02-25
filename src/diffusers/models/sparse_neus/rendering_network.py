# the codes are partly borrowed from IBRNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.checkpoint import checkpoint

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)


# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x * weight, dim=2, keepdim=True)
    var = torch.sum(weight * (x - mean) ** 2, dim=2, keepdim=True)
    return mean, var


class GeneralRenderingNetwork(nn.Module):
    """
    This model is not sensitive to finetuning
    """

    def __init__(self, in_geometry_feat_ch=8, in_rendering_feat_ch=56, anti_alias_pooling=True, add_temb=False, in_rgb_ch=3, regress_rgb=False,
                 pos_enc_dim=0, debug_regress=False):
        super(GeneralRenderingNetwork, self).__init__()

        self.in_geometry_feat_ch = in_geometry_feat_ch
        self.in_rendering_feat_ch = in_rendering_feat_ch
        self.anti_alias_pooling = anti_alias_pooling
        self.in_rgb_ch = in_rgb_ch
        self.debug_regress = debug_regress

        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)

        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_rendering_feat_ch + in_rgb_ch),
                                        activation_func)
        
        base_input_ch = (in_rendering_feat_ch + in_rgb_ch) * 3 + in_geometry_feat_ch + pos_enc_dim
        self.add_temb = add_temb
        self.regress_rgb = regress_rgb
        if add_temb:
            self.emb_proj = nn.Linear(320, 16)
            base_input_ch += 16

        self.base_fc = nn.Sequential(nn.Linear(base_input_ch, 64),
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func)
        if not debug_regress:
            self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                        activation_func,
                                        nn.Linear(32, 33),
                                        activation_func,
                                        )

            self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                        activation_func,
                                        nn.Linear(32, 1),
                                        nn.Sigmoid()
                                        )
        else:
            self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                        activation_func,
                                        nn.Linear(32, 32),
                                        activation_func,
                                        )

            self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                        activation_func,
                                        nn.Linear(32, 32),
                                        )
        if not self.regress_rgb and not debug_regress:
            self.rgb_fc = nn.Sequential(nn.Linear(32 + 1 + 4, 16),
                                        activation_func,
                                        nn.Linear(16, 8),
                                        activation_func,
                                        nn.Linear(8, 1))
        elif self.debug_regress:
            self.rgb_fc = nn.Sequential(nn.Linear(32, 16),
                                        activation_func,
                                        nn.Linear(16, 8),
                                        activation_func,
                                        nn.Linear(8, in_rgb_ch))
        else:
            self.rgb_fc = nn.Sequential(nn.Linear(32 + 1 + 4, 16),
                                        activation_func,
                                        nn.Linear(16, 8),
                                        activation_func,
                                        nn.Linear(8, in_rgb_ch))
            self.blend_fc = nn.Sequential(nn.Linear(32 + 1 + 4, 16),
                                        activation_func,
                                        nn.Linear(16, 8),
                                        activation_func,
                                        nn.Linear(8, 1))
            
        self.use_checkpoint = False
        

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)
        # if self.regress_rgb:
        #     self.blend_fc.apply(weights_init)
    
    def enable_gradient_checkpointing(self):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        # if not self._supports_gradient_checkpointing:
        #     raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        # self.apply(partial(self._set_gradient_checkpointing, value=True))
        self.use_checkpoint = True

    def forward(self, geometry_feat, rgb_feat, ray_diff, mask, pts_mask=None, emb=None):
        '''
        :param geometry_feat: geometry features indicates sdf  [n_rays, n_samples, n_feat]
        :param rgb_feat: rgbs and image features [n_views, n_rays, n_samples, n_feat]
        :param ray_diff: ray direction difference [n_views, n_rays, n_samples, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_views, n_rays, n_samples]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''
        rgb_feat = rgb_feat.permute(1, 2, 0, 3).contiguous()
        ray_diff = ray_diff.permute(1, 2, 0, 3).contiguous()
        mask = mask[:, :, :, None].permute(1, 2, 0, 3).contiguous()
        num_views = rgb_feat.shape[2]
        geometry_feat = geometry_feat[:, :, None, :].repeat(1, 1, num_views, 1)

        direction_feat = self.ray_dir_fc(ray_diff)
        # print('\nDEBUG 0: ', rgb_feat.shape, direction_feat.shape)
        rgb_in = rgb_feat[..., :self.in_rgb_ch]
        rgb_feat = rgb_feat + direction_feat

        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8)
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

        # compute mean and variance across different views for each point
        mean, var = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean, var], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

        # NOTE(lihe): cat global emb feats
        if emb is not None and self.add_temb:
            emb = self.emb_proj(emb).view(1, 1, 1, -1).repeat(globalfeat.shape[0], globalfeat.shape[1], 1, 1)
            globalfeat = torch.cat([globalfeat, emb], dim=-1)
        
        x = torch.cat([geometry_feat, globalfeat.expand(-1, -1, num_views, -1), rgb_feat],
                    dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat+n_geo_feat]

        # NOTE(lihe): to reduce memory, we dont feed all pts to fc, mask out some pts using pts_mask
        debug = True
        if debug:
            assert pts_mask is not None
            n_rays, n_samples, n_views, c = x.shape
            x = x.view(n_rays*n_samples, n_views, c)
            if not self.regress_rgb and not self.debug_regress:
                out_x = torch.ones(n_rays*n_samples, n_views, 1).to(x.dtype).to(x.device) * -1e9
            elif self.debug_regress:
                out_x = torch.ones(n_rays*n_samples, self.in_rgb_ch).to(x.dtype).to(x.device) * 0. # TODO(lihe): figure out whether therer will be some bugs if initilized to zero
            else:
                out_x = torch.ones(n_rays*n_samples, n_views, self.in_rgb_ch + 1).to(x.dtype).to(x.device) * -1e9
                out_x[..., 1:] = 0. # initialize weights to -1e9, rgb to zero
                # out_x = torch.ones(n_rays*n_samples, n_views, self.in_rgb_ch).to(x.dtype).to(x.device) * 0.
            x = x[pts_mask]
            mask_ = mask.view(n_rays*n_samples, n_views, 1)[pts_mask] # [n_rays, n_samples, n_views, 1]
            weight_ = weight.view(n_rays*n_samples, n_views, -1)[pts_mask]
            ray_diff = ray_diff.view(n_rays*n_samples, n_views, -1)[pts_mask]

            if self.use_checkpoint and self.training:
                # NOTE(lihe): no use anymore
                x = checkpoint(self.base_fc, x)

                x_vis = checkpoint(self.vis_fc, x * weight_)

                x_res, vis = torch.split(x_vis, [x_vis.shape[-1] - 1, 1], dim=-1)
                vis = F.sigmoid(vis) * mask_
                x = x + x_res
                vis = checkpoint(self.vis_fc2, x * vis)
                vis = vis * mask_

                # rgb computation
                x = torch.cat([x, vis, ray_diff], dim=-1)

                # color blending or directly predict color
                if not self.regress_rgb:
                    x = checkpoint(self.rgb_fc, x)
                    x = x.masked_fill(mask_ == 0, -1e9)
                    out_x[pts_mask] = x
                else:
                    pred_rgb = checkpoint(self.rgb_fc, x)
                    pred_w = checkpoint(self.blend_fc, x)
                    mask_rgb = mask_.repeat(1, pred_rgb.shape[-1])
                    pred_rgb.masked_fill(mask_rgb == 0, 0.)
                    pred_w.masked_fill(mask_ == 0, -1e9)
                    out_x[pts_mask, :, 0:1] = pred_w # NOTE(lihe): bug
                    out_x[pts_mask, :, 1:] = pred_rgb
                x = out_x.view(n_rays, n_samples, n_views, -1)
            else:
                x = self.base_fc(x)

                # debug regress
                if self.debug_regress:
                    # x_vis = self.vis_fc(x * weight_).mean(dim=1) # N', c
                    x = self.vis_fc(x).mean(dim=1) # N', c
                    x = self.vis_fc2(x)
                    pred_rgb = self.rgb_fc(x)
                    out_x[pts_mask] = pred_rgb
                    x = out_x.view(n_rays, n_samples, -1)
                else:
                    x_vis = self.vis_fc(x * weight_)
                    x_res, vis = torch.split(x_vis, [x_vis.shape[-1] - 1, 1], dim=-1)
                    vis = F.sigmoid(vis) * mask_
                    x = x + x_res
                    vis = self.vis_fc2(x * vis) * mask_

                    # rgb computation
                    x = torch.cat([x, vis, ray_diff], dim=-1)

                    # color blending or directly predict color
                    if not self.regress_rgb:
                        x = self.rgb_fc(x)
                        x = x.masked_fill(mask_ == 0, -1e9)
                        out_x[pts_mask] = x
                    else:
                        pred_rgb = self.rgb_fc(x)
                        pred_w = self.blend_fc(x)
                        mask_rgb = mask_.repeat(1, 1, pred_rgb.shape[-1])
                        pred_rgb.masked_fill(mask_rgb == 0, 0.)
                        pred_w.masked_fill(mask_ == 0, -1e9)
                        out_x[pts_mask, :, 0:1] = pred_w
                        out_x[pts_mask, :, 1:] = pred_rgb
                        # out_x[pts_mask, :, :] = pred_rgb
                    x = out_x.view(n_rays, n_samples, n_views, -1)
        else:
            x = self.base_fc(x)

            x_vis = self.vis_fc(x * weight)
            x_res, vis = torch.split(x_vis, [x_vis.shape[-1] - 1, 1], dim=-1)
            vis = F.sigmoid(vis) * mask
            x = x + x_res
            vis = self.vis_fc2(x * vis) * mask

            # rgb computation
            x = torch.cat([x, vis, ray_diff], dim=-1)
            x = self.rgb_fc(x)

        if not self.regress_rgb and not self.debug_regress:
            # use color blending
            x = x.masked_fill(mask == 0, -1e9)
            blending_weights_valid = F.softmax(x, dim=2)  # color blending
            rgb_out = torch.sum(rgb_in * blending_weights_valid, dim=2)
            # rgb_out = torch.mean(rgb_in, dim=2)
            # rgb_out = rgb_out * 0 + 0.5

            mask = mask.detach().to(rgb_out.dtype)  # [n_rays, n_samples, n_views, 1]
            mask = torch.sum(mask, dim=2, keepdim=False)
            mask = mask >= 2  # more than 2 views see the point
            mask = torch.sum(mask.to(rgb_out.dtype), dim=1, keepdim=False)
            valid_mask = mask > 8  # valid rays, more than 8 valid samples
            
            return rgb_out, valid_mask  # (N_rays, n_samples, 3), (N_rays, 1)
        elif self.debug_regress:
            # x = x.masked_fill(mask == 0, 0.)
            rgb_out = x
            mask = mask.detach().to(rgb_out.dtype)  # [n_rays, n_samples, n_views, 1]
            mask = torch.sum(mask, dim=2, keepdim=False)
            mask = mask >= 2  # more than 2 views see the point
            mask = torch.sum(mask.to(rgb_out.dtype), dim=1, keepdim=False)
            valid_mask = mask > 8  # valid rays, more than 8 valid samples
            
            return rgb_out, valid_mask  # (N_rays, n_samples, 3), (N_rays, 1)
        else:
            rgb_in = x[..., 1:]
            blend_w = x[..., 0:1]
            blending_weights_valid = F.softmax(blend_w, dim=2)  # color blending
            rgb_out = torch.sum(rgb_in * blending_weights_valid, dim=2)

            mask = mask.detach().to(rgb_out.dtype)  # [n_rays, n_samples, n_views, 1]
            mask = torch.sum(mask, dim=2, keepdim=False)
            mask = mask >= 2  # more than 2 views see the point
            mask = torch.sum(mask.to(rgb_out.dtype), dim=1, keepdim=False)
            valid_mask = mask > 8  # valid rays, more than 8 valid samples
            return rgb_out, valid_mask  # (N_rays, n_samples, 3), (N_rays, 1)
            # not use color blending
