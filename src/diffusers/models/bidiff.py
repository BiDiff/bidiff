from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging
from .attention_processor import AttentionProcessor, AttnProcessor
from .embeddings import (
    GaussianFourierProjection,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from .modeling_utils import ModelMixin
from .unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UNetMidBlock2DSimpleCrossAttn,
    ResnetDownsampleBlock2D, 
    SimpleCrossAttnDownBlock2D, 
    get_down_block,
)
from .unet_2d_condition import UNet2DConditionModel

from typing import Any, Callable, Dict, Optional
import os
from easydict import EasyDict

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


#NOTE(lihe): fix the shap-e ddpm sampling bug 
def uncond_guide_model(
    model: Callable[..., torch.Tensor], scale: float
) -> Callable[..., torch.Tensor]:
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2] # [1, C]
        C = half.shape[1]
        # print("====C====", C)
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs) # [2, C] -> [2, 2C]

        # eps, rest = model_out[:, :3], model_out[:, 3:]
        eps, rest = model_out[:, :C], model_out[:, C:] # [2, C], [2, C]
        cond_eps, uncond_eps = torch.chunk(eps, 2, dim=0) # [2, C] -> [1, C], [1, C]
        half_eps = uncond_eps + scale * (cond_eps - uncond_eps) # [1, C]
        eps = torch.cat([half_eps, half_eps], dim=0) # [2, C]
        return torch.cat([eps, rest], dim=1) # [2, 2C]

    return model_fn

#NOTE(lihe): dpm solver sampling 
def uncond_guide_model_x0(
    model: Callable[..., torch.Tensor], scale: float
) -> Callable[..., torch.Tensor]:
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2] # [1, C]
        C = half.shape[1]
        # print("====C====", C)
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs) # [2, C] -> [2, 2C]

        # eps, rest = model_out[:, :3], model_out[:, 3:]
        eps, rest = model_out[:, :C], model_out[:, C:] # [2, C], [2, C]
        cond_eps, uncond_eps = torch.chunk(eps, 2, dim=0) # [2, C] -> [1, C], [1, C]
        half_eps = uncond_eps + scale * (cond_eps - uncond_eps) # [1, C]
        # eps = torch.cat([half_eps, half_eps], dim=0) # [2, C]
        # return torch.cat([eps, rest], dim=1) # [2, 2C]
        return half_eps # [2, C]

    return model_fn

@dataclass
class ControlNetOutput(BaseOutput):
    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor

@dataclass
class BidiffOutput(BaseOutput):
    model_pred: torch.Tensor
    loss_trans: torch.Tensor
    losses: dict
    noisy_latents_3d_prev: Optional[torch.Tensor] = None
    neus_pred_x0: Optional[torch.Tensor] = None
    noisy_latents_3d_prev = None
    neus_pred_x0 = None
    pred_clean_sdf=pred_clean_sdf = None


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 96, 256), # 
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class BidiffModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: int = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        # conv_in_kernel: int = 3,
        # conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads=64,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
        global_pool_conditions: bool = False,
        regress_rgb=False,
        foundation_model='if',
        learn_bg_color=False,
        pos_enc=False,
        debug_sd_feat=False,
        blend_x0=False,
        extra_view_num=0,
        disable_in_color_loss=False,
        abandon_sdf_x0=False,
        debug_regress=False,
        use_resnetfc=False,
        use_3d_prior=False,
        use_controlnet_cond_embedding=False,
        device=None,
        control_3d=False,
        dpm_solver_3d=False,
        model_type='text300M',
        direct_use_3d=False,
        lazy_3d=False,
        view_attn_id=(),
        num_views=8,
        use_view_embed=False,
        lazy_t=None,
        new_sdf_arc=False,
        sdf_gen=False,
        voxel_cond=False,
        use_refer_img=False,
        refer_view_id=5,
        use_featurenet_view_embed=False,
        use_view_pos_embed=False,
        pos_embed_dim=4,
        iso_surface=None,
        occ_diff=False,
        input_res=64,
        render_res=512,
    ):
        super().__init__()

        # Check inputs
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )
        
        # input
        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # # time
        # time_embed_dim = block_out_channels[0] * 4

        # self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        # timestep_input_dim = block_out_channels[0]

        # self.time_embedding = TimestepEmbedding(
        #     timestep_input_dim,
        #     time_embed_dim,
        #     act_fn=act_fn,
        # )
        # time
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )

        self.view_embed = nn.Parameter(torch.randn(num_views, time_embed_dim)) if use_view_embed else None
        if use_view_pos_embed:
            self.view_pos_embedding = TimestepEmbedding(
                in_channels=pos_embed_dim,
                time_embed_dim=time_embed_dim,
                act_fn=act_fn,
                # post_act_fn=timestep_post_act,
            )
            self.view_embed = None
        else:
            self.view_pos_embedding = None

        self.use_refer_img = use_refer_img
        self.refer_view_id = refer_view_id

        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
            logger.info("encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined.")

        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
            )

        if encoder_hid_dim_type == "text_proj":
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        elif encoder_hid_dim_type == "text_image_proj":
            # image_embed_dim DOESN'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image_proj"` (Kadinsky 2.1)`
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
            )

        elif encoder_hid_dim_type is not None:
            raise ValueError(
                f"encoder_hid_dim_type: {encoder_hid_dim_type} must be None, 'text_proj' or 'text_image_proj'."
            )
        else:
            self.encoder_hid_proj = None

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif class_embed_type == "simple_projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'simple_projection' requires `projection_class_embeddings_input_dim` be set"
                )
            self.class_embedding = nn.Linear(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None
        
        if addition_embed_type == "text":
            if encoder_hid_dim is not None:
                text_time_embedding_from_dim = encoder_hid_dim
            else:
                text_time_embedding_from_dim = cross_attention_dim

            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
            )
        elif addition_embed_type == "text_image":
            # text_embed_dim and image_embed_dim DON'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image"` (Kadinsky 2.1)`
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
            )
        elif addition_embed_type is not None:
            raise ValueError(f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'.")
    
        if time_embedding_act_fn is not None:
            raise NotImplementedError

        # control net conditioning embedding
        # self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
        #     conditioning_embedding_channels=block_out_channels[0],
        #     block_out_channels=conditioning_embedding_out_channels,
        # )
        if use_controlnet_cond_embedding:
            self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
                conditioning_channels=in_channels,
                conditioning_embedding_channels=block_out_channels[0],
            )
        else:
            self.controlnet_cond_embedding = nn.Conv2d(
                in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
            )
            # NOTE(lihe): use zero conv to replace controlnet cond embedding
            self.controlnet_cond_embedding = zero_module(self.controlnet_cond_embedding)

        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            if mid_block_only_cross_attention is None:
                mid_block_only_cross_attention = only_cross_attention

            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)
        
        if class_embeddings_concat:
            # The time embeddings are concatenated with the class embeddings. The dimension of the
            # time embeddings passed to the down, middle, and up blocks is twice the dimension of the
            # regular time embeddings
            raise NotImplementedError
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]

        controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            use_view_attn = i in view_attn_id
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                use_view_attn=use_view_attn,
                num_views=num_views,
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block[i]):
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

            if not is_final_block:
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

        # mid
        mid_block_channel = block_out_channels[-1]

        controlnet_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block

        # self.mid_block = UNetMidBlock2DCrossAttn(
        #     in_channels=mid_block_channel,
        #     temb_channels=time_embed_dim,
        #     resnet_eps=norm_eps,
        #     resnet_act_fn=act_fn,
        #     output_scale_factor=mid_block_scale_factor,
        #     resnet_time_scale_shift=resnet_time_scale_shift,
        #     cross_attention_dim=cross_attention_dim,
        #     attn_num_head_channels=attention_head_dim[-1],
        #     resnet_groups=norm_num_groups,
        #     use_linear_projection=use_linear_projection,
        #     upcast_attention=upcast_attention,
        # )
        # mid
        if mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = UNetMidBlock2DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim[-1],
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                use_view_attn=len(view_attn_id) > 0 ,
                num_views=num_views,
            )
        elif mid_block_type == "UNetMidBlock2DSimpleCrossAttn":
            self.mid_block = UNetMidBlock2DSimpleCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                cross_attention_dim=cross_attention_dim[-1],
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                skip_time_act=resnet_skip_time_act,
                only_cross_attention=mid_block_only_cross_attention,
                cross_attention_norm=cross_attention_norm,
                use_view_attn=len(view_attn_id) > 0 ,
                num_views=num_views,
            )
        elif mid_block_type is None:
            self.mid_block = None
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        if isinstance(device, str):
            device = torch.device(device)
        # MvConverter
        # we can call self._internal_dict to get all configs now
        self.lazy_3d = lazy_3d
        # DeepFolyd
        if foundation_model == 'if':
            assert iso_surface in ['dmtet', 'flexicubes'], f"we only support dmtet and flexicubes but got {iso_surface}!"
            from .denoiser3d_rast import Denoiser3DV2Rast as Denoiser3DRast
            denoiser3d_model = Denoiser3DRast
            self.denoiser3d = denoiser3d_model(regress_rgb=regress_rgb,
                                                foundation_model=foundation_model, learn_bg_color=learn_bg_color,
                                                pos_enc=pos_enc, debug_sd_feat=debug_sd_feat, blend_x0=blend_x0,
                                                extra_view_num=extra_view_num, disable_in_color_loss=disable_in_color_loss,
                                                abandon_sdf_x0=abandon_sdf_x0,
                                                debug_regress=debug_regress, use_resnetfc=use_resnetfc,
                                                use_3d_prior=use_3d_prior,
                                                device=device,
                                                model_type=model_type,
                                                direct_use_3d=direct_use_3d,
                                                lazy_3d=lazy_3d,
                                                lazy_t=lazy_t,
                                                new_sdf_arc=new_sdf_arc,
                                                sdf_gen=sdf_gen,
                                                voxel_cond=voxel_cond,
                                                use_featurenet_view_embed=use_featurenet_view_embed,
                                                input_res=input_res,
                                                render_res=render_res,
                                            )
        else:
            raise NotImplementedError
        
        self.view_attn_id = view_attn_id

        self.use_3d_prior = use_3d_prior
        self.control_3d = control_3d
        self.dpm_solver_3d = dpm_solver_3d
        self.model_type = model_type
        self.model_3d = None
        if self.use_3d_prior:
            from diffusers.models.shap_e.shap_e.models.download import load_config, load_model, load_model_adapter
            if not self.control_3d:
                self.model_3d = load_model(model_type, device=device)
                self.model_3d.requires_grad_(False) # disable gradient updating
            else:
                # TODO(lihe): figure out whether to use train or eval mode
                assert model_type == 'text300M', "currently we dont support tuning image300M"
                self.model_3d = load_model_adapter(model_type, device=device, 
                                                   custom_config='/home/lihe/workspace/mvdiff_diffusers/examples/lift3d/configs/text_cond_config_adapter.yaml')
                # also tune 3d model
                for name, param in self.model_3d.named_parameters():
                    if "adapter" not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                        param.data = param.data.float()

                for name, param in self.model_3d.wrapped.backbone.resblocks.named_parameters():
                    if "gate" in name or "adapter" in name:
                        param.data = param.data.float()
                        param.requires_grad = True
            self.model_kwargs_3d = None
        
        # self.skip_denoise = lazy_t is not None # NOTE(lihe): hack here
        self.skip_denoise = False # NOTE(lihe): hack here
        self.sdf_gen = sdf_gen
        self.input_res = input_res
        self.render_res = render_res


    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        model_cfg=None,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
        load_weights_from_unet: bool = True,
        device=None,
    ):
        r"""
        Instantiate Controlnet class from UNet2DConditionModel.

        Parameters:
            unet (`UNet2DConditionModel`):
                UNet model which weights are copied to the ControlNet. Note that all configuration options are also
                copied where applicable.
        """
        controlnet = cls(
            in_channels=unet.config.in_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            mid_block_type=unet.config.mid_block_type,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            encoder_hid_dim=unet.config.encoder_hid_dim,
            encoder_hid_dim_type=unet.config.encoder_hid_dim_type,
            attention_head_dim=unet.config.attention_head_dim,
            dual_cross_attention=unet.config.dual_cross_attention,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            addition_embed_type=unet.config.addition_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            resnet_skip_time_act=unet.config.resnet_skip_time_act,
            resnet_out_scale_factor=unet.config.resnet_out_scale_factor,
            time_embedding_type=unet.config.time_embedding_type,
            time_embedding_dim=unet.config.time_embedding_dim,
            time_embedding_act_fn=unet.config.time_embedding_act_fn,
            timestep_post_act=unet.config.timestep_post_act,
            time_cond_proj_dim=unet.config.time_cond_proj_dim,
            # conv_in_kernel=unet.config.conv_in_kernel,
            # conv_out_kernel=unet.config.conv_out_kernel,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            class_embeddings_concat=unet.config.class_embeddings_concat,
            mid_block_only_cross_attention=unet.config.mid_block_only_cross_attention,
            cross_attention_norm=unet.config.cross_attention_norm,
            addition_embed_type_num_heads=unet.config.addition_embed_type_num_heads,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            regress_rgb=model_cfg.regress_rgb,
            foundation_model=model_cfg.foundation_model,
            learn_bg_color=model_cfg.learn_bg_color,
            pos_enc=model_cfg.pos_enc,
            debug_sd_feat=model_cfg.debug_sd_feat,
            blend_x0=model_cfg.blend_x0,
            extra_view_num=model_cfg.extra_view_num,
            disable_in_color_loss=model_cfg.disable_in_color_loss,
            abandon_sdf_x0=model_cfg.abandon_sdf_x0,
            debug_regress=model_cfg.debug_regress,
            use_resnetfc=model_cfg.use_resnetfc,
            use_3d_prior=model_cfg.use_3d_prior,
            use_controlnet_cond_embedding=model_cfg.use_controlnet_cond_embedding,
            device=device, # feed acclerate.device into mvconverter
            control_3d=model_cfg.control_3d,
            dpm_solver_3d=model_cfg.dpm_solver_3d,
            model_type=model_cfg.model_type,
            direct_use_3d=model_cfg.get('direct_use_3d', False),
            lazy_3d=model_cfg.get('lazy_3d', False),
            view_attn_id=tuple(model_cfg.get('view_attn_id', ())),
            num_views=model_cfg.get('num_views', 8),
            use_view_embed=model_cfg.get('use_view_embed', False),
            lazy_t=model_cfg.get('lazy_t', None),
            new_sdf_arc=model_cfg.get('new_sdf_arc', False),
            sdf_gen=model_cfg.get('sdf_gen', False),
            voxel_cond=model_cfg.get('voxel_cond', False),
            use_refer_img=model_cfg.get('use_refer_img', False),
            refer_view_id=model_cfg.get('refer_view_id', 5),
            use_featurenet_view_embed=model_cfg.get('use_featurenet_view_embed', False),
            use_view_pos_embed=model_cfg.get('use_view_pos_embed', False),
            pos_embed_dim=model_cfg.get('pos_embed_dim', 4),
            iso_surface=model_cfg.get('iso_surface', None),
            occ_diff=model_cfg.get('occ_diff', False),
            input_res=model_cfg.get('input_res', 64),
            render_res=model_cfg.get('render_res', 512),
        )

        # For not config parameters
        # controlnet.xxx = xxx

        if load_weights_from_unet:
            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            if not model_cfg.use_controlnet_cond_embedding: # NOTE(lihe): the weight size mismatch if we also use depth as condition
                controlnet.controlnet_cond_embedding.load_state_dict(unet.conv_in.state_dict()) # use same conv_in as for noisy latents
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

            if controlnet.class_embedding:
                controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())

            controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
            controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        return controlnet


    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Parameters:
            `processor (`dict` of `AttentionProcessor` or `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                of **all** `Attention` layers.
            In case `processor` is a dict, the key needs to define the path to the corresponding cross attention processor. This is strongly recommended when setting trainable attention processors.:

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(AttnProcessor())

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attention_slice
    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maximum amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D, ResnetDownsampleBlock2D, SimpleCrossAttnDownBlock2D)):
            module.gradient_checkpointing = value

    def forward(
        self,
        noisy_latents: torch.FloatTensor, # (b, v, c, h, w)
        timestep: Union[torch.Tensor, float, int], # (b*v,)
        encoder_hidden_states: torch.Tensor,
        unet_encoder_hidden_states: torch.Tensor,
        sample_rays: dict,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
        weight_dtype=None,
        unet=None,
        cond_guidance=True,
        cond_guidance_interval=80,
        num_inference_steps=25,
        control_guidance=False,
        control_guidance_interval=1000,
        noisy_latents_3d=None,
        prompt_3d=None,
        new_batch=False,
        dpm_solver_scheduler=None,
        neus_noisy_latents=None,
        noisy_sdf=None,
        mesh_save_path=None,
        cond_decouple=False,
        background_rgb=-1,
        noisy_latents_hr=None,
        pred_clean_sdf=None,
    ) -> Union[ControlNetOutput, Tuple]:
        # check channel order
        # channel_order = self.config.controlnet_conditioning_channel_order

        # if channel_order == "rgb":
        #     # in rgb order by default
        #     ...
        # elif channel_order == "bgr":
        #     controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        # else:
        #     raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")
        iter_step = sample_rays['step']
        guid_repeat = 2 if not cond_decouple else 3
        b, nv, _, res, _ = noisy_latents.shape
        if noisy_latents_hr is None:
            assert res == 64
            noisy_latents_hr = noisy_latents
            noisy_latents_64 = None
        else:
            noisy_latents_64 = noisy_latents
        encoder_hidden_states = encoder_hidden_states.view(b*nv, *encoder_hidden_states.shape[2:])
        encoder_hidden_states_input = unet_encoder_hidden_states.view(b*nv, *unet_encoder_hidden_states.shape[2:])

        if self.use_refer_img:
            noisy_latents[:, self.refer_view_id] = sample_rays['refer_img']
            timestep.view(b, nv)[:, self.refer_view_id] = 0

        if not self.training:
            sample_rays['latents_3d'] = noisy_latents_3d # BUG!!!! uncond + cond
        
        controlnet_cond, loss_trans, losses, pred_x0, noisy_latents_3d_from_converter, noisy_latents_3d_prev, pred_clean_sdf = self.denoiser3d(feats=noisy_latents_hr, t=timestep, query_ids=None, 
                                                                    sample_rays=sample_rays, unet=unet,
                                                                    encoder_hidden_states=encoder_hidden_states_input,
                                                                    model_3d=self.model_3d, dpm_solver_scheduler=dpm_solver_scheduler,
                                                                    noisy_sdf=noisy_sdf, mesh_save_path=mesh_save_path,
                                                                    background_rgb=background_rgb, new_batch=new_batch, feats_64=noisy_latents_64,
                                                                    pred_clean_sdf=pred_clean_sdf)
        b, nv, *sp = controlnet_cond.shape
        cond_res = sp[-1]
        
        if self.skip_denoise and not self.training:
            neus_pred_x0 = controlnet_cond.clone().view(b*nv, *sp) # (b, v, 3+1, res, res)
            print("+++++++++++ saving neus pred x0 for next step +++++++++++++")
        else:
            neus_pred_x0 = None
            # NOTE(lihe): the shape of pred_x0 is [b*v, c, h, w] !  
        
        # get rendered token embeddings
        if self.control_3d:
            assert self.use_3d_prior
            # NOTE(lihe): rescale to normal color range
            # NOTE(lihe): make sure the gradient can be backward to neus
            # rendered_embeddings = self.model_3d.wrapped.clip.model.get_image_features((controlnet_cond.view(b*nv, *controlnet_cond.shape[2:]) + 1.) / 2.) # b*nv, C
            #NOTE(lihe): detach the rendered embeddings
            debug_cond_3d = sample_rays['clean_images']
            debug_cond_3d = debug_cond_3d.view(b*nv, *debug_cond_3d.shape[2:]) # NOTE(lihe): use clean image to test control3D
            rendered_embeddings = self.model_3d.wrapped.clip.model.get_image_features(debug_cond_3d.detach()) # b*nv, C
            # rendered_embeddings = self.model_3d.wrapped.clip.model.get_image_features(((controlnet_cond.view(b*nv, *controlnet_cond.shape[2:]) + 1.) / 2.).detach()) # b*nv, C
            # rendered_embeddings = rendered_embeddings.view(b, nv, -1)
            rendered_embeddings = rendered_embeddings.view(b, nv, -1)
        
        ################## 3d ###############
        if self.model_type == 'text300M' and noisy_latents_3d_prev is None:
            # NOTE(lihe): add noisy_latents_3d
            if not self.training and self.use_3d_prior:
                # TODO(lihe): check the uncond and conditional guidance !! batch size !!
                assert noisy_latents_3d is not None
                assert prompt_3d is not None
                guidance_scale_3d = 15. if self.model_type == 'text300M' else 3.
                org_batch_size = b // guid_repeat
                if new_batch:
                    # prompt_3d = [p[0].split('.')[0] + '. ' + p[0].split('.')[2] for p in prompt_3d] # delete view information
                    prompt_3d = [p[0].split('.')[2] for p in prompt_3d] # delete view information
                    print("====prompt 3d====", prompt_3d)
                    print("====timesteps====", timestep.shape)
                    
                    model_kwargs_3d = dict(texts=prompt_3d * org_batch_size)
                    if hasattr(self.model_3d, "cached_model_kwargs"):
                        model_kwargs_3d = self.model_3d.cached_model_kwargs(org_batch_size, model_kwargs_3d)
                    
                    if guidance_scale_3d != 1.0 and guidance_scale_3d != 0.0:
                        for k, val in model_kwargs_3d.copy().items():
                            model_kwargs_3d[k] = torch.cat([val, torch.zeros_like(val)], dim=0)
                    
                    # control 3d
                    if self.control_3d:
                        model_kwargs_3d['adapter'] = rendered_embeddings
                        
                    self.model_kwargs_3d = model_kwargs_3d
                else:
                    model_kwargs_3d = self.model_kwargs_3d

                internal_batch_size = org_batch_size
                if guidance_scale_3d != 1.0:
                    model = uncond_guide_model(self.model_3d, guidance_scale_3d) if dpm_solver_scheduler is None else uncond_guide_model_x0(self.model_3d, guidance_scale_3d)
                    internal_batch_size *= guid_repeat
                
                # denoise
                with torch.no_grad():
                    # latents_input = torch.cat([noisy_latents_3d]*guid_repeat)
                    if dpm_solver_scheduler is None:
                        # print("============before 3d prior denoise=============")
                        noisy_latents_3d_prev = self.denoiser3d.ddpm_3d.p_sample(
                            model,
                            noisy_latents_3d, # [b, c]
                            timestep.view(b, nv)[:, 0],
                            clip_denoised=True,
                            model_kwargs=model_kwargs_3d,
                        )['sample']
                    else:
                        scaled_model_input = dpm_solver_scheduler.scale_model_input(noisy_latents_3d, timestep[0])
                        output = model(scaled_model_input, timestep.view(b, nv)[:, 0], **model_kwargs_3d)
                        output = output.clamp(-1, 1)
                        model_input = dpm_solver_scheduler.step(
                                        output, timestep[0], noisy_latents_3d[-(noisy_latents_3d.shape[0] // guid_repeat):], return_dict=False
                                    )[0]
                        noisy_latents_3d_prev = torch.cat([model_input]*guid_repeat)
                    # print("============end 3d prior denoise=============")
                    # if timestep[0] % 100 == 0:
                    if timestep[0] < 100:
                        print("+++++++++++++++++++++++++++++++++++++++++++ Saving noisy latents 3d at timestep {} +++++++++++++++++++++++++++++++++++++++++".format(timestep[0]))
                        import numpy as np
                        np.save('debug/latents/latents_3d_{}.npy'.format(timestep[0]), noisy_latents_3d_prev.detach().cpu().numpy())

            elif self.training and self.use_3d_prior and self.control_3d:
                assert noisy_latents_3d_from_converter is not None
                assert prompt_3d is not None
                # prompt_3d = [p[0].split('.')[0] + '. ' + p[0].split('.')[2] for p in prompt_3d]
                prompt_3d = [p[0].split('.')[2] for p in prompt_3d] # delete view information
                model_kwargs_3d = dict(texts=prompt_3d * b)
                # print("======training prompt=====", prompt_3d)
                if hasattr(self.model_3d, "cached_model_kwargs"):
                        model_kwargs_3d = self.model_3d.cached_model_kwargs(b, model_kwargs_3d)

                # control 3d
                if self.control_3d:
                    model_kwargs_3d['adapter'] = rendered_embeddings
                
                # denoise
                clean_latents_3d = sample_rays['latents_3d']
                pred_latents_3d_x0 = self.denoiser3d.ddpm_3d.p_sample(
                        self.model_3d,
                        noisy_latents_3d_from_converter, # [b, c]
                        timestep.view(b, nv)[:, 0],
                        clip_denoised=True,
                        model_kwargs=model_kwargs_3d,
                    )['pred_xstart']
                denoise_loss_3d = F.mse_loss(clean_latents_3d, pred_latents_3d_x0)

                loss_trans = loss_trans + denoise_loss_3d
                losses['3d_loss'] = denoise_loss_3d.item()

                noisy_latents_3d_prev = None

            else:
                noisy_latents_3d_prev = None
            ################## 3d ###############

        if cond_guidance and not self.training:
            if timestep[0] % cond_guidance_interval < (500 // num_inference_steps) or \
                timestep[0] % cond_guidance_interval > (cond_guidance_interval - 500 // num_inference_steps):
                pred_x0 = F.interpolate(pred_x0, size=(cond_res, cond_res), mode='bilinear', align_corners=False)
                ori_b = b // guid_repeat
                controlnet_cond[-(2 * ori_b) : -ori_b] = pred_x0.view(b, nv, *sp)[-(2 * ori_b) : -ori_b]
                # encoder_hidden_states[:b * v // 2] = encoder_hidden_states[b * v // 2:] 
        
        b, nv, _, res, _ = noisy_latents.shape
        sample = noisy_latents.reshape([b*nv, -1, res, res])

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # timesteps = timesteps.unsqueeze(1).repeat(1, nv).view(-1)
        if timesteps.shape[0] != sample.shape[0]:
            timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.view_embed is not None:
            emb = emb + self.view_embed.to(dtype=emb.dtype).repeat(b, 1)
        if self.view_pos_embedding is not None:
            pos_embed = sample_rays['pos_embed'].view(b * nv, -1) # (b,v,4)
            emb = emb + self.view_pos_embedding(pos_embed.to(dtype=emb.dtype))

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb
        
        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
            emb = emb + aug_emb
        elif self.config.addition_embed_type == "text_image":
            raise NotImplementedError

        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            raise NotImplementedError

        # 2. pre-process
        sample = self.conv_in(sample)
        controlnet_cond = controlnet_cond.view(b*nv, -1, cond_res, cond_res)
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)

        sample = sample + controlnet_cond

        # 3. down
        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                cur_mask = attention_mask
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=cur_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 5. Control net blocks

        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)  # 0.1 to 1.0

            scales = scales * conditioning_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
        else:
            down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

        # if not return_dict:
        #     return (down_block_res_samples, mid_block_res_sample)
        
        # given down_block_res and mid_block_res, run original unet

        if self.use_3d_prior and not self.training: #NOTE(lihe): debug
            # if timestep[0] in range(40, 1000, 80):
            # if timestep[0] >= 850:
            # if timestep[0] >= 0:
            # if False:
            if timestep[0] <= 300:
                print("===== ONLY use Fixed SD ======")
                down_block_res_samples = [0 * sample for sample in down_block_res_samples]
                mid_block_res_sample = 0 * mid_block_res_sample
        # if control_guidance and not self.training:
            # if timestep[0] in range(40, 1000, 80):
            # if timestep[0] >= 850:
        # if not self.training:
            # down_block_res_samples = [0 * sample for sample in down_block_res_samples]
            # mid_block_res_sample = 0 * mid_block_res_sample
        
        if control_guidance_interval < 1000 and not self.training:
            if control_guidance_interval < 1500 // num_inference_steps:
                print('control_guidance_interval < 1500 // num_inference_steps, disable.')
            else:
                if timestep[0] % control_guidance_interval < (control_guidance_interval - 500 // num_inference_steps) and \
                    timestep[0] % control_guidance_interval > (control_guidance_interval - 1500 // num_inference_steps):
                    for sample in down_block_res_samples:
                        sample *= 0
                    mid_block_res_sample *= 0

        model_pred = unet(
                    noisy_latents.view(b*nv, -1, res, res) if (not self.skip_denoise) or neus_noisy_latents is None else neus_noisy_latents.view(b*nv, -1, res, res),
                    timestep,
                    encoder_hidden_states=encoder_hidden_states_input,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample
        
        return EasyDict(
            model_pred=model_pred, loss_trans=loss_trans, losses=losses, noisy_latents_3d_prev=noisy_latents_3d_prev, neus_pred_x0=neus_pred_x0,
            pred_clean_sdf=pred_clean_sdf
        )
        # return BidiffOutput(
        #     model_pred=model_pred, loss_trans=loss_trans, losses=losses, noisy_latents_3d_prev=noisy_latents_3d_prev, neus_pred_x0=neus_pred_x0,
        #     pred_clean_sdf=pred_clean_sdf
        # )
    
    def get_loss(self, output, target):
        model_pred = output.model_pred
        target = target.view(*model_pred.shape)
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss_trans = output.loss_trans
        loss = loss + loss_trans
        return loss

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
