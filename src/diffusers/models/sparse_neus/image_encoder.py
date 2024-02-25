import os
import json
from transformers import ViTImageProcessor, ViTModel
from torch import nn
import torch
import time

class ModLN(nn.Module):
    """
    Modulation with adaLN.
    
    References:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L101
    """
    def __init__(self, inner_dim: int, mod_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_dim, inner_dim * 2),
        )

    @staticmethod
    def modulate(x, shift, scale):
        # x: [N, L, D]
        # shift, scale: [N, D]
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, cond):
        shift, scale = self.mlp(cond).chunk(2, dim=-1)  # [N, D]
        return self.modulate(self.norm(x), shift, scale)  # [N, L, D]


class ModLayer(nn.Module):
    """
    Modulation with adaLN.
    
    References:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L101
    """
    def __init__(self, inner_dim: int, mod_dim: int, eps: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_dim, inner_dim * 2),
        )

    @staticmethod
    def modulate(x, shift, scale):
        # x: [N, L, D]
        # shift, scale: [N, D]
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, cond):
        shift, scale = self.mlp(cond).chunk(2, dim=-1)  # [N, D]
        return self.modulate(x, shift, scale)  # [N, L, D]

class ConditionModulationBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition and another modulation vector applied to sub-blocks.
    """
    # use attention from torch.nn.MultiHeadAttention
    # Block contains a cross-attention layer, a self-attention layer, and a MLP
    def __init__(self, inner_dim: int, cond_dim: int, mod_dim: int, num_heads: int, eps: float,
                 attn_drop: float = 0., attn_bias: bool = True,
                 mlp_ratio: float = 4., mlp_drop: float = 0.):
        super().__init__()
        # self.norm1 = ModLN(inner_dim, mod_dim, eps)
        # self.norm2 = ModLN(inner_dim, mod_dim, eps)
        # self.norm3 = ModLN(inner_dim, mod_dim, eps)
        self.norm1 = nn.LayerNorm(inner_dim, eps=eps)
        self.norm2 = nn.LayerNorm(inner_dim, eps=eps)
        self.norm3 = nn.LayerNorm(inner_dim, eps=eps)
        self.mod_cond = ModLayer(cond_dim, mod_dim, eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x, cond, mod):
        # x: (b, l, c_x)
        # cond: [b, nv, l_cond, c_cond]
        # mod: [b, nv, c_mod]
        b, nv, l_cond, c_cond = cond.shape
        c_mod = mod.shape[-1]
        cond = self.mod_cond(cond.view(b*nv, l_cond, c_cond), mod.view(b*nv, c_mod)).view(b, nv*l_cond, c_cond)
        x = x + self.cross_attn(self.norm1(x), cond, cond)[0]
        before_sa = self.norm2(x)
        x = x + self.self_attn(before_sa, before_sa, before_sa)[0]
        x = x + self.mlp(self.norm3(x))
        return x

class LiftTransformer(nn.Module):
    """
    Transformer with condition and modulation that generates a triplane representation.
    
    Reference:
    Timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L486
    """
    def __init__(self, inner_dim: int, image_feat_dim: int, camera_embed_dim: int,
                 trans_feat_low_res: int, trans_feat_high_res: int, trans_feat_dim: int,
                 num_layers: int, num_heads: int,
                 eps: float = 1e-6, lift_mode: str = 'triplane', num_views: int = None,):
        super().__init__()

        # attributes
        self.trans_feat_low_res = trans_feat_low_res
        self.trans_feat_high_res = trans_feat_high_res
        self.trans_feat_dim = trans_feat_dim
        self.lift_mode = lift_mode
        self.num_views = num_views
        if num_views is not None:
            self.view_embed = nn.Parameter(torch.randn(1, num_views, camera_embed_dim))

        # modules
        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        if lift_mode == 'triplane':
            self.pos_embed = nn.Parameter(torch.randn(1, 3*trans_feat_low_res**2, inner_dim) * (1. / inner_dim) ** 0.5)
            self.deconv = nn.ConvTranspose2d(inner_dim, trans_feat_dim, kernel_size=2, stride=2, padding=0)
        elif lift_mode == 'volume':
            self.pos_embed = nn.Parameter(torch.randn(1, trans_feat_low_res**3, inner_dim) * (1. / inner_dim) ** 0.5)
            self.deconv = nn.ConvTranspose3d(inner_dim, trans_feat_dim, kernel_size=2, stride=2, padding=0)
        else:
            raise NotImplementedError
        self.layers = nn.ModuleList([
            ConditionModulationBlock(
                inner_dim=inner_dim, cond_dim=image_feat_dim, mod_dim=camera_embed_dim, num_heads=num_heads, eps=eps)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(inner_dim, eps=eps)

    def forward(self, image_feats, camera_embeddings=None):
        # image_feats: [b, nv, l, c]
        # camera_embeddings: [b, nv, c_embed]

        b, nv, l, c = image_feats.shape
        res = self.trans_feat_low_res

        if self.num_views is not None:
            assert camera_embeddings is None
            camera_embeddings = self.view_embed.repeat(b, 1, 1)
        assert image_feats.shape[1] == camera_embeddings.shape[1], \
            f"Num views: {image_feats.shape[1]} vs {camera_embeddings.shape[0]}"

        x = self.pos_embed.repeat(b, 1, 1)  # [N, L, C]
        for layer in self.layers:
            x = layer(x, image_feats, camera_embeddings)
        x = self.norm(x)

        # separate each plane and apply deconv
        if self.lift_mode == 'triplane':
            x = x.view(b, 3, res, res, -1)
            x = torch.einsum('nihwc->inchw', x)  # [3, N, C, H, W]
            x = x.contiguous().view(3*b, -1, res, res)  # [3*N, C, H, W]
            x = self.deconv(x)  # [3*N, C', H', W']
            x = x.view(3, b, *x.shape[-3:])  # [3, N, C', H', W']
            x = torch.einsum('inchw->nichw', x)  # [N, 3, C', H', W']
            x = x.contiguous()
        elif self.lift_mode == 'volume':
            x = x.view(b, res, res, res, -1)
            x = torch.einsum('ndhwc->ncdhw', x)
            x = self.deconv(x)

        assert self.trans_feat_high_res == x.shape[-2], \
            f"Output triplane resolution does not match with expected: {x.shape[-2]} vs {self.trans_feat_high_res}"

        return x

class DinoWrapper(nn.Module):
    """
    Dino v1 wrapper using huggingface transformer implementation.
    """
    def __init__(self, model_name: str = 'facebook/dino-vitb16', freeze: bool = True):
        super().__init__()
        self.model = self._build_dino(model_name)
        if freeze:
            self._freeze()

    def forward(self, image):
        # image: [N, C, H, W] [-1, 1]
        # This resampling of positional embedding uses bicubic interpolation
        outputs = self.model(pixel_values=image, interpolate_pos_encoding=True)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

    def _freeze(self):
        print(f"======== Freezing DinoWrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False
    
    @staticmethod
    def _build_dino(model_name: str, proxy_error_retries: int = 3, proxy_error_cooldown: int = 5):
        import requests
        try:
            model = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
            # processor = ViTImageProcessor.from_pretrained(model_name)
            return model #, processor
        except requests.exceptions.ProxyError as err:
            if proxy_error_retries > 0:
                print(f"Huggingface ProxyError: Retrying in {proxy_error_cooldown} seconds...")
                import time
                time.sleep(proxy_error_cooldown)
                return DinoWrapper._build_dino(model_name, proxy_error_retries - 1, proxy_error_cooldown)
            else:
                raise err

if __name__ == '__main__':
    model = DinoWrapper().cuda()
    transformer = LiftTransformer(
        inner_dim=512, image_feat_dim=768, camera_embed_dim=32,
        trans_feat_low_res=32, trans_feat_high_res=64, trans_feat_dim=80,
        num_layers=1, num_heads=8, lift_mode='volume').cuda()
    image = torch.randn(8, 3, 64, 64).cuda()
    time1 = time.time()
    image_feats = model(image).cuda().view(1, 8, -1, 768)
    time2 = time.time()
    print(image_feats.shape)
    view_embed = torch.randn(8, 32, 1, 1).cuda()
    time3 = time.time()
    triplane = transformer(image_feats, view_embed.squeeze(-1).squeeze(-1).unsqueeze(0))
    time4 = time.time()
    print(triplane.shape, time2 - time1, time4 - time3)