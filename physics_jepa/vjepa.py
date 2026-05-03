"""Video JEPA-style transformer encoder.

This module keeps the repo's existing channel-first feature-map contract:
`forward()` returns a tensor shaped `(B, C, T, H, W)`.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange

from .utils.model_utils import DropPath
from .utils.tensors import trunc_normal_


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate a 1D sine/cosine positional embedding from a grid."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    grid_depth: int,
    cls_token: bool = False,
    uniform_power: bool = False,
) -> np.ndarray:
    """Generate a 3D sine/cosine positional embedding."""
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(grid_h, grid_d, grid_w)

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6) * 2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)

    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _to_2tuple(value):
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        values = tuple(int(v) for v in value)
        if len(values) != 2:
            raise ValueError(f"Expected a 2-tuple, got {values}")
        return values
    v = int(value)
    return (v, v)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        bsz, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(bsz, tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim, act_layer=act_layer, drop=drop)

        if init_values and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed3D(nn.Module):
    """Patchify a video into tubelet tokens."""

    def __init__(
        self,
        img_size=224,
        patch_size=8,
        num_frames=16,
        tubelet_size=8,
        in_chans=3,
        embed_dim=384,
    ):
        super().__init__()
        self.img_size = _to_2tuple(img_size)
        self.patch_size = _to_2tuple(patch_size)
        self.num_frames = int(num_frames)
        self.tubelet_size = int(tubelet_size)
        self.num_patches = (
            (self.img_size[0] // self.patch_size[0])
            * (self.img_size[1] // self.patch_size[1])
            * (self.num_frames // self.tubelet_size)
        )
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, self.patch_size[0], self.patch_size[1]),
            stride=(self.tubelet_size, self.patch_size[0], self.patch_size[1]),
        )

    def forward(self, x):
        return self.proj(x)


class VJepaVisionTransformer(nn.Module):
    """Video transformer encoder that returns a feature map.

    The model processes video clips as non-overlapping spatiotemporal tubelets,
    applies standard transformer blocks over the flattened token sequence, and
    reshapes the final token grid back to `(B, C, T, H, W)`.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=8,
        num_frames=16,
        tubelet_size=8,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        use_learnable_pos_emb=False,
        use_checkpoint=False,
        uniform_power=False,
    ):
        super().__init__()

        self.num_features = self.embed_dim = int(embed_dim)
        self.num_frames = int(num_frames)
        self.tubelet_size = int(tubelet_size)
        self.patch_size = _to_2tuple(patch_size)
        self.img_size = _to_2tuple(img_size)
        self.use_checkpoint = bool(use_checkpoint)
        self.uniform_power = bool(uniform_power)

        self.patch_embed = PatchEmbed3D(
            img_size=self.img_size,
            patch_size=self.patch_size,
            num_frames=self.num_frames,
            tubelet_size=self.tubelet_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
            self._init_pos_embed(self.pos_embed.data)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # One norm for token outputs and one for pooled features.
        self.token_norm = norm_layer(embed_dim)
        self.pool_norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        # Keep a dims attribute for downstream code that expects it.
        self.dims = [self.embed_dim]

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_h = self.img_size[0] // self.patch_size[0]
        grid_w = self.img_size[1] // self.patch_size[1]
        grid_depth = self.num_frames // self.tubelet_size
        assert grid_h * grid_w * grid_depth == pos_embed.shape[1], "Positional embedding initialized incorrectly"

        sincos = get_3d_sincos_pos_embed(
            embed_dim,
            grid_size=grid_h,
            grid_depth=grid_depth,
            cls_token=False,
            uniform_power=self.uniform_power,
        )
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {"pos_embed"}

    def _grid_shape(self, x):
        if x.ndim == 5:
            t = x.shape[2] // self.tubelet_size
            h = x.shape[3] // self.patch_size[0]
            w = x.shape[4] // self.patch_size[1]
        elif x.ndim == 4:
            t = 1
            h = x.shape[2] // self.patch_size[0]
            w = x.shape[3] // self.patch_size[1]
        else:
            raise ValueError(f"Expected 4D or 5D input, got {tuple(x.shape)}")
        return t, h, w

    def interpolate_pos_encoding(self, x):
        """Interpolate the positional embedding to the current token grid."""
        pos_embed = self.pos_embed
        _, _, dim = pos_embed.shape

        t, h, w = self._grid_shape(x)
        base_t = self.num_frames // self.tubelet_size
        base_h = self.img_size[0] // self.patch_size[0]
        base_w = self.img_size[1] // self.patch_size[1]

        if (t, h, w) == (base_t, base_h, base_w):
            return pos_embed

        pos_embed_3d = pos_embed.reshape(1, base_t, base_h, base_w, dim).permute(0, 4, 1, 2, 3)
        pos_embed_3d = F.interpolate(
            pos_embed_3d,
            size=(t, h, w),
            mode="trilinear",
            align_corners=False,
        )
        return pos_embed_3d.permute(0, 2, 3, 4, 1).reshape(1, -1, dim)

    def forward_tokens(self, x):
        pos_embed = self.interpolate_pos_encoding(x)
        x = self.patch_embed(x)
        x = rearrange(x, "b c t h w -> b (t h w) c")
        x = x + pos_embed.to(dtype=x.dtype, device=x.device)
        x = self.pos_drop(x)

        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        return self.token_norm(x)

    def get_patch_embeddings(self, x):
        """Return the token sequence before pooling."""
        return self.forward_tokens(x)

    def forward_features(self, x):
        """Return a pooled embedding for linear heads or probes."""
        tokens = self.forward_tokens(x)
        return self.pool_norm(tokens.mean(dim=1))

    def forward(self, x):
        tokens = self.forward_tokens(x)
        t, h, w = self._grid_shape(x)
        x = rearrange(tokens, "b (t h w) c -> b c t h w", t=t, h=h, w=w)
        return x.contiguous()


def vjepa_tiny(**kwargs):
    """Convenience factory for a compact JEPA-style video transformer."""
    return VJepaVisionTransformer(
        embed_dim=kwargs.pop("embed_dim", 384),
        depth=kwargs.pop("depth", 12),
        num_heads=kwargs.pop("num_heads", 6),
        mlp_ratio=kwargs.pop("mlp_ratio", 4.0),
        **kwargs,
    )
