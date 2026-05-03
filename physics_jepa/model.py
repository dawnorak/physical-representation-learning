import copy
import torch
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial
from einops import rearrange
from collections import defaultdict

from physics_jepa.utils.model_utils import (
    ConvEncoder,
    ConvEncoderViTTiny,
    ConvPredictor,
    ConvPredictorViTTiny,
    ConvDecoder,
    MultiscaleConvEncoder,
)
from physics_jepa.vjepa import VJepaVisionTransformer


def _infer_in_chans(cfg, in_chans=None, stage_cfg=None):
    if in_chans is not None:
        return int(in_chans)

    if stage_cfg is not None and stage_cfg.get("fields", None) is not None:
        return len(stage_cfg.fields)

    if cfg.dataset.get("num_chans", None) is not None:
        return int(cfg.dataset.num_chans)

    raise ValueError("Unable to infer input channel count from config")


def _build_cnn_encoder(
    dims,
    num_res_blocks,
    num_frames,
    in_chans=2,
    physics_aware=False,
    field_aware_stem=None,
    periodic_padding=None,
    temporal_downsample_start_stage=None,
    use_global_context_token=None,
    field_group_sizes=None,
):
    if physics_aware:
        field_aware_stem = True if field_aware_stem is None else bool(field_aware_stem)
        periodic_padding = True if periodic_padding is None else bool(periodic_padding)
        if temporal_downsample_start_stage is None:
            temporal_downsample_start_stage = min(3, len(dims) - 1)
        if use_global_context_token is None:
            use_global_context_token = False
        return MultiscaleConvEncoder(
            in_chans=in_chans,
            num_res_blocks=num_res_blocks,
            dims=dims,
            num_frames=num_frames,
            field_aware_stem=field_aware_stem,
            periodic_padding=periodic_padding,
            field_group_sizes=field_group_sizes,
            temporal_downsample_start_stage=temporal_downsample_start_stage,
            use_global_context_token=use_global_context_token,
        )

    return ConvEncoder(
        in_chans=in_chans,
        num_res_blocks=num_res_blocks,
        dims=dims,
        num_frames=num_frames,
    )


def _build_vjepa_encoder(
    dims,
    num_frames,
    in_chans=2,
    img_size=224,
    patch_size=8,
    tubelet_size=None,
    embed_dim=None,
    depth=12,
    num_heads=None,
    mlp_ratio=4.0,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    use_learnable_pos_emb=False,
    use_checkpoint=False,
    uniform_power=False,
):
    resolved_embed_dim = int(embed_dim if embed_dim is not None else dims[-1])
    resolved_num_heads = int(num_heads if num_heads is not None else max(1, resolved_embed_dim // 64))
    resolved_tubelet_size = int(tubelet_size if tubelet_size is not None else max(1, num_frames // 2))

    return VJepaVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=resolved_tubelet_size,
        in_chans=in_chans,
        embed_dim=resolved_embed_dim,
        depth=depth,
        num_heads=resolved_num_heads,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        use_learnable_pos_emb=use_learnable_pos_emb,
        use_checkpoint=use_checkpoint,
        uniform_power=uniform_power,
    )


def build_encoder_from_cfg(cfg, in_chans=None, stage_cfg=None):
    resolved_in_chans = _infer_in_chans(cfg, in_chans=in_chans, stage_cfg=stage_cfg)
    encoder_arch = cfg.model.get("encoder_arch", None)

    if encoder_arch == "vjepa":
        return _build_vjepa_encoder(
            dims=cfg.model.dims,
            num_frames=cfg.dataset.num_frames,
            in_chans=resolved_in_chans,
            img_size=cfg.dataset.get("resolution", 224),
            patch_size=cfg.model.get("patch_size", 8),
            tubelet_size=cfg.model.get("tubelet_size", None),
            embed_dim=cfg.model.get("embed_dim", cfg.model.dims[-1]),
            depth=cfg.model.get("depth", 12),
            num_heads=cfg.model.get("num_heads", None),
            mlp_ratio=cfg.model.get("mlp_ratio", 4.0),
            drop_rate=cfg.model.get("drop_rate", 0.0),
            attn_drop_rate=cfg.model.get("attn_drop_rate", 0.0),
            drop_path_rate=cfg.model.get("drop_path_rate", 0.0),
            use_learnable_pos_emb=cfg.model.get("use_learnable_pos_emb", False),
            use_checkpoint=cfg.model.get("use_checkpoint", False),
            uniform_power=cfg.model.get("uniform_power", False),
        )

    if cfg.model.get("vit_equivalency", None) == "tiny":
        return ConvEncoderViTTiny(
            in_chans=resolved_in_chans,
            num_res_blocks=cfg.model.num_res_blocks,
            dims=cfg.model.dims,
        )

    return _build_cnn_encoder(
        dims=cfg.model.dims,
        num_res_blocks=cfg.model.num_res_blocks,
        num_frames=cfg.dataset.num_frames,
        in_chans=resolved_in_chans,
        physics_aware=cfg.model.get("physics_aware", False),
        field_aware_stem=cfg.model.get("field_aware_stem", None),
        periodic_padding=cfg.model.get("periodic_padding", None),
        temporal_downsample_start_stage=cfg.model.get("temporal_downsample_start_stage", None),
        use_global_context_token=cfg.model.get("use_global_context_token", None),
        field_group_sizes=cfg.model.get("field_group_sizes", None),
    )


def get_model_and_loss_cnn(
    dims,
    num_res_blocks,
    num_frames,
    in_chans=2,
    sim_coeff=25,
    std_coeff=25,
    cov_coeff=1,
    physics_aware=False,
    field_aware_stem=None,
    periodic_padding=None,
    temporal_downsample_start_stage=None,
    use_global_context_token=None,
    field_group_sizes=None,
    vit_equivalency=None,
    encoder_arch=None,
    img_size=None,
    patch_size=None,
    tubelet_size=None,
    embed_dim=None,
    depth=None,
    num_heads=None,
    mlp_ratio=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    use_learnable_pos_emb=False,
    use_checkpoint=False,
    uniform_power=False,
):
    if encoder_arch == "vjepa":
        encoder = _build_vjepa_encoder(
            dims=dims,
            num_frames=num_frames,
            in_chans=in_chans,
            img_size=img_size if img_size is not None else 224,
            patch_size=patch_size if patch_size is not None else 8,
            tubelet_size=tubelet_size,
            embed_dim=embed_dim,
            depth=depth if depth is not None else 12,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio if mlp_ratio is not None else 4.0,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            use_learnable_pos_emb=use_learnable_pos_emb,
            use_checkpoint=use_checkpoint,
            uniform_power=uniform_power,
        )
        loss = partial(vicreg_loss_3d,
                    sim_coeff=sim_coeff,
                    std_coeff=std_coeff,
                    cov_coeff=cov_coeff,
                    n_chunks=5)
        predictor = ConvPredictorViTTiny(dims=[encoder.dims[-1], encoder.dims[-1]])
        return encoder, predictor, loss

    if vit_equivalency == "tiny":
        encoder = ConvEncoderViTTiny(
            in_chans=in_chans,
            num_res_blocks=num_res_blocks,
            dims=dims,
        )
    else:
        encoder = _build_cnn_encoder(
            dims=dims,
            num_res_blocks=num_res_blocks,
            num_frames=num_frames,
            in_chans=in_chans,
            physics_aware=physics_aware,
            field_aware_stem=field_aware_stem,
            periodic_padding=periodic_padding,
            temporal_downsample_start_stage=temporal_downsample_start_stage,
            use_global_context_token=use_global_context_token,
            field_group_sizes=field_group_sizes,
        )
    loss = partial(vicreg_loss_3d,
                sim_coeff=sim_coeff,
                std_coeff=std_coeff,
                cov_coeff=cov_coeff,
                n_chunks=5)
    if vit_equivalency == "tiny":
        predictor = ConvPredictorViTTiny(dims=list(reversed(encoder.dims))[:2])
    else:
        predictor = ConvPredictor(dims=list(reversed(encoder.dims))[:2])
    
    return encoder, predictor, loss

def vicreg_loss_3d(
    x, y, sim_coeff, std_coeff, cov_coeff, n_chunks=10,
    num_groups=1,
    fp32_stats=False,
    zscore_for_cov=False,
    adaptive_cov_scale=False,
    mask=None,
):
    """
    x,y: (B, C, T, H, W)
    """

    # Optionally restrict the loss to masked tokens only.
    if mask is not None:
        if mask.dim() == 5:
            mask = mask.squeeze(1)
        mask = rearrange(mask, 'b t h w -> (b t h w)')
        mask = mask.to(torch.bool)
        x = rearrange(x, 'b c t h w -> (b t h w) c')[mask]
        y = rearrange(y, 'b c t h w -> (b t h w) c')[mask]
    else:
        # Flatten to (N, C) where N = B*T*H*W
        x = rearrange(x, 'b c t h w -> (b t h w) c')
        y = rearrange(y, 'b c t h w -> (b t h w) c')
    if x.numel() == 0 or y.numel() == 0:
        raise ValueError("Masked vicreg_loss_3d received an empty token set")

    N = x.shape[0]

    # Shuffle rows to decorrelate neighborhoods, then chunk
    shuffle_idx = torch.randperm(N, device=x.device)
    x_shuffled = x[shuffle_idx]
    y_shuffled = y[shuffle_idx]

    # Ensure chunks are valid (keep chunks >=1 and not smaller than ~C_group*8)
    n_chunks = max(1, int(n_chunks))
    x_chunks = x_shuffled.chunk(n_chunks, dim=0)
    y_chunks = y_shuffled.chunk(n_chunks, dim=0)

    # Return mean over chunks
    losses = defaultdict(list)
    for x_chunk, y_chunk in zip(x_chunks, y_chunks):
        out = vicreg_loss(
            x_chunk, y_chunk, sim_coeff, std_coeff, cov_coeff,
            num_groups=num_groups, fp32_stats=fp32_stats,
            zscore_for_cov=zscore_for_cov, adaptive_cov_scale=adaptive_cov_scale
        )
        (loss, repr_loss, std_loss, cov_loss,
         std_loss_x, std_loss_y, cov_loss_x, cov_loss_y) = out

        losses['loss'].append(loss)
        losses['repr_loss'].append(repr_loss)
        losses['std_loss'].append(std_loss)
        losses['cov_loss'].append(cov_loss)
        losses['std_loss_x'].append(std_loss_x)
        losses['std_loss_y'].append(std_loss_y)
        losses['cov_loss_x'].append(cov_loss_x)
        losses['cov_loss_y'].append(cov_loss_y)

    return {k: torch.stack(v).mean() for k, v in losses.items()}


def vicreg_loss(
    x, y, sim_coeff, std_coeff, cov_coeff,
    num_groups=8, fp32_stats=True, zscore_for_cov=False, adaptive_cov_scale=False
):
    """
    x, y: (N, C)
    Group-wise covariance penalty for stability when N << C.
    """
    N, C = x.shape
    assert C % num_groups == 0, f"C={C} must be divisible by num_groups={num_groups}"
    Cg = C // num_groups

    def off_diagonal(m):
        n, m_ = m.shape
        assert n == m_
        return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    # ---- representation loss (keep in original dtype) ----
    repr_loss = F.mse_loss(x, y)

    # For stats, optionally upcast to fp32 for stability
    xs = x.float() if fp32_stats else x
    ys = y.float() if fp32_stats else y

    # Center
    xs = xs - xs.mean(dim=0)
    ys = ys - ys.mean(dim=0)

    # ---- variance (same as your impl) ----
    std_x = torch.sqrt(xs.var(dim=0, unbiased=False) + 1e-4)
    std_y = torch.sqrt(ys.var(dim=0, unbiased=False) + 1e-4)
    std_loss_x = torch.mean(F.relu(1.0 - std_x)) / 2.0
    std_loss_y = torch.mean(F.relu(1.0 - std_y)) / 2.0
    std_loss = std_loss_x + std_loss_y

    # ---- (optional) z-score before covariance so cov ~ correlations ----
    if zscore_for_cov:
        sx = std_x.detach().clamp_min(1e-3)
        sy = std_y.detach().clamp_min(1e-3)
        xs = xs / sx
        ys = ys / sy

    # ---- group-wise covariance ----
    cov_loss_x = xs.new_tensor(0.0)
    cov_loss_y = ys.new_tensor(0.0)

    # Optional: adapt cov weight when N is small relative to Cg
    # scale ~ 1 when N >= 8*Cg, smaller otherwise
    if adaptive_cov_scale:
        scale = min(1.0, float(N) / float(8 * Cg))
    else:
        scale = 1.0

    for g in range(num_groups):
        xg = xs[:, g*Cg:(g+1)*Cg]
        yg = ys[:, g*Cg:(g+1)*Cg]

        # covariance within group (unbiased=False to match above var)
        cov_xg = (xg.T @ xg) / max(1, (N - 1))
        cov_yg = (yg.T @ yg) / max(1, (N - 1))

        cov_loss_x = cov_loss_x + off_diagonal(cov_xg).pow_(2).sum().div(Cg)
        cov_loss_y = cov_loss_y + off_diagonal(cov_yg).pow_(2).sum().div(Cg)

    cov_loss_x = cov_loss_x / num_groups
    cov_loss_y = cov_loss_y / num_groups
    cov_loss = scale * (cov_loss_x + cov_loss_y)

    total_loss = (
        sim_coeff * repr_loss
        + std_coeff * std_loss
        + cov_coeff * cov_loss
    )

    return total_loss, repr_loss, std_loss, cov_loss, std_loss_x, std_loss_y, cov_loss_x, cov_loss_y

# randall's method: match distribution of embeddings to isotropic Gaussian
class BCS(torch.nn.Module):
    def __init__(self, num_slices=1024):
        super().__init__()
        self.num_slices = num_slices
        self.step = 0

    @staticmethod
    def epps_pulley(x):
        def all_reduce(x, op):
            if dist.is_available() and dist.is_initialized():
                op = dist.nn.ReduceOp.__dict__[op]
                dist.nn.all_reduce(x, op=op)
                return x
            else:
                return x

        # integration points
        t = torch.linspace(-4, 4, 17, device=x.device)
        # theoretical CF for N(0, 1)
        exp_f = torch.exp(-0.5 * t**2)
        # ECF
        x_t = x.unsqueeze(2) * t  # (N, M, T)
        ecf = (1j * x_t).exp().mean(0)
        ecf = all_reduce(ecf, op="AVG")
        # weighted L2 distance
        err = exp_f * (ecf - exp_f).abs() ** 2
        T = torch.trapz(err, t, dim=1)
        return T

    def forward(self, x, y):
        views = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)
        with torch.no_grad():
            dev = views.device
            g = torch.Generator(device=dev)
            g.manual_seed(self.step)
            proj_shape = (views.size(2), self.num_slices)
            A = torch.randn(proj_shape, device=dev, generator=g)
            A /= A.norm(p=2, dim=0)
        views_A = views @ A
        self.step += 1
        return sum(self.epps_pulley(v).mean() for v in views_A) / len(views)

def vicreg_loss_bcs(x, y, sim_coeff, bcs_coeff, num_slices=1024):
    bcs = BCS(num_slices=num_slices)

    # Flatten to (N, C) where N = B*T*H*W
    x = rearrange(x, 'b c t h w -> b (t h w c)')
    y = rearrange(y, 'b c t h w -> b (t h w c)')

    sim_loss = F.mse_loss(x, y)
    bcs_loss = bcs(x, y)

    loss_dict = {
        'loss': sim_coeff * sim_loss + bcs_coeff * bcs_loss,
        'sim_loss': sim_loss,
        'bcs_loss': bcs_loss,
    }
    return loss_dict

def get_decoder(dims):
    return ConvDecoder(dims=dims)

def get_autoencoder(dims, in_chans=2):
    encoder = ConvEncoder(dims=dims, in_chans=in_chans)
    decoder = ConvDecoder(dims=list(reversed(dims)), out_chans=in_chans)
    return encoder, decoder
