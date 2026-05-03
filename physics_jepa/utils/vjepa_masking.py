import math
from collections.abc import Sequence

import torch


def _as_pair(value, name):
    if value is None:
        raise ValueError(f"{name} must not be None")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"{name} must be a pair, got {value}")
        return float(value[0]), float(value[1])
    v = float(value)
    return v, v


def _make_generator(device, seed):
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def _sample_block_size(spec, generator, device, duration, height, width):
    _rand = torch.rand(1, generator=generator, device=device).item()
    min_t, max_t = spec["temporal_scale"]
    temporal_mask_scale = min_t + _rand * (max_t - min_t)
    t = max(1, int(duration * temporal_mask_scale))

    _rand = torch.rand(1, generator=generator, device=device).item()
    min_s, max_s = spec["spatial_scale"]
    spatial_mask_scale = min_s + _rand * (max_s - min_s)
    spatial_num_keep = int(height * width * spatial_mask_scale)

    _rand = torch.rand(1, generator=generator, device=device).item()
    min_ar, max_ar = spec["aspect_ratio"]
    aspect_ratio = min_ar + _rand * (max_ar - min_ar)

    h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
    w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
    h = min(max(1, h), height)
    w = min(max(1, w), width)
    t = min(max(1, t), duration)
    return t, h, w


def _sample_block_mask(spec, block_size, generator, device, duration, height, width):
    t, h, w = block_size
    top = int(torch.randint(0, height - h + 1, (1,), generator=generator, device=device).item())
    left = int(torch.randint(0, width - w + 1, (1,), generator=generator, device=device).item())
    start = int(torch.randint(0, duration - t + 1, (1,), generator=generator, device=device).item())

    mask = torch.ones((duration, height, width), dtype=torch.bool, device=device)
    mask[start:start + t, top:top + h, left:left + w] = False

    max_context_duration = spec["max_context_duration"]
    if max_context_duration < duration:
        mask[max_context_duration:, :, :] = False

    return mask


def build_vjepa_mask_specs(cfgs_mask):
    if not cfgs_mask:
        raise ValueError("cfgs_mask must contain at least one mask config")

    specs = []
    for m in cfgs_mask:
        spatial_scale = _as_pair(m.get("spatial_scale"), "spatial_scale")
        temporal_scale = _as_pair(m.get("temporal_scale"), "temporal_scale")
        aspect_ratio = _as_pair(m.get("aspect_ratio", (0.75, 1.5)), "aspect_ratio")
        specs.append(
            {
                "num_blocks": int(m.get("num_blocks", 1)),
                "spatial_scale": spatial_scale,
                "temporal_scale": temporal_scale,
                "aspect_ratio": aspect_ratio,
                "max_temporal_keep": float(m.get("max_temporal_keep", 1.0)),
                "max_context_duration": None,  # filled later
            }
        )
    return specs


def sample_vjepa_masks(
    batch_size,
    input_shape,
    cfgs_mask,
    patch_size,
    tubelet_size,
    device,
    seed,
):
    if not isinstance(patch_size, tuple):
        patch_size = (patch_size, patch_size)
    if len(input_shape) != 5:
        raise ValueError(f"Expected input shape [B, C, T, H, W], got {input_shape}")

    _, _, num_frames, height, width = input_shape
    token_t = num_frames // tubelet_size
    token_h = height // patch_size[0]
    token_w = width // patch_size[1]
    if token_t < 1 or token_h < 1 or token_w < 1:
        raise ValueError(
            f"Invalid token grid from shape={input_shape}, patch_size={patch_size}, tubelet_size={tubelet_size}"
        )

    specs = build_vjepa_mask_specs(cfgs_mask)
    for spec in specs:
        spec["max_context_duration"] = max(1, int(token_t * float(spec.get("max_temporal_keep", 1.0))))

    generator = _make_generator(device, seed)
    context_masks = []
    pred_masks = []

    for sample_idx in range(batch_size):
        visible = torch.ones((token_t, token_h, token_w), dtype=torch.bool, device=device)
        for spec in specs:
            block_visible = torch.ones_like(visible)
            for _ in range(spec["num_blocks"]):
                block_size = _sample_block_size(spec, generator, device, token_t, token_h, token_w)
                block_visible &= _sample_block_mask(spec, block_size, generator, device, token_t, token_h, token_w)
            visible &= block_visible

        if not visible.any():
            # Fallback to a simple visible cube to avoid empty-token batches.
            visible = torch.ones_like(visible)
            visible[0:1, 0:max(1, token_h // 2), 0:max(1, token_w // 2)] = False

        pred = (~visible).unsqueeze(0)
        ctx = visible.repeat_interleave(tubelet_size, dim=0)
        ctx = ctx.repeat_interleave(patch_size[0], dim=1)
        ctx = ctx.repeat_interleave(patch_size[1], dim=2)

        context_masks.append(ctx.unsqueeze(0))
        pred_masks.append(pred)

    context_masks = torch.stack(context_masks, dim=0).to(device=device, dtype=torch.float32)
    pred_masks = torch.stack(pred_masks, dim=0).to(device=device, dtype=torch.float32)
    return context_masks, pred_masks
