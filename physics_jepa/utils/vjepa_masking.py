import math
from collections.abc import Sequence
from multiprocessing import Value

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


class _MaskGenerator:
    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=1,
        max_context_frames_ratio=1.0,
        max_keep=None,
    ):
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size, crop_size)
        if not isinstance(spatial_patch_size, tuple):
            spatial_patch_size = (spatial_patch_size, spatial_patch_size)
        self.height = crop_size[0] // spatial_patch_size[0]
        self.width = crop_size[1] // spatial_patch_size[1]
        self.duration = num_frames // temporal_patch_size
        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(1, int(self.duration * max_context_frames_ratio))
        self.max_keep = max_keep
        self._itr_counter = Value("i", -1)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            return i.value

    def _sample_block_size(self, generator):
        _rand = torch.rand(1, generator=generator).item()
        min_t, max_t = self.temporal_pred_mask_scale
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = self.spatial_pred_mask_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = self.aspect_ratio
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(max(1, h), self.height)
        w = min(max(1, w), self.width)
        t = min(max(1, t), self.duration)
        return t, h, w

    def _sample_block_mask(self, block_size, generator):
        t, h, w = block_size
        top = int(torch.randint(0, self.height - h + 1, (1,), generator=generator).item())
        left = int(torch.randint(0, self.width - w + 1, (1,), generator=generator).item())
        start = int(torch.randint(0, self.duration - t + 1, (1,), generator=generator).item())

        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.bool)
        mask[start:start + t, top:top + h, left:left + w] = False

        if self.max_context_duration < self.duration:
            mask[self.max_context_duration:, :, :] = False

        return mask

    def sample(self, generator, block_size=None):
        if block_size is None:
            block_size = self._sample_block_size(generator)
        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.bool)
        for _ in range(self.npred):
            mask &= self._sample_block_mask(block_size, generator)
        return mask


class VJepaMaskCollator:
    def __init__(
        self,
        cfgs_mask,
        crop_size=(224, 224),
        num_frames=16,
        patch_size=(16, 16),
        tubelet_size=2,
    ):
        if not cfgs_mask:
            raise ValueError("cfgs_mask must contain at least one mask config")
        if not isinstance(patch_size, tuple):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.tubelet_size = int(tubelet_size)
        self.mask_generators = []
        for m in cfgs_mask:
            self.mask_generators.append(
                _MaskGenerator(
                    crop_size=crop_size,
                    num_frames=num_frames,
                    spatial_patch_size=patch_size,
                    temporal_patch_size=tubelet_size,
                    spatial_pred_mask_scale=_as_pair(m.get("spatial_scale"), "spatial_scale"),
                    temporal_pred_mask_scale=_as_pair(m.get("temporal_scale"), "temporal_scale"),
                    aspect_ratio=_as_pair(m.get("aspect_ratio", (0.75, 1.5)), "aspect_ratio"),
                    npred=int(m.get("num_blocks", 1)),
                    max_context_frames_ratio=float(m.get("max_temporal_keep", 1.0)),
                    max_keep=m.get("max_keep", None),
                )
            )

    def step(self):
        for mask_generator in self.mask_generators:
            mask_generator.step()

    def _mask_to_input_space(self, mask):
        # mask is [D, H, W] on the token grid.
        mask = mask.repeat_interleave(self.tubelet_size, dim=0)
        mask = mask.repeat_interleave(self.patch_size[0], dim=1)
        mask = mask.repeat_interleave(self.patch_size[1], dim=2)
        return mask

    def __call__(self, batch):
        batch = torch.utils.data.default_collate(batch)
        context = batch.get("context", None)
        if context is None:
            return batch

        batch_size = context.shape[0]
        device = context.device
        token_masks = []
        input_masks = []

        for sample_idx in range(batch_size):
            # Union all configured masks for a single sample so the visible set is
            # the complement of all masked blocks, similar to the official multiblock collator.
            visible = torch.ones_like(self.mask_generators[0].sample(torch.Generator()))
            for mask_generator in self.mask_generators:
                seed = mask_generator.step()
                g = torch.Generator()
                g.manual_seed(seed)
                block_size = mask_generator._sample_block_size(g)
                visible &= mask_generator.sample(g, block_size=block_size)

            # Avoid degenerate samples.
            if not visible.any():
                g = torch.Generator()
                g.manual_seed(sample_idx + 1)
                visible = self.mask_generators[0].sample(g)

            token_masks.append(visible)
            input_masks.append(self._mask_to_input_space(visible))

        token_masks = torch.stack(token_masks, dim=0).to(device=device)
        input_masks = torch.stack(input_masks, dim=0).unsqueeze(1).to(device=device)

        batch["vjepa_pred_mask"] = (~token_masks).unsqueeze(1).to(device=device, dtype=context.dtype)
        batch["vjepa_context_mask"] = input_masks.to(dtype=context.dtype)
        batch["context"] = batch["context"] * input_masks.to(dtype=context.dtype)
        return batch
