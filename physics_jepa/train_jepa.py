import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from .train import Trainer
from .utils.hydra import compose
from .utils.vjepa_masking import sample_vjepa_masks

class JepaTrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._vjepa_mask_seed = 0

    def pred_fn(self, batch, model_components, loss_fn):
        encoder, predictor = model_components[:2]
        fusion = model_components[2] if len(model_components) > 2 else None

        ctx = batch['context']
        mask = None
        if self.cfg.model.get("encoder_arch", None) == "vjepa" and self.train_cfg.get("vjepa_masking", False):
            vjepa_mask_cfgs = self.train_cfg.get("vjepa_mask_cfgs", None)
            if not vjepa_mask_cfgs:
                raise ValueError("vjepa_masking is enabled but vjepa_mask_cfgs is missing")
            ctx_mask, pred_mask = sample_vjepa_masks(
                batch_size=ctx.shape[0],
                input_shape=ctx.shape,
                cfgs_mask=vjepa_mask_cfgs,
                patch_size=self.cfg.model.get("patch_size", 16),
                tubelet_size=self.cfg.model.get("tubelet_size", None) or max(1, self.cfg.dataset.num_frames // 2),
                device=ctx.device,
                seed=self._vjepa_mask_seed,
            )
            self._vjepa_mask_seed += 1
            ctx = ctx * ctx_mask
            batch["vjepa_context_mask"] = ctx_mask
            batch["vjepa_pred_mask"] = pred_mask

        ctx_embed = encoder(ctx)
        if fusion is not None:
            temporal_stride = self.train_cfg.get("multi_scale_temporal_stride", 4)
            global_ctx = ctx[:, :, ::temporal_stride, :, :]
            global_ctx = F.interpolate(
                global_ctx,
                size=ctx.shape[2:],
                mode='nearest',
            )
            global_ctx_embed = encoder(global_ctx)
            ctx_embed = fusion(torch.cat([ctx_embed, global_ctx_embed], dim=1))

        tgt = batch["target"]
        tgt_embed = encoder(tgt)
        pred = predictor(ctx_embed)
        # Compute loss on projected embeddings
        if len(pred.shape) < 5:
            loss_dict = loss_fn(pred.unsqueeze(2), tgt_embed.unsqueeze(2))
        else:
            loss_dict = loss_fn(pred, tgt_embed)

        return pred, loss_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default=f"{Path(__file__).parent.parent}/configs/train_grayscott.yml")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--predictor_path", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = compose(args.config, args.overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.dry_run = args.dry_run
    # cfg.train.encoder_path = args.encoder_path
    # cfg.train.predictor_path = args.predictor_path
    
    cfg.model.objective = "jepa"

    print(OmegaConf.to_yaml(cfg, resolve=True))

    trainer = JepaTrainer(cfg)
    trainer.train()
