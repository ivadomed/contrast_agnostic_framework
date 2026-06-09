"""
nnUNetTrainerOnHarmonyBaseline — vanilla nnUNet training for ON-Harmony.

Uses nnUNet's default data loader with all built-in augmentations.
No synthesis; establishes the T1w→segmentation performance ceiling.
"""
from __future__ import annotations

import numpy as np
import torch

from on_harmony.trainers.base import nnUNetTrainerOnHarmonyBase


class nnUNetTrainerOnHarmonyBaseline(nnUNetTrainerOnHarmonyBase):
    """
    Baseline trainer for ON-Harmony.

    nnUNetTrainerOnHarmonyBase already IS vanilla nnUNet training — it adds
    nothing to the training loop beyond the anti-contamination do_split().
    This class only adds WandB image logging on top.

    MRO: nnUNetTrainerOnHarmonyBaseline
      → nnUNetTrainerOnHarmonyBase  (do_split)
      → nnUNetTrainerFast            (seed, epochs, WandB hooks)
      → nnUNetTrainer
    """

    def _log_wandb_images(self, epoch: int = 0) -> None:
        try:
            import wandb
            if wandb.run is None:
                return
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            batch = next(self.dataloader_val)
            data = batch["data"][:1].to(self.device)
            target = batch["target"]
            target_primary = target[0][:1] if isinstance(target, list) else target[:1]
            target_primary = target_primary.to(self.device)

            self.network.eval()
            with torch.no_grad():
                logits = self.network(data)
                logits = logits[0] if isinstance(logits, (list, tuple)) else logits
                pred = logits.argmax(1, keepdim=True)
            self.network.train()

            def to_np(t): return t.detach().cpu().float().numpy()
            def norm_brain(v2d):
                brain = v2d[v2d != 0]
                if len(brain) == 0: return np.zeros_like(v2d)
                lo, hi = np.percentile(brain, 2), np.percentile(brain, 98)
                out = np.zeros_like(v2d)
                m = v2d != 0
                out[m] = np.clip((v2d[m] - lo) / max(hi - lo, 1e-6), 0, 1)
                return out

            mid = data.shape[-1] // 2
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(norm_brain(to_np(data[0, 0])[:, :, mid]),         cmap="gray");                  axes[0].set_title("T1w patch"); axes[0].axis("off")
            axes[1].imshow(to_np(target_primary[0, 0])[:, :, mid],           cmap="tab10", vmin=0, vmax=6); axes[1].set_title("GT seg");    axes[1].axis("off")
            axes[2].imshow(to_np(pred[0, 0])[:, :, mid],                     cmap="tab10", vmin=0, vmax=6); axes[2].set_title("Pred");      axes[2].axis("off")
            plt.suptitle(f"baseline fold{self.fold} ep{epoch}", fontsize=9)
            plt.tight_layout()
            wandb.log({"val/panel": wandb.Image(fig)}, step=epoch)
            plt.close(fig)
        except Exception as e:
            print(f"[WandB] baseline image log failed: {e}")
