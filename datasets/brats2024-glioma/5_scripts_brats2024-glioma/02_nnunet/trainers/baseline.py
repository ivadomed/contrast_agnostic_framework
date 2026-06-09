"""
nnUNetTrainerBraTS2024GliomaBaseline — vanilla nnUNet training on BraTS 2024 Glioma.

Uses nnUNet's default data loader and all built-in augmentations.
No synthesis; establishes the multi-modal T1n/T1c/T2w/FLAIR→segmentation baseline.

MRO: nnUNetTrainerBraTS2024GliomaBaseline
  → nnUNetTrainerBraTS2024GliomaBase  (do_split — anti-contamination guard)
  → nnUNetTrainerFast                  (seed, epochs, WandB hooks)
  → nnUNetTrainer
"""
from __future__ import annotations

import numpy as np
import torch

from brats2024_glioma.trainers.base import nnUNetTrainerBraTS2024GliomaBase


class nnUNetTrainerBraTS2024GliomaBaseline(nnUNetTrainerBraTS2024GliomaBase):
    """
    Baseline trainer for BraTS 2024 Glioma.

    nnUNetTrainerBraTS2024GliomaBase is vanilla nnUNet training — it adds nothing
    to the training loop beyond the anti-contamination do_split(). This class
    adds WandB image logging on top.
    """

    def _log_wandb_images(self, epoch: int = 0) -> None:
        try:
            import wandb
            if wandb.run is None:
                return
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            def to_np(t): return t.detach().cpu().float().numpy()

            def norm_brain(v2d):
                brain = v2d[v2d != 0]
                if len(brain) == 0:
                    return np.zeros_like(v2d)
                lo, hi = np.percentile(brain, 2), np.percentile(brain, 98)
                out = np.zeros_like(v2d)
                m = v2d != 0
                out[m] = np.clip((v2d[m] - lo) / max(hi - lo, 1e-6), 0, 1)
                return out

            def make_panel(batch, label: str):
                d   = batch["data"][:1].to(self.device)
                tgt = batch["target"]
                tgt = (tgt[0][:1] if isinstance(tgt, list) else tgt[:1]).to(self.device)
                self.network.eval()
                with torch.no_grad():
                    out = self.network(d)
                    out = out[0] if isinstance(out, (list, tuple)) else out
                    pr  = out.argmax(1, keepdim=True)
                self.network.train()

                mid = d.shape[-1] // 2
                vol = to_np(d[0])   # (4, D, H, W)
                channel_names = ["T1n", "T1c (Gd)", "T2w", "FLAIR"]
                n_cols = 4 + 2
                fig2, axes2 = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))
                for ch, name in enumerate(channel_names):
                    axes2[ch].imshow(norm_brain(vol[ch, :, :, mid]), cmap="gray")
                    axes2[ch].set_title(name, fontsize=8)
                    axes2[ch].axis("off")
                n_classes = out.shape[1]
                axes2[4].imshow(to_np(tgt[0, 0, :, :, mid]), cmap="tab10", vmin=0, vmax=n_classes)
                axes2[4].set_title("GT seg", fontsize=8)
                axes2[4].axis("off")
                axes2[5].imshow(to_np(pr[0, 0, :, :, mid]), cmap="tab10", vmin=0, vmax=n_classes)
                axes2[5].set_title("Pred", fontsize=8)
                axes2[5].axis("off")
                plt.suptitle(f"BraTS2024 baseline fold{self.fold} ep{epoch} [{label}]", fontsize=9)
                plt.tight_layout()
                return fig2

            val_batch   = next(self.dataloader_val)
            train_batch = next(self.dataloader_train)
            val_fig   = make_panel(val_batch,   "val")
            train_fig = make_panel(train_batch, "train")
            wandb.log(
                {
                    "val/panel":   wandb.Image(val_fig),
                    "train/panel": wandb.Image(train_fig),
                },
                step=epoch,
            )
            plt.close(val_fig)
            plt.close(train_fig)
        except Exception as e:
            self.print_to_log_file(f"[WandB] baseline image log failed: {e}")
