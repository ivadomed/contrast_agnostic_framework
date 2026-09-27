"""
nnUNetTrainerCHAOSBaseline — CHAOS MR T1-in baseline (no synthesis).

Single input channel (T1-DUAL in-phase), standard nnUNet augmentation. This is the
real-data baseline against which domain-randomization synthesis (v26_6-style) is
compared for cross-modality generalization (out-phase / T2-SPIR / CT).

Inherits split validation from nnUNetTrainerCHAOSBase and seed/epochs/WandB hooks
from nnUNetTrainerFast. Overrides _log_wandb_images to log an input / GT / prediction
panel to WandB (every NNUNET_PLOT_EVERY epochs + final epoch).
"""
from __future__ import annotations

import numpy as np
import torch

from chaos.trainers.base import nnUNetTrainerCHAOSBase


class nnUNetTrainerCHAOSBaseline(nnUNetTrainerCHAOSBase):

    def _log_wandb_images(self, epoch: int = 0) -> None:
        try:
            import wandb
            if wandb.run is None:
                return
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            def make_panel(dataloader, label: str):
                batch = next(dataloader)
                data = batch["data"][:1].to(self.device)          # (1, 1, D, H, W)
                target = batch["target"]
                target_primary = (
                    target[0][:1] if isinstance(target, list) else target[:1]
                ).to(self.device)

                self.network.eval()
                with torch.no_grad():
                    logits = self.network(data)
                    logits = logits[0] if isinstance(logits, (list, tuple)) else logits
                    pred = logits.argmax(1, keepdim=True)
                self.network.train()

                def to_np(t):
                    return t.detach().cpu().float().numpy()

                def norm_body(v2d):
                    body = v2d[v2d != 0]
                    if len(body) == 0:
                        return np.zeros_like(v2d)
                    lo, hi = np.percentile(body, 2), np.percentile(body, 98)
                    out = np.zeros_like(v2d)
                    m = v2d != 0
                    out[m] = np.clip((v2d[m] - lo) / max(hi - lo, 1e-6), 0, 1)
                    return out

                # CHAOS volumes are thin in z; take the central axial slice.
                mid = data.shape[2] // 2
                vol = to_np(data[0])           # (1, D, H, W)
                n_classes = logits.shape[1]    # 5 (bg + 4 organs)

                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                axes[0].imshow(norm_body(vol[0, mid]), cmap="gray")
                axes[0].set_title("T1 in-phase", fontsize=8); axes[0].axis("off")
                axes[1].imshow(to_np(target_primary[0, 0, mid]),
                               cmap="tab10", vmin=0, vmax=n_classes)
                axes[1].set_title("GT seg", fontsize=8); axes[1].axis("off")
                axes[2].imshow(to_np(pred[0, 0, mid]),
                               cmap="tab10", vmin=0, vmax=n_classes)
                axes[2].set_title("Pred", fontsize=8); axes[2].axis("off")
                plt.suptitle(f"CHAOS baseline fold{self.fold} ep{epoch} [{label}]", fontsize=9)
                plt.tight_layout()
                return fig

            val_fig = make_panel(self.dataloader_val, "val")
            train_fig = make_panel(self.dataloader_train, "train")
            wandb.log({"val/panel": wandb.Image(val_fig),
                       "train/panel": wandb.Image(train_fig)}, step=epoch)
            plt.close(val_fig)
            plt.close(train_fig)
        except Exception as e:
            self.print_to_log_file(f"[WandB] CHAOS baseline image log failed: {e}")
