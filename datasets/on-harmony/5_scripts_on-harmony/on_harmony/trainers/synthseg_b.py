"""
nnUNetTrainerOnHarmonySynthSegB — SynthSeg Mode B for ON-Harmony.

Same label maps as Mode A; mix_prior_and_random=True blends EM-estimated
intensity priors with random draws for more realistic synthetic contrasts.
"""
from __future__ import annotations

import os
from pathlib import Path

from on_harmony.trainers.synthseg_a import nnUNetTrainerOnHarmonySynthSegA


class nnUNetTrainerOnHarmonySynthSegB(nnUNetTrainerOnHarmonySynthSegA):
    """SynthSeg Mode B trainer for ON-Harmony (same labels as A, mix_prior_and_random=True)."""

    synthseg_mode: str = "B"

    def _log_wandb_images(self, epoch: int = 0) -> None:
        # Reuse Mode A logging; only the suptitle changes.
        try:
            import wandb
            if wandb.run is None:
                return
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import torch

            def _panel(batch):
                data_cpu = batch["data"][0].float()
                seg_cpu = batch["target"][0]
                from src.synthesis.v26_6_synthesis import gpu_spatial_augment
                data_gpu = data_cpu.unsqueeze(0).to(self.device)
                seg_gpu = seg_cpu.float().unsqueeze(0).to(self.device)
                data_gpu, seg_gpu = gpu_spatial_augment(data_gpu, seg_gpu, p_rotation=0.2)
                data_cpu = data_gpu.squeeze(0).cpu()
                seg_cpu = seg_gpu.squeeze(0).cpu().to(torch.int16)
                t = self._train_transforms(image=data_cpu, segmentation=seg_cpu)
                inp = t["image"].unsqueeze(0).to(self.device)
                seg_t = t["segmentation"]
                seg_t = seg_t[0] if isinstance(seg_t, (list, tuple)) else seg_t
                seg_sl = (seg_t[0] if seg_t.dim() == 4 else seg_t)[:, :, seg_t.shape[-1] // 2].float().numpy()
                self.network.eval()
                with torch.no_grad():
                    logits = self.network(inp)
                logits = logits[0] if isinstance(logits, (list, tuple)) else logits
                pred = logits.argmax(1, keepdim=True)
                self.network.train()
                mid = inp.shape[-1] // 2
                return inp[0, 0, :, :, mid].cpu().float().numpy(), seg_sl, pred[0, 0, :, :, mid].cpu().float().numpy()

            tr_batch = next(self.dataloader_train)
            val_batch = next(self.dataloader_val)

            log_dict = {}
            for tag, batch in [("train", tr_batch), ("val", val_batch)]:
                img_sl, seg_sl, pred_sl = _panel(batch)
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img_sl,  cmap="gray");                  axes[0].set_title("Input");  axes[0].axis("off")
                axes[1].imshow(seg_sl,  cmap="tab10", vmin=0, vmax=6); axes[1].set_title("GT seg"); axes[1].axis("off")
                axes[2].imshow(pred_sl, cmap="tab10", vmin=0, vmax=6); axes[2].set_title("Pred");   axes[2].axis("off")
                plt.suptitle(f"synthseg_B {tag} fold{self.fold} ep{epoch}", fontsize=9)
                plt.tight_layout()
                log_dict[f"{tag}/panel"] = wandb.Image(fig)
                plt.close(fig)

            wandb.log(log_dict, step=epoch)
        except Exception:
            pass
