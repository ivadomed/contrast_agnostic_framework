"""
nnUNetTrainerCHAOSAugLabDefault — AugLab GPU augmentation trainer for CHAOS.

The "synthseg" runs (with/without EM) use THIS trainer plus a SynthSeg-style
AUGLAB_PARAMS_GPU_JSON config (set by the 04_train wrapper) — there is no separate
SynthSeg trainer class, exactly as in brats2024-glioma.

MRO: nnUNetTrainerCHAOSAugLabDefault
  → nnUNetTrainerCHAOSBase   (do_split — anti-contamination guard)
  → nnUNetTrainerFast        (seed, epochs, WandB hooks)
  → nnUNetTrainerDAExtGPU    (GPU transforms, train_step)
  → nnUNetTrainer
"""
from __future__ import annotations

import importlib.resources
import os

import numpy as np
import torch

import auglab.configs as _auglab_configs
from auglab.trainers.nnUNetTrainerDAExt import nnUNetTrainerDAExtGPU
from chaos.trainers.base import nnUNetTrainerCHAOSBase

_DEFAULT_CONFIG = str(
    importlib.resources.files(_auglab_configs) / "transform_params_gpu_default01-23.json"
)


class nnUNetTrainerCHAOSAugLabDefault(nnUNetTrainerCHAOSBase, nnUNetTrainerDAExtGPU):
    """
    AugLab GPU augmentation trainer for CHAOS (Dataset060, single T1 in-phase channel).

    GPU spatial + intensity transforms applied in train_step; the AUGLAB_PARAMS_GPU_JSON
    env var selects the config (standard aug vs SynthSeg / SynthSeg+EM). Validation runs
    on clean (un-augmented) data, paper-faithful to upstream nnUNetTrainerDAExtGPU.
    """

    validation_uses_augmentation: bool = False

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        if "AUGLAB_PARAMS_GPU_JSON" not in os.environ:
            os.environ["AUGLAB_PARAMS_GPU_JSON"] = _DEFAULT_CONFIG
        super().__init__(plans, configuration, fold, dataset_json, device)

    def _log_wandb_images(self, epoch: int = 0) -> None:
        """4-panel: T1 in-phase | AugLab augmented | GT seg | Prediction.

        Sliced AXIALLY (first spatial axis) for anisotropic CHAOS volumes — consistent
        with the CHAOS baseline / v26_6_2 panels. The train panel shows the augmented
        input; the val panel mirrors validation_step (clean unless val augmentation on).
        """
        try:
            import wandb
            if wandb.run is None:
                return
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            def to_np(t):
                return t.detach().cpu().float().numpy()

            def norm_display(v):
                lo, hi = v.min(), v.max()
                if hi - lo < 1e-6:
                    return np.zeros_like(v)
                return np.clip((v - lo) / (hi - lo), 0, 1)

            def make_panel(batch, apply_aug: bool):
                data = batch["data"][0:1].to(self.device, non_blocking=True)
                tgt = batch["target"]
                target = (tgt if not isinstance(tgt, (list, tuple)) else tgt[0]).to(self.device, non_blocking=True)
                target = target[0:1]

                if apply_aug:
                    with torch.no_grad():
                        net_in, net_target = self.transforms(data, target)
                else:
                    net_in, net_target = data, target

                # AXIAL: slice the first spatial axis (D), keep (H, W) in-plane.
                mid = data.shape[2] // 2
                raw_slice = norm_display(to_np(data[0, 0])[mid])
                aug_slice = norm_display(to_np(net_in[0, 0])[mid])
                gt_slice = to_np(net_target[0, 0].float())[mid]

                self.network.eval()
                with torch.no_grad():
                    logits = self.network(net_in)
                logits = logits[0] if isinstance(logits, (list, tuple)) else logits
                pred = logits.argmax(1, keepdim=True)
                pred_slice = to_np(pred[0, 0].float())[mid]
                self.network.train()

                return raw_slice, aug_slice, gt_slice, pred_slice

            log_dict = {}
            for tag, loader, apply_aug in [
                ("train", self.dataloader_train, True),
                ("val", self.dataloader_val, self.validation_uses_augmentation),
            ]:
                raw, aug, gt, pred = make_panel(next(loader), apply_aug)
                aug_title = "AugLab aug" if apply_aug else "input (no aug)"
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(raw,  cmap="gray");                   axes[0].set_title("T1 in-phase"); axes[0].axis("off")
                axes[1].imshow(aug,  cmap="gray");                   axes[1].set_title(aug_title);     axes[1].axis("off")
                axes[2].imshow(gt,   cmap="tab10", vmin=0, vmax=6);  axes[2].set_title("GT seg");      axes[2].axis("off")
                axes[3].imshow(pred, cmap="tab10", vmin=0, vmax=6);  axes[3].set_title("Prediction");  axes[3].axis("off")
                plt.suptitle(f"synthseg/auglab {tag} fold{self.fold} ep{epoch}", fontsize=9)
                plt.tight_layout()
                log_dict[f"{tag}/panel"] = wandb.Image(fig)
                plt.close(fig)

            log_dict["epoch"] = epoch
            _safe_step = max(epoch, getattr(wandb.run, "step", epoch) or epoch)
            wandb.log(log_dict, step=_safe_step)
        except Exception as e:
            import traceback
            print(f"[WandB] CHAOS auglab image log failed: {e}\n{traceback.format_exc()}")
