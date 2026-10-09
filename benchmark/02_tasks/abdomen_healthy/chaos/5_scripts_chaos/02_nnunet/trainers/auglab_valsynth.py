"""
nnUNetTrainerCHAOSAugLabValSynth — AugLab trainer that applies SYNTHESIS in validation.

Unlike validation_uses_augmentation (which only changed the WandB viz panel — actual
validation ran on clean data), this trainer makes validation genuinely synthesise each
sample, so the validation metrics reflect performance on synthesised images.

Two configs, both set by the 04_train wrapper:
  AUGLAB_PARAMS_GPU_JSON      — TRAIN pipeline (full augmentation, synth at train prob)
  AUGLAB_VAL_PARAMS_GPU_JSON  — VAL pipeline: SYNTH-ONLY (only the synth transform on,
                                at the desired val probability; all other augs off)

Validation mirrors the pure-V26_6_2 val step: synth-only, then Dice on the synth image.
Used by both auglabAug_v26_6_2_train050_val100 and synthseg_EM_train100_val100 — the
synth transform and probabilities differ only by which config the wrapper points at.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch import autocast

from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform

from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

from auglab.transforms.gpu.transforms import AugTransformsGPU
from auglab.trainers.utils import DownsampleSegForDSTransformCustom

from chaos.trainers.auglab_default import nnUNetTrainerCHAOSAugLabDefault


class nnUNetTrainerCHAOSAugLabValSynth(nnUNetTrainerCHAOSAugLabDefault):
    """AugLab for CHAOS with real synth-only validation (val metrics on synth images)."""

    # The WandB val panel below renders the synth-only val transform, so flag it on.
    validation_uses_augmentation: bool = True

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        val_json = os.environ.get("AUGLAB_VAL_PARAMS_GPU_JSON")
        if not val_json:
            raise RuntimeError(
                "AUGLAB_VAL_PARAMS_GPU_JSON must be set (synth-only val config) for "
                "nnUNetTrainerCHAOSAugLabValSynth."
            )
        self.val_transforms_synth = AugTransformsGPU(json_path=val_json).to(self.device)
        print(f"Using AugLab VAL synth-only transforms from: {val_json}")

    # Val transforms run in the dataloader: keep target FULL-RES (no DS downsample);
    # synthesis + DS-downsampling happen in validation_step (mirrors train_step).
    def get_validation_transforms(self, *args, **kwargs):
        return ComposeTransforms([RemoveLabelTansform(-1, 0)])

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = (target[0] if isinstance(target, (list, tuple)) else target).to(
            self.device, non_blocking=True
        )

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            # Synth-only validation augmentation (real — affects metrics).
            data, target = self.val_transforms_synth(data, target)
            ds_scales = self._get_deep_supervision_scales()
            if ds_scales is not None:
                target = DownsampleSegForDSTransformCustom(ds_scales=ds_scales)(target)
            output = self.network(data)
            del data
            l = self.loss(output, target)

        # ── online Dice (replicates stock nnUNet validation_step) ──────────────
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float16)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = (~target[:, -1:]) if target.dtype == torch.bool else (1 - target[:, -1:])
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard, fp_hard, fn_hard = tp_hard[1:], fp_hard[1:], fn_hard[1:]
        return {"loss": l.detach().cpu().numpy(), "tp_hard": tp_hard, "fp_hard": fp_hard, "fn_hard": fn_hard}

    def _log_wandb_images(self, epoch: int = 0) -> None:
        """4-panel: input | train(full aug) or val(synth-only) | GT | Prediction.

        HONEST: the val panel applies the SAME synth-only transform validation uses,
        not the train pipeline — so it shows what validation metrics are computed on.
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

            def norm(v):
                lo, hi = v.min(), v.max()
                return np.zeros_like(v) if hi - lo < 1e-6 else np.clip((v - lo) / (hi - lo), 0, 1)

            def panel(batch, transforms):
                data = batch["data"][0:1].to(self.device, non_blocking=True)
                tgt = batch["target"]
                target = (tgt[0] if isinstance(tgt, (list, tuple)) else tgt)[0:1].to(self.device, non_blocking=True)
                with torch.no_grad():
                    net_in, net_target = (transforms(data, target) if transforms is not None else (data, target))
                mid = data.shape[2] // 2
                raw_s = norm(to_np(data[0, 0])[mid])
                aug_s = norm(to_np(net_in[0, 0])[mid])
                gt_s = to_np(net_target[0, 0].float())[mid]
                self.network.eval()
                with torch.no_grad():
                    logits = self.network(net_in)
                logits = logits[0] if isinstance(logits, (list, tuple)) else logits
                pred_s = to_np(logits.argmax(1, keepdim=True)[0, 0].float())[mid]
                self.network.train()
                return raw_s, aug_s, gt_s, pred_s

            log_dict = {}
            for tag, loader, tfm, aug_title in [
                ("train", self.dataloader_train, self.transforms, "AugLab train aug"),
                ("val", self.dataloader_val, self.val_transforms_synth, "synth-only (val)"),
            ]:
                raw, aug, gt, pred = panel(next(loader), tfm)
                fig, ax = plt.subplots(1, 4, figsize=(16, 4))
                ax[0].imshow(raw, cmap="gray");                  ax[0].set_title("T1 in-phase"); ax[0].axis("off")
                ax[1].imshow(aug, cmap="gray");                  ax[1].set_title(aug_title);     ax[1].axis("off")
                ax[2].imshow(gt, cmap="tab10", vmin=0, vmax=6);  ax[2].set_title("GT seg");      ax[2].axis("off")
                ax[3].imshow(pred, cmap="tab10", vmin=0, vmax=6);ax[3].set_title("Prediction");  ax[3].axis("off")
                plt.suptitle(f"{type(self).__name__} {tag} fold{self.fold} ep{epoch}", fontsize=9)
                plt.tight_layout()
                log_dict[f"{tag}/panel"] = wandb.Image(fig)
                plt.close(fig)
            log_dict["epoch"] = epoch
            _safe_step = max(epoch, getattr(wandb.run, "step", epoch) or epoch)
            wandb.log(log_dict, step=_safe_step)
        except Exception as e:
            import traceback
            print(f"[WandB] CHAOS auglab valsynth image log failed: {e}\n{traceback.format_exc()}")
