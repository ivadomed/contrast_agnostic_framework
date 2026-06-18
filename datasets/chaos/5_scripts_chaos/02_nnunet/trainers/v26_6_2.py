"""
nnUNetTrainerCHAOSV26_6_2 — V26_6_2 on-the-fly synthesis for CHAOS.

Thin dataset binding: it only wires CHAOS's data loading (nnUNet's default
loader + V26_6 synth-friendly transforms) and sets the axial WandB slice axis.
ALL synthesis / step / viz logic lives in the shared base
(src/nnunet/trainers/v26_6_base.py via nnUNetTrainerV26_6_2):

  * synthesis variant      → nnUNetTrainerV26_6_2._synthesize (per-label remap)
  * train/val synth prob   → train_synth_prob / val_synth_prob (default 0.9 / 1.0)
  * patch train/val steps  → _patch_train_step / _patch_validation_step
  * honest WandB panels    → base renders the exact tensor each step fed the net

To change how much is synthesised, set train_synth_prob / val_synth_prob — never
re-implement the steps here.

MRO: nnUNetTrainerCHAOSV26_6_2
  → nnUNetTrainerCHAOSBase   (do_split — anti-contamination guard)
  → nnUNetTrainerV26_6_2     (per-label synthesis hook)
  → nnUNetTrainerV26_6       (steps, viz, synth probability)
  → nnUNetTrainerFast / nnUNetTrainer
"""
from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from chaos.trainers.base import nnUNetTrainerCHAOSBase
from src.nnunet.trainers.v26_6_2_base import nnUNetTrainerV26_6_2
from src.nnunet.trainers.v26_6_base import _build_synth_training_transforms


class nnUNetTrainerCHAOSV26_6_2(nnUNetTrainerCHAOSBase, nnUNetTrainerV26_6_2):
    """V26_6_2 for CHAOS (Dataset060, single T1-DUAL in-phase channel)."""

    # Anisotropic abdominal volumes → slice the first (axial) spatial axis.
    _wandb_slice_axis: int = 0

    # ── Data loading: nnUNet default loader + V26_6 synth transforms ───────────

    def get_dataloaders(self):
        self._configure_v26_6()
        return nnUNetTrainer.get_dataloaders(self)

    def get_training_transforms(self, *args, **kwargs):
        return _build_synth_training_transforms(
            self._patch_size_cfg,
            self._rotation_for_DA,
            self._deep_supervision_scales,
            self._mirror_axes,
            self._do_dummy_2d,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            ignore_label=self.label_manager.ignore_label,
        )

    def get_validation_transforms(self, *args, **kwargs):
        from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
        from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
        return ComposeTransforms([RemoveLabelTansform(-1, 0)])

    # ── Steps: patch pipeline, all logic in the base ───────────────────────────

    def train_step(self, batch: dict) -> dict:
        return self._patch_train_step(batch)

    def validation_step(self, batch: dict) -> dict:
        return self._patch_validation_step(batch)
