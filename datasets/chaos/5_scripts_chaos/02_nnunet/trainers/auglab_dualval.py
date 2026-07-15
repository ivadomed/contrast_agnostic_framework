"""
nnUNetTrainerCHAOSAugLabDualVal — ONE training run, TWO independent "best"
checkpoints (clean/val000 and synth-only/val100).

OPT-IN ONLY. This is a separate class — nnUNetTrainerCHAOSAugLabDefault and
nnUNetTrainerCHAOSAugLabValSynth are untouched, and nothing changes for any
wrapper script unless it explicitly sets TRAINER=nnUNetTrainerCHAOSAugLabDualVal.
Use this ONLY when you deliberately want both a val000 and a val100 checkpoint
of the SAME train-time config without training it twice.

Rationale: for a given rung's train-time config (AUGLAB_PARAMS_GPU_JSON), the
training data pipeline is IDENTICAL whether you want the val000 or val100
variant — only validation-time behavior (and therefore which epoch's weights
get selected "best") differs. Training both separately duplicates the full
training cost for zero difference in what the model actually learns.

Implementation: reuses the two existing trainers' logic directly rather than
reimplementing it —
  - the primary/clean pass is nnUNetTrainerCHAOSAugLabDefault's OWN inherited
    (stock nnUNetTrainer) validation_step and on_validation_epoch_end,
    completely unmodified, so checkpoint_best.pth is exactly what a standalone
    nnUNetTrainerCHAOSAugLabDefault run would have produced;
  - the second, independent synth-only pass calls
    nnUNetTrainerCHAOSAugLabValSynth.validation_step UNBOUND on self (not
    copy-pasted), and replicates nnU-Net's own EMA formula (see
    nnunetv2/training/logging/nnunet_logger.py:92-96) for its own best-so-far
    tracker, so it is selection-equivalent to what a standalone
    nnUNetTrainerCHAOSAugLabValSynth run would have picked as ITS
    checkpoint_best.pth — not an approximation of it.
Output:
    checkpoint_best.pth         <- clean/val000 best (untouched upstream path)
    checkpoint_best_val100.pth  <- synth-only/val100 best (this trainer's addition)
    checkpoint_final.pth        <- last epoch, identical either way (one trajectory)

Requires AUGLAB_VAL_PARAMS_GPU_JSON (same synth-only VAL config
nnUNetTrainerCHAOSAugLabValSynth requires) in addition to AUGLAB_PARAMS_GPU_JSON.
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import torch

from nnunetv2.utilities.collate_outputs import collate_outputs

from auglab.transforms.gpu.transforms import AugTransformsGPU

from chaos.trainers.auglab_default import nnUNetTrainerCHAOSAugLabDefault
from chaos.trainers.auglab_valsynth import nnUNetTrainerCHAOSAugLabValSynth

# nnU-Net's own EMA smoothing constants for 'ema_fg_dice' — replicated here so
# the synth/val100 tracker is selection-equivalent to a standalone ValSynth
# run's own checkpoint_best.pth, not just numerically similar to it.
_EMA_ALPHA = 0.9
_EMA_BETA = 0.1


class nnUNetTrainerCHAOSAugLabDualVal(nnUNetTrainerCHAOSAugLabDefault):
    """Clean/val000 validation (inherited, unmodified) + a second, independent
    synth-only/val100 validation pass and best-checkpoint track, from a single
    training run. See module docstring — opt-in only."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        val_json = os.environ.get("AUGLAB_VAL_PARAMS_GPU_JSON")
        if not val_json:
            raise RuntimeError(
                "AUGLAB_VAL_PARAMS_GPU_JSON must be set (synth-only val config) for "
                "nnUNetTrainerCHAOSAugLabDualVal."
            )
        self.val_transforms_synth = AugTransformsGPU(json_path=val_json).to(self.device)
        self._ema_fg_dice_val100 = None
        self._best_ema_val100 = None
        self.print_to_log_file(
            f"[DualVal] checkpoint_best.pth = clean/val000, "
            f"checkpoint_best_val100.pth = synth-only/val100. Synth VAL config: {val_json}")

    def on_validation_epoch_end(self, val_outputs: List[dict]) -> None:
        # Primary pass: clean/val000. Completely unmodified upstream behaviour on
        # the val_outputs already computed by the standard run_training loop —
        # mean_fg_dice/ema_fg_dice/val_losses logged as always, checkpoint_best.pth
        # saved later by the base on_epoch_end exactly like a standalone
        # nnUNetTrainerCHAOSAugLabDefault run.
        super().on_validation_epoch_end(val_outputs)

        # Second pass: synth-only/val100. Reuses ValSynth's OWN validation_step
        # verbatim (unbound call) so this is selection-equivalent to what a
        # standalone nnUNetTrainerCHAOSAugLabValSynth run would compute, not a
        # reimplementation that could quietly drift from it.
        with torch.no_grad():
            synth_outputs = [
                nnUNetTrainerCHAOSAugLabValSynth.validation_step(self, next(self.dataloader_val))
                for _ in range(self.num_val_iterations_per_epoch)
            ]

        outputs_collated = collate_outputs(synth_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)
        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice_val100 = np.nanmean(global_dc_per_class)

        self._ema_fg_dice_val100 = (
            mean_fg_dice_val100 if self._ema_fg_dice_val100 is None
            else self._ema_fg_dice_val100 * _EMA_ALPHA + _EMA_BETA * mean_fg_dice_val100
        )
        self.print_to_log_file('val100 (synth) pseudo dice', np.round(mean_fg_dice_val100, decimals=4))

        if self._best_ema_val100 is None or self._ema_fg_dice_val100 > self._best_ema_val100:
            self._best_ema_val100 = self._ema_fg_dice_val100
            self.print_to_log_file(
                f"Yayy! New best EMA pseudo Dice (val100/synth): "
                f"{np.round(self._best_ema_val100, decimals=4)}")
            self.save_checkpoint(os.path.join(self.output_folder, 'checkpoint_best_val100.pth'))
