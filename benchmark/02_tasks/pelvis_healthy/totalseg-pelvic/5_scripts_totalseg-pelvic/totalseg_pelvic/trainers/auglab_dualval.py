"""
nnUNetTrainerTotalsegPelvicAugLabDualVal — ONE training run, TWO independent "best"
checkpoints (clean/val000 and synth-only/val100), with NO third/intermediate RUN_ID ever
appearing anywhere.

Direct port of autopet.trainers.auglab_dualval.nnUNetTrainerAutoPETAugLabDualVal to
totalseg-pelvic (same mechanism — see that module's docstring, itself ported from
toothfairy2 / ambl / atlas_liver_hcc / open_ms, for the full rationale). OPT-IN ONLY:
nnUNetTrainerTotalsegPelvicAugLabDefault and nnUNetTrainerTotalsegPelvicAugLabValSynth are
untouched, and nothing changes for any wrapper script unless it explicitly sets
TRAINER=nnUNetTrainerTotalsegPelvicAugLabDualVal.

Directory-level compatibility (on_train_end) — NO third-named directory ever exists: the
wrapper script's RUN_ID must contain "_val000_" exactly once (that IS the real training
directory); at on_train_end this materializes exactly ONE sibling "_val100_" RUN_ID
directory, hard-linked (not copied) from this run's checkpoint_best_val100.pth.

Requires AUGLAB_VAL_PARAMS_GPU_JSON (same synth-only VAL config
nnUNetTrainerTotalsegPelvicAugLabValSynth requires) in addition to AUGLAB_PARAMS_GPU_JSON.
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import torch

from nnunetv2.utilities.collate_outputs import collate_outputs

from auglab.transforms.gpu.transforms import AugTransformsGPU

from totalseg_pelvic.trainers.auglab_default import nnUNetTrainerTotalsegPelvicAugLabDefault
from totalseg_pelvic.trainers.auglab_valsynth import nnUNetTrainerTotalsegPelvicAugLabValSynth

_EMA_ALPHA = 0.9
_EMA_BETA = 0.1

_VAL000_MARKER = "_val000_"
_VAL100_MARKER = "_val100_"
_META_FILES_RUN_LEVEL = ("dataset.json", "dataset_fingerprint.json", "plans.json")
_META_FILES_FOLD_LEVEL = ("debug.json",)


def materialize_split_runs(output_folder: str, run_id: str, log=print) -> None:
    """Hard-link this fold's checkpoint_best_val100.pth into a sibling RUN_ID directory —
    identical logic to every other dataset's auglab_dualval.materialize_split_runs, kept
    as a separate copy per this project's convention of self-contained per-dataset
    trainer packages."""
    if run_id.count(_VAL000_MARKER) != 1:
        log(f"[DualVal] RUN_ID '{run_id}' doesn't contain '_val000_' exactly once — "
            f"skipping val100 mirror-directory materialization.")
        return

    fold_name = os.path.basename(output_folder.rstrip("/"))
    trainer_dir = os.path.basename(os.path.dirname(output_folder))
    dataset_dir_full = os.path.dirname(os.path.dirname(output_folder))
    dataset_name = os.path.basename(dataset_dir_full)
    run_dir = os.path.dirname(dataset_dir_full)
    results_base = os.path.dirname(run_dir)

    def _link(src: str, dst: str) -> None:
        if not os.path.isfile(src):
            return
        if os.path.exists(dst) or os.path.islink(dst):
            if os.path.exists(dst) and os.path.samefile(src, dst):
                return
            os.remove(dst)
        os.link(src, dst)

    mirror_run_id = run_id.replace(_VAL000_MARKER, _VAL100_MARKER, 1)
    mirror_trainer_dir = os.path.join(results_base, mirror_run_id, dataset_name, trainer_dir)
    mirror_fold_dir = os.path.join(mirror_trainer_dir, fold_name)
    os.makedirs(mirror_fold_dir, exist_ok=True)

    for meta in _META_FILES_RUN_LEVEL:
        _link(os.path.join(os.path.dirname(output_folder), meta),
              os.path.join(mirror_trainer_dir, meta))

    _link(os.path.join(output_folder, "checkpoint_best_val100.pth"),
          os.path.join(mirror_fold_dir, "checkpoint_best.pth"))
    _link(os.path.join(output_folder, "checkpoint_final.pth"),
          os.path.join(mirror_fold_dir, "checkpoint_final.pth"))

    for meta in _META_FILES_FOLD_LEVEL:
        _link(os.path.join(output_folder, meta), os.path.join(mirror_fold_dir, meta))

    log(f"[DualVal] materialized {mirror_run_id} ({fold_name}) from checkpoint_best_val100.pth "
        f"-> {mirror_fold_dir}/checkpoint_best.pth (hard link)")


class nnUNetTrainerTotalsegPelvicAugLabDualVal(nnUNetTrainerTotalsegPelvicAugLabDefault):
    """Clean/val000 validation (inherited, unmodified) + a second, independent
    synth-only/val100 validation pass and best-checkpoint track, from a single training
    run. Opt-in only. The wrapper script MUST name RUN_ID with "_val000_"."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        val_json = os.environ.get("AUGLAB_VAL_PARAMS_GPU_JSON")
        if not val_json:
            raise RuntimeError(
                "AUGLAB_VAL_PARAMS_GPU_JSON must be set (synth-only val config) for "
                "nnUNetTrainerTotalsegPelvicAugLabDualVal."
            )
        self.val_transforms_synth = AugTransformsGPU(json_path=val_json).to(self.device)
        self._ema_fg_dice_val100 = None
        self._best_ema_val100 = None
        self.print_to_log_file(
            f"[DualVal] checkpoint_best.pth = clean/val000, "
            f"checkpoint_best_val100.pth = synth-only/val100. Synth VAL config: {val_json}")

    def on_validation_epoch_end(self, val_outputs: List[dict]) -> None:
        super().on_validation_epoch_end(val_outputs)

        with torch.no_grad():
            synth_outputs = [
                nnUNetTrainerTotalsegPelvicAugLabValSynth.validation_step(self, next(self.dataloader_val))
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

    def on_train_end(self) -> None:
        super().on_train_end()
        if self.local_rank != 0:
            return
        run_id = os.environ.get("RUN_ID", "")
        if not run_id:
            self.print_to_log_file("[DualVal] RUN_ID env var not set — skipping val100 mirror-directory "
                                    "materialization.")
            return
        materialize_split_runs(self.output_folder, run_id, log=self.print_to_log_file)
