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

Directory-level compatibility (on_train_end): the RUN_ID directory this
trainer actually writes to (e.g. ..._dualval_<TS>/) keeps all 3 raw
checkpoints for reference:
    checkpoint_best.pth         <- clean/val000 best (untouched upstream path)
    checkpoint_best_val100.pth  <- synth-only/val100 best (this trainer's addition)
    checkpoint_final.pth        <- last epoch, identical either way (one trajectory)
But nothing downstream (predict/eval/resume) should have to know that split
exists. So at on_train_end, this trainer ALSO materializes two sibling RUN_ID
directories — ..._val000_<TS>/ and ..._val100_<TS>/, same <TS>, "_dualval_"
swapped for "_val000_"/"_val100_" — each hard-linked (not copied: same
content, zero extra disk) to a STANDARD checkpoint_best.pth/checkpoint_final.pth
under the trainer's normal fold_N/ layout. Predict/eval/resume scripts can
then point at either mirror RUN_ID exactly as if two separate trainings had
happened, with no CHECKPOINT=/PREDICT_OUTPUT_SUBDIR= special-casing anywhere
else in the pipeline. See materialize_split_runs() (module-level, reusable
from a standalone script for runs already in flight when this was added).

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

_DUALVAL_MARKER = "_dualval_"
_META_FILES_RUN_LEVEL = ("dataset.json", "dataset_fingerprint.json", "plans.json")
_META_FILES_FOLD_LEVEL = ("debug.json",)


def materialize_split_runs(output_folder: str, run_id: str, log=print) -> None:
    """Hard-link this fold's DualVal outputs into two sibling RUN_ID directories
    (.../<run_id with _val000_>/... and .../<run_id with _val100_>/...), each
    laid out exactly like a standalone single-checkpoint run.

    output_folder: the trainer's self.output_folder for THIS fold, i.e.
        <RESULTS_BASE>/<run_id>/<dataset_name>/<trainer_dir>/fold_<N>
    run_id: must contain "_dualval_" exactly once (the RUN_ID this fold trained
        under) — if it doesn't, this is a no-op (not a DualVal run, or already
        a materialized mirror — never recurse into a mirror).

    Safe to call multiple times / call again after the fact (e.g. from a
    standalone post-hoc script for runs launched before this existed): all
    operations are idempotent (mkdir -p semantics, overwrite-if-exists links).
    """
    if run_id.count(_DUALVAL_MARKER) != 1:
        log(f"[DualVal] RUN_ID '{run_id}' doesn't contain '_dualval_' exactly once — "
            f"skipping mirror-directory materialization (not a DualVal run?).")
        return

    fold_name = os.path.basename(output_folder.rstrip("/"))          # "fold_N"
    trainer_dir = os.path.basename(os.path.dirname(output_folder))    # "nnUNetTrainer...__nnUNetPlans__3d_fullres"
    dataset_dir_full = os.path.dirname(os.path.dirname(output_folder))  # .../<run_id>/<dataset_name>
    dataset_name = os.path.basename(dataset_dir_full)
    run_dir = os.path.dirname(dataset_dir_full)                      # .../<run_id>
    results_base = os.path.dirname(run_dir)                          # RESULTS_BASE (nnUNet/ or auglab/ dir)

    def _link(src: str, dst: str) -> None:
        if not os.path.isfile(src):
            return
        if os.path.exists(dst) or os.path.islink(dst):
            if os.path.samefile(src, dst) if os.path.exists(dst) else False:
                return
            os.remove(dst)
        os.link(src, dst)

    for suffix, src_best_name in ((("_val000_"), "checkpoint_best.pth"),
                                   (("_val100_"), "checkpoint_best_val100.pth")):
        mirror_run_id = run_id.replace(_DUALVAL_MARKER, suffix, 1)
        mirror_trainer_dir = os.path.join(results_base, mirror_run_id, dataset_name, trainer_dir)
        mirror_fold_dir = os.path.join(mirror_trainer_dir, fold_name)
        os.makedirs(mirror_fold_dir, exist_ok=True)

        for meta in _META_FILES_RUN_LEVEL:
            _link(os.path.join(os.path.dirname(output_folder), meta),
                  os.path.join(mirror_trainer_dir, meta))

        _link(os.path.join(output_folder, src_best_name),
              os.path.join(mirror_fold_dir, "checkpoint_best.pth"))
        _link(os.path.join(output_folder, "checkpoint_final.pth"),
              os.path.join(mirror_fold_dir, "checkpoint_final.pth"))

        for meta in _META_FILES_FOLD_LEVEL:
            _link(os.path.join(output_folder, meta), os.path.join(mirror_fold_dir, meta))

        log(f"[DualVal] materialized {mirror_run_id} ({fold_name}) from {src_best_name} "
            f"-> {mirror_fold_dir}/checkpoint_best.pth (hard link)")


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

    def on_train_end(self) -> None:
        super().on_train_end()
        if self.local_rank != 0:
            return
        run_id = os.environ.get("RUN_ID", "")
        if not run_id:
            self.print_to_log_file("[DualVal] RUN_ID env var not set — skipping mirror-directory "
                                    "materialization (predict/eval will need CHECKPOINT= override).")
            return
        materialize_split_runs(self.output_folder, run_id, log=self.print_to_log_file)
