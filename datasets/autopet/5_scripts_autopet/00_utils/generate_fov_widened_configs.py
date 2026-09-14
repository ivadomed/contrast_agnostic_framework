#!/usr/bin/env python3
"""
Generate AutoPET-specific, FOV-widened copies of the shared AugLab training configs.

WHY THIS EXISTS (autopet is the first dataset in this project with a real train/eval
FOV mismatch — user-directed fix, 2026-09-13): AutoPET trains on whole-body PET/CT, but
its cross-modality companion (HNTS-MRG, MRI) is a tightly-cropped head-and-neck-only
volume. Measured directly (not assumed) from the real preprocessed nnU-Net plan:
  - AutoPET's own 3d_fullres training PATCH is 128^3 voxels @ [3.0, 2.036, 2.036] mm
    spacing = a 384 x 260 x 260 mm physical footprint (S-I x A-P x L-R).
  - HNTS-MRG's real physical FOV (measured from a real case) is 256 x 256 x 164 mm.
  - In-plane (260 vs 256mm) is ALREADY closely matched — no correction needed there.
  - Superior-inferior (384 vs 164mm) is the real gap: AutoPET patches show ~2.3x more
    craniocaudal context than HNTS-MRG ever will.

FIX: widen nnU-Net's/AugLab's spatial "scaling" (zoom) augmentation, and switch it to
INDEPENDENT per-axis scaling (p_synchronize_scaling_across_axes: 0) rather than widening
isotropically — an isotropic widening calibrated to fix the S-I gap would ALSO
over-shrink the in-plane axes, which don't need correcting. Independent per-axis scaling
lets training patches sometimes present a much tighter S-I window (approaching HNTS-MRG's
actual craniocaudal tightness) without forcing an unrealistic in-plane crop every time.

⚠️ SCALE DIRECTION — verified EMPIRICALLY (a tiny synthetic-cube test volume through each
real transform class, not read off a docstring — the two libraries turned out to use
OPPOSITE conventions, which would have been a costly, silent, wrong-direction bug if
assumed instead of tested):
  - nnUNetSpatialTransform (batchgeneratorsv2): confirmed via its own source comment
    "larger numbers = smaller objects!" — a SMALLER scale value means TIGHTER framing
    (more physical extent excluded from the patch — what we want more of, to match
    HNTS-MRG). So this one widens the LOWER bound down (0.7 -> 0.4).
  - AffineTransform (kornia's get_affine_matrix3d + warp_affine3d, used by srcsm): tested
    directly — scale=0.5 shrank a test object from 216 to 27 voxels (zoom OUT), scale=1.5
    grew it to 704 voxels (zoom IN). OPPOSITE of the above: a LARGER scale value means
    TIGHTER framing here. So this one widens the UPPER bound up (1.32 -> 2.3), not the
    lower bound down.
To roughly reach HNTS-MRG's S-I tightness (384mm patch -> 164mm target, ratio ~0.43 / ~2.34x
depending on which direction a given library's convention runs), each widened bound targets
that same ~2.3x factor from its own library's neutral (1.0) point.

Concretely, versus the shared default:
  nnUNetSpatialTransform.p_scaling:                        0.2  -> 0.3   (seen more often)
  nnUNetSpatialTransform.scaling:                    [0.7, 1.4] -> [0.4, 1.4]
  nnUNetSpatialTransform.p_synchronize_scaling_across_axes:  1  -> 0     (independent per-axis)

⚠️ srcsm's AffineTransform.scale is DELIBERATELY LEFT UNWIDENED (2026-09-14, reverted after
a real training failure): the first version of this script widened it [0.72, 1.32] ->
[0.72, 2.3] using the same "target ~2.3x from neutral" logic as above. The TamIA sizing
probe caught `train_loss nan` from epoch 0 onward, persistent across 6+ epochs — not a
transient init artifact. Root cause (plausible, not exhaustively proven): srcsm's config
itself documents [0.72, 1.32] as a faithful reproduction of the SRCSM paper's own spatial
recipe (see this file's `__notes__.spatial_scale`), and `SemRandConvGPU`'s intensity path
has its own FIXED normalization assumptions (`clamp_min=-1.0`, hardcoded
`robust_quantiles`) tuned for that range — an aggressively zoomed-OUT patch (scale up to
2.3, kornia convention = tighter/zoomed-in at HIGH values so 2.3 is actually zoomed IN,
not out — but the point stands that 2.3 is far outside the paper-validated range) likely
produces degenerate per-channel intensity statistics that break RCNet's normalization.
Rather than debug the SemRandConvGPU/kornia interaction further, srcsm is excluded from
this widening entirely and kept paper-faithful — it did not have the FOV-mismatch problem
being solved here to begin with (srcsm's own spatial recipe was never meant to need
HNTS-MRG-scale correction), so excluding it costs nothing. All other methods (which use
nnUNetSpatialTransform, not AffineTransform, for scaling) keep the widened
[0.4, 1.4]/independent-per-axis fix — verified clean, healthy decreasing loss on
baseline/synthseg_EM/ours in the same probe run.

This widening is applied to EVERY method (baseline included, via a matching trainer-level
override in autopet/trainers/base.py — see nnUNetTrainerAutoPETBase) so the 6-method
comparison stays apples-to-apples; none of them get an unfair FOV-robustness advantage.

VAL configs (p_scaling=0, p_rotation=0 already — deterministic, no augmentation) are
copied VERBATIM, unmodified — evaluation should never get random FOV augmentation.

Output: datasets/autopet/5_scripts_autopet/00_utils/auglab_configs_fov/<same filename>
Every 04_0X_train_*.sh wrapper's AUGLAB_CONFIGS_DIR points here instead of the shared
sub-workspaces/auglab_workspace/AugLab/auglab/configs/ — a deliberate, documented
deviation from this project's usual "reuse the shared, dataset-agnostic config" rule,
justified by autopet being the first dataset with a real train/eval FOV mismatch this
severe. Re-run this script if the shared upstream configs ever change and need re-syncing
(diffs would show which fields upstream touched vs. this script's own FOV overrides).

Usage: .venv/bin/python generate_fov_widened_configs.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[4] / "sub-workspaces" / "auglab_workspace" / "AugLab" / "auglab" / "configs"
OUT_DIR = Path(__file__).resolve().parent / "auglab_configs_fov"

NEW_SCALING = [0.4, 1.4]          # nnUNetSpatialTransform (batchgeneratorsv2): smaller=tighter
NEW_P_SCALING = 0.3
# srcsm's AffineTransform.scale is intentionally NOT widened — see module docstring
# ("srcsm's AffineTransform.scale is DELIBERATELY LEFT UNWIDENED") for why.

# Configs actually referenced by autopet's 04_train wrappers (train-side, need widening)
# and val-side (copied verbatim, unmodified).
TRAIN_CONFIGS = [
    "transform_params_gpu_default01-23.json",                                          # auglab_default
    "transform_params_gpu_default01-23_Synthseg.json",                                 # synthseg_noEM
    "transform_params_gpu_default01-23_Synthseg_EM.json",                              # synthseg_EM
    "transform_params_gpu_srcsm_semrandconv.json",                                     # srcsm (AffineTransform, not nnUNetSpatialTransform)
    "transform_params_gpu_baseline_kmeans_spatialDA_train050.json",                    # ladder rung 2
    "transform_params_gpu_baseline_kmeans_label_remap_spatialDA_train050.json",        # ladder rung 3
    "transform_params_gpu_baseline_kmeans_label_remap_voronoi_spatialDA_train050.json",# ladder rung 4
    "transform_params_gpu_v26_6_2_synth_spatialDA_train050.json",                      # ladder rung 5
    "transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json",  # OURS (DualVal train)
]
VAL_CONFIGS = [
    "transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json",         # OURS + rung 5 val
]


def _widen_train_config(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    spat = cfg.get("nnUNetSpatialTransform")
    if spat is not None and spat.get("p_scaling", 0) > 0:
        spat["scaling"] = NEW_SCALING
        spat["p_scaling"] = NEW_P_SCALING
        spat["p_synchronize_scaling_across_axes"] = 0
    # AffineTransform (srcsm only) is deliberately left untouched — see module docstring.
    return cfg


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in TRAIN_CONFIGS:
        src = SRC_DIR / name
        cfg = json.loads(src.read_text())
        widened = _widen_train_config(cfg)
        (OUT_DIR / name).write_text(json.dumps(widened, indent=2))
        print(f"[fov_configs] widened -> {name}")

    for name in VAL_CONFIGS:
        src = SRC_DIR / name
        (OUT_DIR / name).write_text(src.read_text())
        print(f"[fov_configs] copied verbatim (val, no augmentation) -> {name}")

    print(f"[fov_configs] done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
