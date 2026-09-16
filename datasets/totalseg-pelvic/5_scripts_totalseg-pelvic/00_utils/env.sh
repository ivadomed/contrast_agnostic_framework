#!/usr/bin/env bash
# Source this at the top of every totalseg-pelvic pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_totalseg-pelvic root)
#
# totalseg-pelvic = pelvic/hip musculoskeletal segmentation from TotalSegmentator's
# public CT + MRI releases (University Hospital Basel). Onboarded 2026-09-15 as this
# project's first musculoskeletal (non-organ) training task, and first task built from
# two UNPAIRED, already-labeled public releases rather than a single archive/challenge —
# closest structural analog is chaos (unpaired CT+MRI, shared organ label set), NOT
# autopet (paired 2-channel single archive).
#
# TWO SOURCE RELEASES, SAME LABEL TAXONOMY, DIFFERENT PATIENTS (unpaired):
#   - TotalSegmentator CT  (this file = CT default; see env_mri.sh for the MRI override,
#     same pattern as chaos t1in/t2spir, autopet ct/pet). Zenodo record 10047292 (DOI
#     10.5281/zenodo.10047292), CC BY 4.0.
#   - TotalSegmentator MRI. Zenodo record 11367005 (DOI 10.5281/zenodo.11367005),
#     CC BY-NC-SA **2.0** (Generic) — NOT 4.0, verified directly from the record's own
#     API metadata field (`license.id: "cc-by-nc-sa-2.0"`), not just page prose. Same NC
#     restriction as duke-breast-mri: fine for this project's research/paper use, NOT
#     eligible for the CC-BY-only git-annex public upload list if that's ever revisited.
#
# CLAIMED-N vs COUNTED-N (pre-flight checklist item #2 — both counted directly from each
# archive's own zip central directory, not trusted from page prose):
#   CT:  1228 volumes claimed, 1228 counted (subject ids s0000-s1429, sparse — the
#        organizers' own internal training pool is larger; 1228 is the public release).
#   MRI: 298 volumes claimed, 298 counted (subject ids s0001-s0298, contiguous). Also
#        cross-checked against the bundled meta.csv: 251 University-Hospital-Basel +
#        47 IDC-origin = 298, matching the source paper's reported 251/47 split exactly.
#
# LABEL SUBSET — pelvic/hip musculoskeletal only (10 classes), chosen because these are
# the ones verified at 100% FILE coverage in BOTH releases (298/298 MRI, 1228/1228 CT
# ship all 10 mask files per case) among the 42 classes the two releases share by name.
# ⚠️ File coverage is NOT content coverage — see 02_nnunet/02_00_convert.py's corrected
# docstring: TotalSegmentator's source scans vary in FOV (chest/head/knee/etc.), so a real
# ~33-39% of candidate cases ship a "correctly" all-zero mask for every one of these 10
# classes (their scan just doesn't cover the pelvis/hip at all). 02_00_convert.py filters
# these out via an actual content check (non-background-voxel test), same pattern as
# autopet's positive/negative filter — see 4_splits_totalseg-pelvic/{ct,mri}/usable_cases.json
# for the real per-modality usable counts after filtering.
#   hip_left, hip_right, sacrum,
#   gluteus_maximus_left, gluteus_maximus_right,
#   gluteus_medius_left, gluteus_medius_right,
#   gluteus_minimus_left, gluteus_minimus_right,
#   iliopsoas_left, iliopsoas_right
# Deliberately NOT abdominal organs (redundant with chaos/amos) and NOT vascular
# structures (ruled out in favor of a genuinely new tissue class for this project).
#
# LABEL SEMANTICS — both releases' ground truth was built via the SAME disclosed,
# semi-automated iterative process (preliminary model -> auto-segment -> manual
# correction -> retrain), reviewed by a board-certified radiologist (12 yrs exp for the
# MRI release) — see the source papers (CT: Wasserthal et al., Radiology:AI 2023;
# MRI: D'Antonoli et al., Radiology 2025). This is disclosed methodology, not an inferred
# claim — no "malignant-only"-style mislabeling risk found on inspection.
#
# MRI IS SEQUENCE-HETEROGENEOUS, NOT ONE NAMED CONTRAST — the bundled meta.csv shows a
# real mix of SE/GR/IR sequences across scanners/institutes; the source paper's own
# framing is "sequence-independent segmentation" by design. Describe it that way in any
# downstream doc — do not imply this is a single T1w/T2w-style contrast the way most of
# this project's other datasets are. Also carries real anisotropic thick-slice spacing in
# some cases (e.g. 4.4mm slice thickness observed in spot checks) — a genuine clinical-
# routine characteristic, handled by normal nnU-Net resampling, not a special case.
#
# CT SUBSAMPLED FOR TRACTABILITY — the public CT release (1228 cases) is far larger than
# any other dataset in this project's roster; 01_create_splits/01_01_create_splits.py
# samples a CT_SAMPLE_N-case CANDIDATE batch (see that script), and 01_02_finalize_splits.py
# builds the final split from whichever of those turn out usable (non-empty content — see
# LABEL SUBSET note above), topping up with more candidates if short of target. The full
# MRI release (298 cases) was used as the sole candidate batch — no top-up for MRI, final
# usable-MRI-N is just whatever survives the content filter. See
# 4_splits_totalseg-pelvic/{ct,mri}/{usable_cases.json,partition.json} for the real final
# counts. This two-phase design (candidates -> convert+filter -> finalize) is itself a
# correction: content usability can only be known after fetching+merging each case's
# masks, so it can't be decided before conversion the way the original single-phase
# splitter assumed.
#
# ORIENTATION — CT cases checked so far are consistently RAS; ~10% of MRI cases ship in
# non-canonical orientation (LPS/LIP/PIL) and are reoriented to RAS (both image and label,
# consistently) during conversion — see 02_nnunet/02_00_convert.py's `_reorient_canonical`.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="totalseg-pelvic"
export NNUNET_DATASET_ID="Dataset130_TotalsegPelvic_CT"
export MODEL_TYPE="totalseg_pelvic_model"
export TRAINING_CONTRAST="${TRAINING_CONTRAST:-ct}"

# EPOCH POLICY: no dataset-specific precedent yet. Using chaos's 200 epochs as the
# closest structural analog (unpaired CT+MRI, shared organ/tissue label set) — revisit if
# a sizing probe suggests otherwise. NOT autopet's 2000 (whole-body PET/CT is a very
# different scale problem) or toothfairy2's 1000.
export NNUNET_NUM_EPOCHS_DEFAULT="${NNUNET_NUM_EPOCHS_DEFAULT:-200}"

# Training runs on TamIA (whole-node H100) per project standing default — see CLAUDE.md
# "Slurm jobs go to TamIA, not Vulcan." Placeholder walltime; RE-TIME from a real
# per-epoch measurement (sizing probe) before trusting this — pelvic-region CT/MRI crops
# are smaller than autopet's whole-body volumes but this project has no direct precedent
# for TotalSegmentator's specific crop sizes yet.
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-24:00:00}"

# common_env config (plain vars — consumed by common_env, not exported to the env):
CE_SUBDIRS="raw preprocessed splits"      # 0_raw + 2_nnUNet/preprocessed + 4_splits
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"

# CT and MRI are UNPAIRED — completely different patient cohorts (unlike chaos's
# t1in/t2spir or autopet's ct/pet, which are the same patients' different
# sequences/channels) — so each modality needs its OWN independent split, not a shared
# splits_final.json. Override common_env's flat default with a per-contrast subdir.
export SPLITS_DIR="${DATASET_ROOT}/4_splits_totalseg-pelvic/${TRAINING_CONTRAST}"

# BIDS conversion — corrected 2026-09-15 (this comment previously said "no BIDS
# conversion" like autopet, which was wrong for this dataset per direct user correction).
# 00_utils/00_01_bidsify.py stages raw -> 1_BIDS_totalseg-pelvic/pelvis-totalsegpelvic/
# (real BIDS anat images + 10 unmerged per-structure derivative masks per case), and
# 02_nnunet/02_00_convert.py reads ONLY from that BIDS layer (no direct Zenodo/network
# access there anymore) and merges the 10 per-structure binary masks into one multi-class
# label map per case. Git-annex leaf name is `pelvis-totalsegpelvic` (anatomy-first, per
# this project's git-annex-bids-prep naming convention — see project_git_annex_bids_upload
# memory), not the dataset's own internal slug `totalseg-pelvic`.

export METRICS_ROOT="${METRICS_ROOT:-${DATASET_ROOT}/8_results_totalseg-pelvic/02_metrics}"
# Unconditional, NOT a ${nnUNet_results:-...} guard — see autopet/00_utils/env.sh's own
# comment for the exact bug this guards against (run_job_pack executing CT then MRI
# wrappers sequentially in one shell must recompute this fresh each time, not keep the
# first contrast's stale value).
export nnUNet_results="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/nnUNet"
export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${DATASET_ROOT}/6_checkpoints_totalseg-pelvic}"
export RESULTS_DIR="${RESULTS_DIR:-${DATASET_ROOT}/8_results_totalseg-pelvic}"

# Raw archive staging location (outside git, huge zips) — used by 02_00_convert.py.
# Both zips are staged on $SCRATCH via plain login-node curl (the compute-node squid
# proxy blocks zenodo.org — confirmed during pre-flight, same workaround already used for
# cirrmri-liver's OSF download).
export TOTALSEG_CT_ZIP="${TOTALSEG_CT_ZIP:-/scratch/paulh/totalseg_preflight/ct_full.zip}"
export TOTALSEG_MRI_ZIP="${TOTALSEG_MRI_ZIP:-/scratch/paulh/totalseg_preflight/mri.zip}"
