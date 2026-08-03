#!/usr/bin/env bash
# Source this at the top of every picai-prostate pipeline script:
#   source "$(dirname "$0")/../00_utils/env.sh"   (from a step subdir)
#   source "$(dirname "$0")/00_utils/env.sh"       (from 5_scripts_picai-prostate root)
#
# picai-prostate = clinically significant prostate cancer (csPCa) LESION segmentation on
# biparametric MRI (PI-CAI public training/development set, Saha et al. 2022, CC-BY-NC-4.0).
#
# WHY THIS DATASET (see datasets/picai-prostate/README.md for the full rationale): it is an
# INTRA-TISSUE target with no visible interface — a csPCa lesion is a subtle intensity/texture
# change *inside* otherwise normal-looking prostate parenchyma, with no capsule, no edge and no
# anatomical boundary to latch onto. That puts it in the same family as open-ms (MS lesions in
# white matter) and brats2024-glioma (tumour sub-regions inside brain), i.e. the TEXTURE-DEFINED
# side of the causal-ablation ladder — and unlike chaos/msd-spleen/sliver07 organs, which are
# boundary-defined. It is the first NON-BRAIN texture-defined task in the suite.
#
# Contrasts (all three co-registered onto one common grid by 00_01_bidsify.py, so a single
# lesion mask serves every contrast — same arrangement as open-ms):
#   t2w  axial T2-weighted            — TRAINING MODALITY 1 (this file)
#   adc  apparent diffusion coeff.    — TRAINING MODALITY 2 (env_adc.sh)
#   hbv  high b-value DWI (computed)  — test-only third contrast
# t2w vs adc vs hbv are genuinely different contrast regimes (anatomical T2 vs a quantitative
# diffusion parameter map vs a heavily diffusion-weighted image), which is exactly the
# cross-contrast generalisation axis this project measures.
#
# STANDARD results layout (as chaos / open-ms / brats2024-glioma): trained models are
# co-located with their predictions under
#   01_predictions/<MODEL_TYPE>/<TRAINING_CONTRAST>/<nnUNet|auglab>/<RUN_ID>/
# metrics under 02_metrics/<MODEL_TYPE>/<TRAINING_CONTRAST>/. nnUNet_results points at the
# nnUNet-category base; auglab train scripts override NNUNET_RESULTS_BASE → .../auglab.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="picai-prostate"
# DATASET_ROLE training|test-only|both — defaults to "training" in common_env.
export NNUNET_DATASET_ID="Dataset080_PICAI_T2W"
export MODEL_TYPE="picai_prostate_model"
export TRAINING_CONTRAST="${TRAINING_CONTRAST:-t2w}"
# Walltime for a single-fold run_job submission (Vulcan/romane backend). On tamia the
# whole-node pack chain (04_40_tamia_pack_*.sh) overrides walltime via PACK_TIME instead,
# because tamia caps GPU jobs at 24h and chains them with --dependency.
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-60:00:00}"

# common_env config (plain vars — consumed by common_env, not exported to the env):
BIDS_SUBDIR="picai-prostate-bpmri"        # → BIDS_ROOT under 1_BIDS_<name>/
CE_SUBDIRS="raw preprocessed splits"      # 0_raw + 2_nnUNet/preprocessed + 4_splits
source "${DATASET_ROOT}/../00_commun_scripts/00_00_utils/common_env.sh"

export METRICS_ROOT="${DATASET_ROOT}/8_results_picai-prostate/02_metrics"
export nnUNet_results="${DATASET_ROOT}/8_results_picai-prostate/01_predictions/${MODEL_TYPE}/${TRAINING_CONTRAST}/nnUNet"
export CHECKPOINTS_DIR="${DATASET_ROOT}/6_checkpoints_picai-prostate"
export RESULTS_DIR="${DATASET_ROOT}/8_results_picai-prostate"
