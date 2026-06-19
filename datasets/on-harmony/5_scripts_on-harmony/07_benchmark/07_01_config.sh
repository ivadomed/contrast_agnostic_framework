#!/usr/bin/env bash
# =============================================================================
# Benchmark configuration — source this or override vars before sourcing.
# =============================================================================
# Usage:
#   source scripts/benchmark/config.sh              # use defaults
#   METHOD=synthseg_a source scripts/benchmark/config.sh
#   GPUS=0-1 LABEL_SET=31class source scripts/benchmark/config.sh

# ── Method ────────────────────────────────────────────────────────────────────
# baseline | v26_6 | synthseg_a | synthseg_b
METHOD="${METHOD:-v26_6}"

# ── Dataset ───────────────────────────────────────────────────────────────────
# on_harmony  (extend here for future datasets)
DATASET="${DATASET:-on_harmony}"

# ── Label set ─────────────────────────────────────────────────────────────────
# 7class   → Background + CortGM + WM + CSF + SubcortGM + Brainstem + Cerebellum
# 31class  → All ~31 SynthSeg/FreeSurfer structures (bilateral kept separate)
#            NOTE: requires re-running dataset conversion + preprocessing
LABEL_SET="${LABEL_SET:-7class}"

# ── Hardware ──────────────────────────────────────────────────────────────────
# GPUS: single slot (0), range (0-3), or list (0,2)
GPUS="${GPUS:-0-3}"
# DA_WORKERS: batchgenerators data-augmentation workers (0 = main thread)
#   Use 0 for synthesis-based methods (they generate on the fly).
#   Use 4-16 for baseline (standard nnUNet augmentation).
DA_WORKERS="${DA_WORKERS:-auto}"

# ── Training ──────────────────────────────────────────────────────────────────
N_EPOCHS="${N_EPOCHS:-500}"
# Folds to train (space-separated). Default = all 4.
FOLDS="${FOLDS:-0 1 2 3}"
# RUN_ID: set to an existing run ID to resume, or "auto" to create timestamped.
RUN_ID="${RUN_ID:-auto}"

# ── Paths (from env.sh — single source of truth) ─────────────────────────────
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
# PROJECT_ROOT is set by env.sh — do not override it here.
PY="$PROJECT_ROOT/.venv/bin/python"

case "$DATASET" in
    on_harmony)
        BIDS_DIR="$BIDS_ROOT"
        case "$LABEL_SET" in
            7class)
                DATASET_ID="030"; DATASET_NAME="Dataset030_OnHarmonyT1w"; N_CLASSES=7 ;;
            31class)
                DATASET_ID="031"; DATASET_NAME="Dataset031_OnHarmonyT1w31"; N_CLASSES=31 ;;
            *)  echo "[config] ERROR: unknown label_set '$LABEL_SET'" >&2; return 1 ;;
        esac
        ;;
    *)
        echo "[config] ERROR: unknown dataset '$DATASET'" >&2; return 1 ;;
esac

NNUNET_RAW="$nnUNet_raw"
NNUNET_PRE="$nnUNet_preprocessed"
NNUNET_RES="$nnUNet_results"
EVAL_DIR="$RESULTS_DIR/01_results"
LOG_DIR="/tmp/nnunet_${METHOD}"

# ── Method → trainer name ─────────────────────────────────────────────────────
case "$METHOD" in
    baseline)   TRAINER="nnUNetTrainerOnHarmonyBaseline";  AUTO_DA=16 ;;
    v26_6)      TRAINER="nnUNetTrainerOnHarmonyV26_6";     AUTO_DA=0  ;;
    synthseg_a) TRAINER="nnUNetTrainerOnHarmonySynthSegA"; AUTO_DA=0  ;;
    synthseg_b) TRAINER="nnUNetTrainerOnHarmonySynthSegB"; AUTO_DA=0  ;;
    *)          echo "[config] ERROR: unknown method '$METHOD'" >&2; return 1 ;;
esac

[ "$DA_WORKERS" = "auto" ] && DA_WORKERS=$AUTO_DA

# ── GPU slot list from GPUS ───────────────────────────────────────────────────
# Expand "0-3" → "0 1 2 3", "0,2" → "0 2", "0" → "0"
if [[ "$GPUS" =~ ^[0-9]+-[0-9]+$ ]]; then
    START="${GPUS%-*}"; END="${GPUS#*-}"
    GPU_LIST=$(seq $START $END | tr '\n' ' ')
elif [[ "$GPUS" =~ , ]]; then
    GPU_LIST="${GPUS//,/ }"
else
    GPU_LIST="$GPUS"
fi
GPU_LIST="${GPU_LIST%% }"  # trim trailing space

echo "[config] METHOD=$METHOD  DATASET=$DATASET  LABEL_SET=$LABEL_SET  TRAINER=$TRAINER"
echo "[config] GPUS=$GPUS → slots: $GPU_LIST  DA_WORKERS=$DA_WORKERS  N_EPOCHS=$N_EPOCHS"
