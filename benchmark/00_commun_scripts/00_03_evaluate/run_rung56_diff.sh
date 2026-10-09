#!/usr/bin/env bash
# Read the rung 5 -> rung 6 (boundary PV) difference WITHOUT adding rung 6 to the ladders: re-run each
# registered ladder script (ladder_pv_branch.yaml) with LADDER_PV_BRANCH=1, which writes only the sibling
# <ablations_root>_pv/ branch (the ladders' own ablations/ outputs are regenerated unchanged), then print
# the step table (rung56_diff.py). One small CPU job via run_job (no python on the login node).
#   bash run_rung56_diff.sh [ladder script ...]     (default: every ladder that is registered AND has rung-6 metrics)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../../.." && pwd)"
T=benchmark/02_tasks
source "${PROJECT_ROOT}/${T}/brain_ms/open-ms/5_scripts_open-ms/00_utils/env.sh"   # run_job + PROJECT_ROOT only
cd "${PROJECT_ROOT}"
DEFAULT=(
  "$T/brain_ms/open-ms/5_scripts_open-ms/06_evaluate/06_19_ladder_summary_ood.py"
  "$T/brain_ms/open-ms/5_scripts_open-ms/06_evaluate/06_18_ladder_summary_t1w.py"
  "$T/mandible_healthy/toothfairy2/5_scripts_toothfairy2/06_evaluate/06_05_ladder_summary.py"
  "$T/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic/06_evaluate/06_10_ladder_summary_ct.py"
  "$T/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic/06_evaluate/06_11_ladder_summary_mri.py"
  "$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_13_ladder_summary.py"
  "$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_14_ladder_summary_t2w.py"
  "$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_16_ladder_summary_t2f.py"
  "$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_33_ladder_summary_t1c.py"
  "$T/abdomen_healthy/chaos/5_scripts_chaos/06_evaluate/06_34_ladder_summary_t1in.py"
  "$T/abdomen_healthy/chaos/5_scripts_chaos/06_evaluate/06_35_ladder_summary_t2spir.py"
  "$T/brain_healthy/on-harmony/5_scripts_on-harmony/06_evaluate/06_10_ladder_summary.py"
  "$T/brain_healthy/on-harmony/5_scripts_on-harmony/06_evaluate/06_11_ladder_summary_t2w.py"
  "$T/brain_healthy/on-harmony/5_scripts_on-harmony/06_evaluate/06_12_ladder_summary_dwi_ap.py"
  "$T/breast_cancer/ispy2/5_scripts_ispy2/06_evaluate/06_10_ladder_summary_t1wce.py"
  "$T/breast_cancer/ispy2/5_scripts_ispy2/06_evaluate/06_11_ladder_summary_t2w.py"
)
SCRIPTS=("${@:-${DEFAULT[@]}}")
LOG="${PROJECT_ROOT}/benchmark/01_commun_results/_rung56_diff_$(date +%Y%m%d_%H%M%S).log"
CMD=""; for s in "${SCRIPTS[@]}"; do CMD+=".venv/bin/python '${s}' > /dev/null || echo 'LADDER FAILED: ${s}'; "; done
CMD+=".venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/rung56_diff.py ${T}"
LADDER_PV_BRANCH=1 run_job --name rung56_diff --gpus 0 --cpus 4 --mem 16G --time 00:45:00 --wait --log "${LOG}" -- bash -c "export LADDER_PV_BRANCH=1; ${CMD}"
cat "${LOG}"
