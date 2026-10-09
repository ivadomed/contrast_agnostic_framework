#!/usr/bin/env bash
# CPU-only build of the pre-registered FLATTEN/RAMP intervention inputs. See
# intervention_build.py's own docstring for the full pre-registration.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.."
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh

run_job --name brats_intervention_build --gpus 0 --cpus 4 --mem 16G --time 00:30:00 --wait \
  --log /scratch/${USER}/brats_intervention/logs/build.log -- \
  .venv/bin/python benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1/scripts/intervention_build.py
