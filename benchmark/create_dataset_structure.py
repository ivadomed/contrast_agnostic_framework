import os

# Slot type prefixes must match validate_standard_dataset_structure.py exactly
# (e.g. slot 3 is "conf", not "config"; slot 8 results subdirs are 01_predictions/02_metrics).
level_1_structure = ["0_raw", "1_BIDS", "2_nnUNet", "3_conf", "4_splits", "5_scripts", "6_checkpoints", "7_analysis", "8_results", "9_tests"]
# 2_nnUNet required subdirs (validator: REQUIRED_NNUNET_SUBDIRS)
level_2_nnunet_structure = ["raw", "preprocessed"]
level_2_scripts_structure = ["00_utils", "01_create_splits", "02_nnunet", "03_preprocess", "04_train", "05_predict", "06_evaluate"]
# 8_results subdirs:
#   01_predictions/{nnUNet,auglab,...}/{run_id}/fold{k}/{contrast}/*.nii.gz
#   02_metrics/{category}_{run_id}/fold{k}/  +  02_00_aggregated_metrics.md
level_2_results_structure = ["01_predictions", "02_metrics"]

# Canonical 06_evaluate/ entry-point skeleton (established repo-wide 2026-09-27, see
# the project notes' "How experiments work" section). Only 06_00-06_03 are dataset-agnostic
# enough to template here -- 06_04_combined_modality_summary.sh is deliberately NOT
# generated: it only applies to a dataset training >=2 of its own modalities, which
# this script has no way to know, and 06_01_evaluate_run.sh's real body (labels, test
# items, GT paths) is too dataset-specific to fill in -- a TODO-marked stub is scaffolded
# instead so the numbering/naming slot exists from day one without pretending it's
# plug-and-play. Real per-contrast yaml configs can't be templated either (they need
# real run_ids that only exist after training), hence the empty configs/ dir.
#
# These templates use ${PROJECT_ROOT} (exported by 00_utils/env.sh -> common_env.sh)
# instead of a hardcoded parents[N]/relative-hop depth, so they work regardless of
# whether the dataset ends up under benchmark/02_tasks/<task>/<name>/ or elsewhere --
# same depth-independence fix already applied to common_env.sh's own PROJECT_ROOT.

_EVALUATE_PY_TEMPLATE = '''#!/usr/bin/env python3
"""
{dataset} evaluator -- thin shim over the shared, method-agnostic evaluator at
benchmark/00_commun_scripts/00_03_evaluate/evaluate.py (Dice / HD95 per case per
label). Same CLI; see that header.

TODO: describe this dataset's label set here (ids, any --label_map remap needed
for cross-dataset evaluation, etc.) -- see toothfairy2's or totalseg-pelvic's
06_00_evaluate.py for real examples of this docstring.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise SystemExit("PROJECT_ROOT not set -- source 00_utils/env.sh before running this script")
sys.path.insert(0, str(Path(PROJECT_ROOT) / "benchmark" / "00_commun_scripts" / "00_03_evaluate"))
from evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
'''

_EVALUATE_RUN_SH_TEMPLATE = '''#!/usr/bin/env bash
# Evaluate one {dataset} prediction run (all folds x test item(s)), scoring against
# nnUNet_raw ground truth. Writes eval_all.csv + eval_summary.md per fold under
#   8_results_{dataset}/02_metrics/{dataset}_model/<contrast>/<CATEGORY>_<RUN_ID>/fold{{F}}/
#
# TODO this is a scaffolded stub, not a working script yet -- fill in:
#   - ITEMS (test item names, e.g. one per contrast/modality)
#   - LABELS (this dataset's foreground label names, matching dataset.json)
#   - DATASET_ID / GT_DIR (where nnUNet_raw ground truth actually lives)
# See totalseg-pelvic's or toothfairy2's 06_01_evaluate_run.sh for a complete
# reference implementation of this same pattern.
#
# Usage: bash 06_01_evaluate_run.sh <RUN_ID> <CATEGORY:nnUNet|auglab> [FOLD(default all)]
#
# CATEGORY must be passed explicitly and correctly -- a wrong CATEGORY does NOT error,
# it silently writes an empty, _logs-only metrics dir. Check eval_all.csv actually
# exists before trusting a run finished.
set -euo pipefail
source "$(dirname "${{BASH_SOURCE[0]}}")/../00_utils/env.sh"
cd "${{PROJECT_ROOT}}"

RUN_ID="${{1:?need RUN_ID}}"
CATEGORY="${{2:?need CATEGORY (nnUNet|auglab)}}"
FOLD_ARG="${{3:-all}}"

echo "TODO: 06_01_evaluate_run.sh for {dataset} is a scaffolded stub -- fill in ITEMS/LABELS/GT_DIR" >&2
exit 1
'''

_AGGREGATE_FROM_CONFIG_SH_TEMPLATE = '''#!/usr/bin/env bash
# Aggregate a {dataset} results table from a YAML config (canonical shared driver --
# do NOT compute ad-hoc tables or p-values; the inline "sig. vs ref" column is auto-wired).
#   bash 06_02_aggregate_from_config.sh [configs/{dataset}_01_results.yaml]
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
source "${{HERE}}/../00_utils/env.sh"
cd "${{PROJECT_ROOT}}"
CFG="${{1:-configs/{dataset}_01_results.yaml}}"
[[ "$CFG" != /* ]] && CFG="${{HERE}}/${{CFG}}"
.venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/aggregate_from_config.py "${{CFG}}"
'''

_SIGNIFICANCE_FROM_CONFIG_SH_TEMPLATE = '''#!/usr/bin/env bash
# Full paired-significance report for {dataset} (canonical shared driver).
#   bash 06_03_significance_from_config.sh [configs/{dataset}_01_results.yaml]
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
source "${{HERE}}/../00_utils/env.sh"
cd "${{PROJECT_ROOT}}"
CFG="${{1:-configs/{dataset}_01_results.yaml}}"
[[ "$CFG" != /* ]] && CFG="${{HERE}}/${{CFG}}"
.venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/significance_from_config.py "${{CFG}}"
'''


def _write_if_absent(path, content, executable=False):
    """Write a template file only if it doesn't already exist (idempotent/safe to
    re-run against a partially-populated real dataset without clobbering real work)."""
    if os.path.exists(path):
        return False
    with open(path, "w") as f:
        f.write(content)
    if executable:
        os.chmod(path, 0o755)
    return True


def _seed_06_evaluate_skeleton(dataset_name, eval_dir):
    """Scaffold the canonical 06_evaluate/ entry-point skeleton established repo-wide
    2026-09-27 (06_00 evaluate.py shim -> 06_01 evaluate_run.sh -> 06_02
    aggregate_from_config.sh -> 06_03 significance_from_config.sh). See the module-level
    comment above for what is/isn't templated and why."""
    os.makedirs(os.path.join(eval_dir, "configs"), exist_ok=True)
    _write_if_absent(
        os.path.join(eval_dir, f"06_00_evaluate_{dataset_name}.py"),
        _EVALUATE_PY_TEMPLATE.format(dataset=dataset_name),
        executable=True,
    )
    _write_if_absent(
        os.path.join(eval_dir, "06_01_evaluate_run.sh"),
        _EVALUATE_RUN_SH_TEMPLATE.format(dataset=dataset_name),
        executable=True,
    )
    _write_if_absent(
        os.path.join(eval_dir, "06_02_aggregate_from_config.sh"),
        _AGGREGATE_FROM_CONFIG_SH_TEMPLATE.format(dataset=dataset_name),
        executable=True,
    )
    _write_if_absent(
        os.path.join(eval_dir, "06_03_significance_from_config.sh"),
        _SIGNIFICANCE_FROM_CONFIG_SH_TEMPLATE.format(dataset=dataset_name),
        executable=True,
    )


def create_dataset_structure(dataset_name):
    base_path = dataset_name
    for level in level_1_structure:
        os.makedirs(os.path.join(base_path, level + "_" + dataset_name), exist_ok=True)
    for level in level_2_nnunet_structure:
        os.makedirs(os.path.join(base_path, "2_nnUNet_" + dataset_name, level), exist_ok=True)
    for level in level_2_scripts_structure:
        os.makedirs(os.path.join(base_path, "5_scripts_" + dataset_name, level), exist_ok=True)
    for level in level_2_results_structure:
        os.makedirs(os.path.join(base_path, "8_results_" + dataset_name, level), exist_ok=True)
    _seed_06_evaluate_skeleton(
        dataset_name,
        os.path.join(base_path, "5_scripts_" + dataset_name, "06_evaluate"),
    )
    # Drop a `.gitkeep` in every created dir so the (often data-only, gitignored)
    # folder structure is preserved on GitHub. Force-add them with:
    #   git add -f benchmark/<dataset>/**/.gitkeep
    # (see benchmark/seed_skeleton_gitkeep.py to backfill existing datasets).
    for dirpath, _dirnames, _files in os.walk(base_path):
        keep = os.path.join(dirpath, ".gitkeep")
        if not os.path.exists(keep):
            open(keep, "a").close()

if __name__ == "__main__":
    dataset_name = input("Enter the name of the dataset: ")
    create_dataset_structure(dataset_name)
    print(f"Dataset structure for '{dataset_name}' created successfully at {os.path.join('.', dataset_name)}.")