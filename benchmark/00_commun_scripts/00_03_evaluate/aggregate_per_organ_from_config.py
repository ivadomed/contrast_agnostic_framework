#!/usr/bin/env python3
"""
Config-driven, per-organ (per-label) aggregation of segmentation evaluation results.

Sibling to aggregate_from_config.py, for datasets whose headline report needs to
show EACH anatomical label separately rather than collapsed into one macro-averaged
number per contrast (amos: liver/right_kidney/left_kidney/spleen; sliver07: liver
only). aggregate_from_config.py's cross_fold_class_mean intentionally POOLS every
label into one number per contrast -- correct for this project's other datasets,
wrong here, where "how well does each individual organ transfer" is the whole
point of the report. aggregate_from_config.py itself is left untouched; this is a
parallel script for this one report shape, not a modification of the canonical path
every other dataset depends on.

Reuses the same fold-loading/report-building primitives amos/sliver07's own former
hand-rolled `06_02_aggregate_results.py` scripts already imported piecemeal --
00_00_utils/eval_aggregate.py's load_run/build_report -- plus this file's own
resolve_run_dir (the same nnUNet_/auglab_/exact prefix search
aggregate_from_config.py uses), so a config-driven per-organ table stays on the
exact same eval_all.csv contract and folder-resolution rules as every other
canonical aggregate entry point -- only the report SHAPE differs (one column per
label, not one pooled number per contrast).

Config YAML format (same style as aggregate_from_config.py):
  title: "AMOS — MR->CT/MRI Generalization Results"
  metrics_dir: "${METRICS_ROOT}/chaos_model/t1in"
  output_dir: "${METRICS_ROOT}/chaos_model/t1in"      # explicit -- never omit
  output_prefix: "00_comparison"
  runs:
    - chaos_t1in_baseline_20260614_153230
    - chaos_t1in_auglabAug_v26_6_2_train050_val100_20260616_112420
    - ...

Runs declared in the config but not found under metrics_dir still get a row (all
"—" cells) rather than being dropped, matching the convention every other
config-driven aggregate entry point in this project follows.

Usage:
  python aggregate_per_organ_from_config.py <config.yaml>
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_00_utils"))  # sibling: 00_commun_scripts/00_00_utils
from eval_aggregate import load_run, build_report  # noqa: E402

_PREFIXES = ("nnUNet_", "auglab_", "")


def resolve_run_dir(metrics_dir: Path, key: str) -> Path | None:
    """Same nnUNet_/auglab_/exact prefix search as aggregate_from_config.py's
    own resolve_run_dir -- kept as a separate copy (not imported cross-module)
    since it's a single 5-line function and this file should stay independently
    readable without needing to open its sibling."""
    for prefix in _PREFIXES:
        p = metrics_dir / f"{prefix}{key}"
        if p.is_dir():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Per-organ aggregation of eval results from a YAML config listing run IDs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("config", help="Path to YAML config file")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    metrics_dir = Path(os.path.expandvars(cfg["metrics_dir"]))
    out_dir = Path(os.path.expandvars(cfg.get("output_dir", cfg["metrics_dir"])))
    title = cfg.get("title", "Aggregated Results")
    output_prefix = cfg.get("output_prefix", "00_comparison")
    run_keys = cfg.get("runs", [])

    if not metrics_dir.is_dir():
        print(f"  note: metrics_dir does not exist yet: {metrics_dir}", file=sys.stderr)

    runs = {}
    any_data = False
    for key in run_keys:
        run_dir = resolve_run_dir(metrics_dir, key)
        if run_dir is None:
            print(f"  blank row {key}: directory not found in {metrics_dir}", file=sys.stderr)
            runs[key] = {}
            continue
        data = load_run(run_dir)
        if not data:
            print(f"  blank row {key}: no eval_all.csv found", file=sys.stderr)
        else:
            any_data = True
        runs[key] = data

    if not any_data:
        sys.exit("No runs with evaluation data found.")

    out_path = out_dir / f"{output_prefix}.md"
    build_report(runs, out_path, title)


if __name__ == "__main__":
    main()
