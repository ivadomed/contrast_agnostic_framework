#!/usr/bin/env python3
"""
BraTS 2024 Glioma cross-fold aggregator — thin shim over the shared aggregator
at datasets/00_commun_scripts/00_03_evaluate/aggregate_results.py.

The aggregation logic (load_run, cross_fold_stats, report + heatmap builders) was
near-identical across datasets and now lives in the commun scripts. This file is
kept (same path, same CLI: --metrics_dir [--run_keys ...]) so
06_02_aggregate_results.sh is unchanged.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_03_evaluate"))
import aggregate_results  # noqa: E402

if __name__ == "__main__":
    aggregate_results.main(default_title="BraTS 2024 Glioma")
