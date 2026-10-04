"""Shared causal-ablation-ladder entry point driven by the roster pin file: a dataset's 06_1X_ladder_summary_<contrast>.py is a
~10-line wrapper around run(). Builds the 7 canonical rungs (roster_runs.LADDER: baseline -> +kmeans -> +label_remap ->
+voronoi(noise) -> v26_6_2 real-fill -> +AugLab val000 -> +AugLab val100) from roster_run_ids.tsv and calls
ladder_ood_common.run_ladder(). Rungs 2-5 are addressed as 'ablations/<RUN_ID>' (their metrics live in <contrast>/ablations/)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ladder_ood_common import run_ladder  # noqa: E402
from roster_runs import LADDER, pin_path, read_pins  # noqa: E402


def build_rungs(pins: dict) -> list:
    rungs, missing = [], []
    for label, ingredient, method, in_ablations in LADDER:
        if method not in pins:
            missing.append(method); continue
        rid = pins[method][1]
        rungs.append((label, ingredient, f"ablations/{rid}" if in_ablations else rid))
    if missing:
        raise SystemExit(f"ladder: no pinned run for {missing} -- predict/evaluate them first (a partial ladder compares different rungs)")
    return rungs


def run(*, dataset_root, model_type, contrast, ood_contrasts, task_name, **extra):
    """extra -> forwarded to run_ladder (extra_ood_sources / ood_groups for cross-dataset pooling)."""
    root = Path(dataset_root).resolve(); name = root.name
    metrics_root = root / f"8_results_{name}" / "02_metrics" / model_type / contrast
    rungs = build_rungs(read_pins(pin_path(root, name, model_type, contrast)))
    return run_ladder(task_name=task_name, contrast_label=contrast, metrics_root=metrics_root,
                      ablations_root=metrics_root / "ablations", in_domain=contrast,
                      ood_contrasts=list(ood_contrasts), rungs=rungs, **extra)
