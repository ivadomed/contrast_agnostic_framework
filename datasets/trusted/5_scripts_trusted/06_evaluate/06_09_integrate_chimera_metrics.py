#!/usr/bin/env python3
"""
Fold the chimera experiment's kidney metric into the STANDARD per-modality summary,
so it shows up as a third modality column ("chimera") next to ct/us in
02_01_summary_by_modality.md and the 02_01_heatmap_{dice,hd95}.png heatmaps.

The chimera eval (06_06/06_07) writes an ISOLATED namespace
  METRICS_ROOT/chaos_model/<contrast>/chimera/<cat>_<run>/fold{k}/chimera_metrics.csv
(with two labels: kidney + kidney_centerprior control). This step copies ONLY the
`kidney` rows into the standard run dir
  METRICS_ROOT/chaos_model/<contrast>/<cat>_<run>/fold{k}/chimera_metrics.csv
then re-runs the shared summarize_fold with --groups ct us chimera so each fold's
eval_all.csv carries all three modalities (WITHOUT clobbering ct/us — all present
groups are passed). Run 06_03_aggregate_results.sh afterwards to redraw the heatmaps.

Idempotent. Login-safe (CSV only). The center-prior control stays out of the standard
table (it lives in 06_08's chimera_experiment_summary.md).

    python 06_09_integrate_chimera_metrics.py --metrics_root <…/02_metrics/chaos_model>
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

SF = Path(__file__).resolve().parents[3] / "00_commun_scripts" / "00_03_evaluate" / "summarize_fold.py"
VENV = Path(__file__).resolve().parents[4] / ".venv" / "bin" / "python"

# Modality label shown in the standard summary/heatmaps. Named to make explicit that
# only the pasted US kidney(s) are scored (via the ROI), not the surrounding CT.
CHIMERA_MODALITY = "chimera (us-kidney)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_root", required=True, help="…/02_metrics/chaos_model")
    ap.add_argument("--namespace", default="chimera_fov",
                    help="chimera metrics namespace to integrate (default: chimera_fov, "
                         "the FOV-restricted side-aware eval; use 'chimera' for the ROI control)")
    a = ap.parse_args()
    root = Path(a.metrics_root)
    n_folds = 0
    for contrast_dir in sorted(root.glob("*")):
        chim_base = contrast_dir / a.namespace
        if not chim_base.is_dir():
            continue
        for chim_run in sorted(chim_base.glob("*")):          # <cat>_<run>
            std_run = contrast_dir / chim_run.name
            if not std_run.is_dir():
                print(f"  skip {chim_run.name}: no standard run dir", file=sys.stderr); continue
            for fold in sorted(chim_run.glob("fold*")):
                src = fold / "chimera_metrics.csv"
                if not src.exists():
                    continue
                rows = [r for r in csv.DictReader(open(src)) if r["label"] == "kidney"]
                if not rows:
                    continue
                for r in rows:                       # relabel the modality for the summary
                    r["group"] = CHIMERA_MODALITY
                dst_fold = std_run / fold.name
                dst_fold.mkdir(parents=True, exist_ok=True)
                (dst_fold / "chimera_metrics.csv").unlink(missing_ok=True)   # drop stale "chimera" file
                with (dst_fold / f"{CHIMERA_MODALITY}_metrics.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["group", "case", "label", "dice", "hd95"])
                    w.writeheader(); w.writerows(rows)
                groups = [g for g in ("ct", "us", CHIMERA_MODALITY) if (dst_fold / f"{g}_metrics.csv").exists()]
                subprocess.run([str(VENV), str(SF), str(dst_fold), chim_run.name,
                                fold.name.replace("fold", ""),
                                "--group-col", "modality", "--groups-word", "Modalities",
                                "--label-word", "Organs", "--groups", *groups], check=True)
                n_folds += 1
    print(f"Integrated chimera kidney metric into {n_folds} standard fold dirs. "
          f"Now run 06_03_aggregate_results.sh (both contrasts) to redraw heatmaps.")


if __name__ == "__main__":
    main()
