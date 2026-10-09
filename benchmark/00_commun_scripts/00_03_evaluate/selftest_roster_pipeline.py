#!/usr/bin/env python3
"""End-to-end SELF-TEST of the roster-driven aggregation chain on SYNTHETIC metrics, in a throwaway project tree -- run it for any new
dataset before real predictions exist (and after touching roster_runs.py / write_configs_from_roster.py / ladder_from_roster.py).
It cannot touch real results: everything lives under --workdir (default $SCRATCH/selftest_<dataset>), and PROJECT_ROOT is pointed there so
the generated configs' ${PROJECT_ROOT}/... paths resolve inside it.

Builds  <workdir>/benchmark/02_tasks/<task>/<dataset>/8_results_<dataset>/{01_predictions/<model_type>/<contrast>/roster_run_ids.tsv,
02_metrics/<model_type>/<contrast>[/ablations]/<category>_<RUN_ID>/fold{0,1,2}/eval_all.csv}  for the 7 headline methods + 4 ladder rungs,
then runs, in order: write_configs_from_roster.py -> aggregate_from_config.py (per contrast) -> significance_from_config.py (per contrast)
-> combined_modality_summary.py -> ladder_from_roster.run (per contrast), and asserts every expected output exists and no step failed.

  CPU only; submit through run_job (do not run on a login node):
  run_job --name selftest --gpus 0 --cpus 2 --mem 8G --time 00:20:00 --wait -- .venv/bin/python \
      benchmark/00_commun_scripts/00_03_evaluate/selftest_roster_pipeline.py --dataset isles2022 --task brain_stroke \
      --model-type isles2022_model --contrasts dwi flair --items dwi adc flair
"""
from __future__ import annotations
import argparse, csv, os, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from roster_runs import LADDER, HEADLINE_METHODS  # noqa: E402

CATEGORY = {"baseline": "nnUNet"}  # everything else: auglab (matches the predict wrappers)
BASE = {"baseline": .40, "synthseg_noEM": .45, "synthseg_EM": .47, "auglab_default": .50, "srcsm": .52,
        "auglabAug_v26_6_2_train050_val000": .58, "auglabAug_v26_6_2_train050_val100": .57, "baseline_kmeans": .42,
        "baseline_kmeans_label_remap": .44, "baseline_kmeans_label_remap_voronoi": .46, "v26_6_2_train050_val000": .54}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--task", required=True); ap.add_argument("--model-type", required=True)
    ap.add_argument("--contrasts", nargs="+", required=True); ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--label", default="lesion"); ap.add_argument("--n-cases", type=int, default=40)
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    work = Path(a.workdir or Path(os.environ.get("SCRATCH", "/tmp")) / f"selftest_{a.dataset}").resolve()
    D = work / "benchmark" / "02_tasks" / a.task / a.dataset
    R = D / f"8_results_{a.dataset}"
    ablation = {m for _, _, m, ab in LADDER if ab}
    methods = list(dict.fromkeys(HEADLINE_METHODS + [m for _, _, m, _ in LADDER]))
    rng = np.random.default_rng(0)
    cases = [f"{a.dataset}_{i:04d}" for i in range(1, a.n_cases + 1)]
    for c in a.contrasts:
        pins = []
        for m in methods:
            cat = CATEGORY.get(m, "auglab"); rid = f"{a.dataset}_{c}_{m}_20990101_000000"; pins.append(f"{m}\t{cat}\t{rid}")
            out = R / "02_metrics" / a.model_type / c / ("ablations" if m in ablation else "") / f"{cat}_{rid}"
            for f in range(3):
                (out / f"fold{f}").mkdir(parents=True, exist_ok=True)
                with open(out / f"fold{f}" / "eval_all.csv", "w", newline="") as fh:
                    w = csv.writer(fh); w.writerow(["group", "case", "label", "dice", "hd95"])
                    for it in a.items:
                        pen = 0.0 if it == c else 0.08
                        for cs in cases:
                            w.writerow([it, cs, a.label, float(np.clip(BASE[m] - pen + rng.normal(0, .1), 0, 1)), float(abs(rng.normal(30 - 40 * BASE[m], 8)))])
        pin = R / "01_predictions" / a.model_type / c / "roster_run_ids.tsv"; pin.parent.mkdir(parents=True, exist_ok=True)
        pin.write_text("\n".join(pins) + "\n")
    print("synthetic tree:", D)

    env = dict(os.environ, PROJECT_ROOT=str(work))
    py = sys.executable
    def step(name, cmd):
        print(f"=== {name}"); r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-1500:], r.stderr[-2500:]); sys.exit(f"SELFTEST FAILED at: {name}")
    step("write configs", [py, str(HERE / "write_configs_from_roster.py"), "--dataset-root", str(D), "--model-type", a.model_type,
                           "--contrasts", *a.contrasts, "--items", *a.items])
    cfg = D / f"5_scripts_{a.dataset}" / "06_evaluate" / "configs"
    for c in a.contrasts:
        step(f"aggregate {c}", [py, str(HERE / "aggregate_from_config.py"), str(cfg / f"{a.dataset}_{c}_01_results.yaml")])
        step(f"significance {c}", [py, str(HERE / "significance_from_config.py"), str(cfg / f"{a.dataset}_{c}_significance_01.yaml")])
    step("combined", [py, str(HERE / "combined_modality_summary.py"), str(cfg / f"{a.dataset}_combined_01_results.yaml")])
    for c in a.contrasts:
        ood = [i for i in a.items if i != c]
        step(f"ladder {c}", [py, "-c", "import sys; sys.path.insert(0, %r); from ladder_from_roster import run; "
                                       "run(dataset_root=%r, model_type=%r, contrast=%r, ood_contrasts=%r, task_name=%r)"
                                       % (str(HERE), str(D), a.model_type, c, ood, f"selftest {c}")])
    M = R / "02_metrics" / a.model_type
    expect = [M / c / f for c in a.contrasts for f in ("01_results_summary.md", "01_results_significance.md", "ablations/ladder_summary.md",
                                                       "ablations/ladder_series.json")] + [M / "combined_contrasts" / "01_results_summary.md"]
    missing = [str(p) for p in expect if not p.exists()]
    if missing:
        sys.exit("SELFTEST FAILED, missing outputs:\n  " + "\n  ".join(missing))
    print(f"SELFTEST PASSED ({len(expect)} outputs under {M})")


if __name__ == "__main__":
    main()
