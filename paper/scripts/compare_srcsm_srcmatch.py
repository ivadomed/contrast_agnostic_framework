#!/usr/bin/env python3
"""
SRCSM with its published test-time source matching ("srcsm_srcmatch", metrics <cat>_<srcsm run>_srcmatch written by
scripts/cluster/srcsm_srcmatch/run_setting.sh) next to plain SRCSM and every other method, on the task-level
estimand of tab:meta (each task's own combined config, run unchanged through combined_modality_summary.py with one
extra run key per modality and an explicit scratch output_dir -- the real configs and tables are never touched).

Reports, per task and across tasks (same patient-level macroDelta sign-flip design as tab:meta's p column):
  * Dice/HD95 of SRCSM and SRCSM + matching,
  * p that PALETTE-Aug beats SRCSM + matching (one-sided, Holm over the competitors incl. the new row),
  * p that matching changes SRCSM (two-sided, SRCSM + matching vs SRCSM).
A task enters only if EVERY modality has its srcmatch metrics (else it is listed as missing).

Usage:  .venv/bin/python paper/scripts/compare_srcsm_srcmatch.py
"""
from __future__ import annotations

import copy
import os
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_palette_alone_per_task as P  # noqa: E402  (run_task, meta_modalities, crosstask, MT, _holm, _fmt_p)

REPO = HERE.parent.parent
OUT = REPO / "paper" / "generated_results" / "srcsm_srcmatch"
OURS = "auglabAug_v26_6_2_train050_val000"
COMPS = ["baseline", "synthseg_noEM", "synthseg_EM", "auglab_default", "srcsm", "srcsm_srcmatch"]


def main():
    os.environ.setdefault("PROJECT_ROOT", str(REPO))
    meta = yaml.safe_load((HERE / "meta_task_heatmap_paper.yaml").read_text())
    scratch = Path(tempfile.mkdtemp(prefix="srcsm_srcmatch_", dir=os.environ.get("SCRATCH")))
    rows, missing, xt = [], [], []
    for t in meta["tasks"]:
        cfg_path = Path(os.path.expandvars(t["config"].replace("${PROJECT_ROOT}", str(REPO))))
        ds_root = cfg_path.parents[2].parent
        ds, tdir = ds_root.name, str(ds_root.relative_to(REPO / "benchmark/02_tasks").parent)
        mr = str(ds_root / f"8_results_{ds}" / "02_metrics")
        cfg = yaml.safe_load(cfg_path.read_text())
        new = copy.deepcopy(cfg)
        ok = True
        os.environ["METRICS_ROOT"] = mr
        for m in new["modalities"]:
            rid = m.get("runs", {}).get("srcsm")
            if not rid:
                ok = False; missing.append(f"{t['name']}/{m['name']}: no SRCSM run"); continue
            for s in (m.get("sources") or [{"metrics_dir": m["metrics_dir"]}]):
                md = Path(os.path.expandvars(str(s["metrics_dir"]).replace("${PROJECT_ROOT}", str(REPO))))
                if not any(md.glob(f"*_{rid}_srcmatch/fold*/eval_all.csv")):
                    ok = False; missing.append(f"{t['name']}/{m['name']}: no {rid}_srcmatch in {md}")
            m["runs"] = dict(m["runs"], srcsm_srcmatch=f"{rid}_srcmatch")
        if not ok:
            continue
        res = P.run_task(new, ds, tdir, scratch / f"task_{ds}")
        mods, groups = P.meta_modalities(new, mr)
        task = {"name": t["name"], "modalities": mods, "contrast_groups": groups}
        xt.append(task)
        two = {}
        for metric in ("dice", "hd95"):
            ents = P.MT.task_design_entries(mods, "srcsm_srcmatch", "srcsm", metric, groups, 0)
            two[metric] = P._mpd(ents, 1, metric == "dice")[1] if ents else float("nan")
        rows.append((t["name"], res, two))
        print(t["name"], {k: res.get("dice", {}).get(k) for k in ("srcsm", "srcsm_srcmatch", OURS)}, flush=True)

    L = ["# SRCSM with its test-time source matching", "",
         "Task-level estimand of tab:meta. `sig` = Holm-corrected one-sided p that PALETTE-Aug (val000) beats the row, "
         "within the task (the combined table's own `sig. vs ref`). `match vs plain` = two-sided p, SRCSM + matching vs SRCSM.", ""]
    for metric, name in (("dice", "Dice (%)"), ("hd95", "HD95 (mm)")):
        L += [f"## {name}", "", "| task | SRCSM | sig | SRCSM + matching | sig | PALETTE-Aug | match vs plain p |", "|---|---|---|---|---|---|---|"]
        for task, res, two in rows:
            d = res.get(metric, {})
            g = lambda k, i: ("--" if d.get(k, (None, None))[i] is None else f"{d[k][i]:.3g}")
            L.append(f"| {task} | {g('srcsm', 0)} | {g('srcsm', 1)} | {g('srcsm_srcmatch', 0)} | {g('srcsm_srcmatch', 1)} | "
                     f"{g(OURS, 0)} | {P._fmt_p(two[metric])} |")
        L.append("")
    L += [f"## Cross-task (tasks: {', '.join(t['name'] for t in xt)})", ""]
    for metric in ("dice", "hd95"):
        hb = metric == "dice"
        raw = [P.crosstask(xt, OURS, c, metric, hb)[2] for c in COMPS]
        L.append(f"- {metric}: PALETTE-Aug better than " + ", ".join(f"{c} p_holm={P._fmt_p(p)}" for c, p in zip(COMPS, P._holm(raw))))
        d, p2, _ = P.crosstask(xt, "srcsm_srcmatch", "srcsm", metric, hb)
        L.append(f"- {metric}: SRCSM + matching minus SRCSM macroΔ={d:+.4f}, two-sided p={P._fmt_p(p2)}")
    L += ["", "missing:"] + (missing or ["none"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
