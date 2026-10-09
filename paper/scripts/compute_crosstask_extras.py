#!/usr/bin/env python3
"""
Numbers the paper quotes next to tab:meta that the heatmap script itself does not print:

  1. val000 vs val100 (tab:suppl-val100): cross-task test on the tasks that have both
     arms, two-sided and both one-sided directions, Dice and HD95.
  2. Reverse tests (where PALETTE-Aug does not win): per task, p that a competitor is
     BETTER than PALETTE-Aug, Holm over the competitors within the task -- the p_lose of
     meta_task_heatmap._per_task_sig, i.e. the same design as the table's per-task marks.
  3. How much of the null variance of each cross-task test is carried by the units that
     macro_perm_design does NOT enumerate exactly (the normal-approximated remainder):
     sum(rest^2) / sum(all^2) over unit totals, for the tab:meta tests.

Everything goes through benchmark/00_commun_scripts/00_03_evaluate/meta_task_heatmap.py
(load_task_modalities, task_design_entries, _per_task_sig) and stat_tests.py
(macro_perm_design, holm) -- no second implementation of the test.

Usage:  .venv/bin/python paper/scripts/compute_crosstask_extras.py [meta_yaml]
        (default: paper/scripts/meta_task_heatmap_paper.yaml)
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
EVAL = REPO / "benchmark/00_commun_scripts/00_03_evaluate"
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_00_utils"))
sys.path.insert(0, str(EVAL))
from stat_tests import macro_perm_design, ENUM_UNITS, _unit_totals, fmt_p  # noqa: E402

spec = importlib.util.spec_from_file_location("meta_task_heatmap", EVAL / "meta_task_heatmap.py")
MT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MT)

OURS = "auglabAug_v26_6_2_train050_val000"
V100 = "auglabAug_v26_6_2_train050_val100"
COMPS = ["baseline", "synthseg_noEM", "synthseg_EM", "auglab_default", "srcsm"]
OUT = REPO / "paper/generated_results/crosstask_extras.md"


def load_tasks(meta_yaml: Path) -> list:
    os.environ.setdefault("PROJECT_ROOT", str(REPO))
    cfg = yaml.safe_load(meta_yaml.read_text())
    tasks = []
    for t in cfg["tasks"]:
        mods, groups = MT.load_task_modalities(t, str(REPO))
        if mods:
            tasks.append({"name": t["name"], "modalities": mods, "contrast_groups": groups})
    return tasks


def crosstask_entries(tasks, ref, comp, metric):
    present = []
    for t in tasks:
        e = MT.task_design_entries(t["modalities"], ref, comp, metric, t["contrast_groups"], 0)
        if e:
            present.append((t["name"], e))
    entries = [(j, leaf, unit, c) for j, (_, ents) in enumerate(present) for _, leaf, unit, c in ents]
    return entries, [n for n, _ in present]


def remainder_share(entries, K):
    t = np.array(list(_unit_totals(entries, K).values()))
    o = np.argsort(-np.abs(t))
    rest = t[o[ENUM_UNITS:]]
    return float((rest ** 2).sum() / (t ** 2).sum()), t.size


def main():
    meta_yaml = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "paper/scripts/meta_task_heatmap_paper.yaml"
    tasks = load_tasks(meta_yaml)
    L = [f"# Cross-task extras (from {meta_yaml.relative_to(REPO)})", ""]

    L += ["## 1. val100 vs val000 (tasks with both arms)", "",
          "| metric | tasks | macroΔ (val100 − val000) | p two-sided | p val100 better | p val000 better |",
          "|---|---|---|---|---|---|"]
    for metric in ("dice", "hd95"):
        hb = metric == "dice"
        ents, names = crosstask_entries(tasks, V100, OURS, metric)
        d, p2, p_v100 = macro_perm_design(ents, len(names), hb)
        _, _, p_v000 = macro_perm_design(ents, len(names), not hb)
        scale = 100 if metric == "dice" else 1
        L.append(f"| {metric} | {', '.join(names)} | {d * scale:+.2f} | {fmt_p(p2)} | {fmt_p(p_v100)} | {fmt_p(p_v000)} |")

    L += ["", "## 2. Reverse test per task: p that the competitor is BETTER than PALETTE-Aug",
          "(Holm over the 5 competitors within the task; same design as tab:meta's per-task marks)", "",
          "| task | metric | " + " | ".join(COMPS) + " |", "|---|---|" + "---|" * len(COMPS)]
    for metric in ("dice", "hd95"):
        sig = MT._per_task_sig(tasks, OURS, COMPS, metric, metric == "dice")
        for i, t in enumerate(tasks):
            L.append(f"| {t['name']} | {metric} | " + " | ".join(fmt_p(sig[c][i][2]) for c in COMPS) + " |")

    L += ["", f"## 3. Null variance carried by the normal-approximated remainder (ENUM_UNITS={ENUM_UNITS})", "",
          "| metric | competitor | units | remainder share of null variance | p one-sided (ours better, raw) |",
          "|---|---|---|---|---|"]
    for metric in ("dice", "hd95"):
        for c in COMPS:
            ents, names = crosstask_entries(tasks, OURS, c, metric)
            share, n = remainder_share(ents, len(names))
            p1 = macro_perm_design(ents, len(names), metric == "dice")[2]
            L.append(f"| {metric} | {c} | {n} | {share:.3f} | {fmt_p(p1)} |")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
