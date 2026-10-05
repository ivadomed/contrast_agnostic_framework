#!/usr/bin/env python3
"""Print the rung 5 -> rung 6 (boundary PV) step of every ladder that has an <ablations_root>_pv/ branch
(produced by running the ladder script with LADDER_PV_BRANCH=1; see ladder_pv_branch.yaml). Reads each
branch's ladder_series.json (OOD pooled Dice/HD95 per rung + the engine's own paired, Holm-corrected
step test) -- no re-implementation of any statistic. Usage: rung56_diff.py <task_root> [...]
 (each arg: a benchmark/02_tasks dir; every */*/8_results_*/02_metrics/**/ablations_pv/ladder_series.json found is shown)"""
import glob, json, sys
rows = []
for root in sys.argv[1:] or ["."]:
    for p in sorted(glob.glob(f"{root}/**/ablations_pv/ladder_series.json", recursive=True)):
        j = json.load(open(p)); lab = j["labels"]
        if not (len(lab) >= 2 and "real fill" in lab[-2].lower() and "PV" in lab[-1]):
            print(f"SKIP (unexpected rung layout) {p}: {lab[-2:]}"); continue
        D, H = j["dice"], j["hd95"]; s = j["rung_step_significance"]; sd, sh = s["dice"][-1], s["hd95"][-1]
        rows.append((j.get("task_name", p), D[-2], D[-1], sd["p_holm"], H[-2], H[-1], sh["p_holm"], sd["n_cases"]))
print(f"{'ladder':52s} {'r5 Dice':>8s} {'r6 Dice':>8s} {'dDice':>7s} {'p_holm':>9s}  {'r5 HD95':>8s} {'r6 HD95':>8s} {'dHD95':>7s} {'p_holm':>9s}  n")
for t, d5, d6, pd, h5, h6, ph, n in rows:
    print(f"{t[:52]:52s} {d5:8.2f} {d6:8.2f} {d6-d5:+7.2f} {pd:9.3g}  {h5:8.2f} {h6:8.2f} {h6-h5:+7.2f} {ph:9.3g}  {n}")
