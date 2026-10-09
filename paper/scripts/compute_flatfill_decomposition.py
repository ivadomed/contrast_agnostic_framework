#!/usr/bin/env python3
"""
Rung 4.5 (FLAT fill) decomposition of the fill swap, 2026-10-08 (Paul: "does the gain come from noise removal or
texture addition?"). The ladder's fill swap is rung 4 (noise fill) -> rung 5 (real fill); rung 4.5 fills every region
of the same partition with its target mean and keeps PALETTE's label step (scripts/cluster/rung45_flat/), so

    noise removal    = rung 4   -> rung 4.5
    texture addition = rung 4.5 -> rung 5      (the ONLY difference: the within-region term alpha_c * (x - mean_c))

and the two add up to the fill swap exactly (same cases, same pooling).

Test = the paper's own task-level test (make_per_contrast_curves.panel_pooled / _panel_pooled_grouped: every training
modality of the task, held-out contrasts only, same-patient pairs merged by the engine's helpers, two-sided paired
Wilcoxon), re-run on a different PAIR of run keys. Deltas follow panel_pooled's estimands but are re-derived from the
case values: generic ladders = mean over merged patient pairs; grouped ladders (Abdomen, Breast) = equal weight per OOD
contrast group of the case means, averaged over the task's ladders (panel_pooled reads that one off the stored rung-4/5
series, which do not exist for rung 4.5). The rung-4.5 run key of
each ladder is its rung-4 key with the lblvor RUN_ID swapped for the setting's flatfill RUN_ID.

SELF-CHECK (always run): with the pair (rung 4, rung 5) this script must reproduce panel_pooled's task-level p and
delta for every panel; it exits non-zero otherwise.

Ladders without rung-4.5 metrics print "pending". Usage:
  .venv/bin/python paper/scripts/compute_flatfill_decomposition.py [--metric dice|hd95]
"""
from __future__ import annotations

import argparse
import copy
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_per_contrast_curves as mpc  # noqa: E402

SCRATCH = Path(os.environ.get("SCRATCH", "/scratch/paulh"))
R4_IDS = SCRATCH / "rung4_lblvor/run_ids.txt"
R45_IDS = [SCRATCH / "rung45_flat/run_ids.txt", SCRATCH / "rung45_flat/run_ids_tamia.txt"]


def _ids(paths):
    out = []
    for p in paths:
        if Path(p).exists():
            out += [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]
    return out


R4 = _ids([R4_IDS])
R45 = _ids(R45_IDS)


def flat_key(r4_key: str):
    """rung-4 key -> rung-4.5 key (None if no flatfill run is registered for that setting)."""
    hits = [r for r in R4 if r in r4_key]
    if not hits:          # this ladder is not yet re-pointed to its lblvor rung 4 -> nothing comparable
        return None
    if len(hits) > 1:
        raise ValueError(f"rung-4 key {r4_key!r} matches {len(hits)} lblvor RUN_IDs")
    stem = re.sub(r"_lblvor_\d{8}_\d{6}$", "", hits[0])
    cand = [r for r in R45 if re.fullmatch(re.escape(stem) + r"_flatfill_\d{8}_\d{6}", r)]
    if len(cand) > 1:
        raise ValueError(f"{stem}: {len(cand)} flatfill RUN_IDs {cand}")
    return r4_key.replace(hits[0], cand[0]) if cand else None


def _generic(ladders, metric):
    """make_per_contrast_curves.panel_pooled's generic branch, without its single-ladder shortcut."""
    pairs_by_contrast = {}
    for d in ladders:
        keys = d["run_keys"]
        prev_key, cur_key = keys[mpc.FILL - 1], keys[mpc.FILL]
        oods = set(mpc.ood_contrasts(d))
        train = d.get("contrast_label", "?")
        for root, subdir in mpc.metrics_roots(d):
            ns = mpc._dataset_name(root)
            pk, ck = mpc._src_key(subdir, prev_key), mpc._src_key(subdir, cur_key)
            prev = mpc.load_case_means(mpc.resolve_run_dir(root, pk), metric)
            cur = mpc.load_case_means(mpc.resolve_run_dir(root, ck), metric)
            for item in set(prev) | set(cur):
                if oods and item not in oods and not any(o.endswith("/" + item) for o in oods):
                    continue
                a, b = prev.get(item, {}), cur.get(item, {})
                bucket = pairs_by_contrast.setdefault(f"{train}/{item}", {})
                for k in set(a) & set(b):
                    bucket[f"{ns}|{k}"] = (a[k], b[k])
    x, y = mpc._merge_patient_pairs(pairs_by_contrast)
    # single-ladder panels: panel_pooled reports the engine's series delta = equal weight per held-out contrast
    macro = [np.mean([v[1] for v in b.values()]) - np.mean([v[0] for v in b.values()]) for b in pairs_by_contrast.values() if b]
    return x, y, (float(np.mean(macro)) if macro else np.nan)


def _grouped(ladders, metric):
    """make_per_contrast_curves._panel_pooled_grouped's pairing (patient-merged, for the test) PLUS its delta's estimand:
    panel_pooled reads the delta off the stored series = per ladder, equal weight per OOD contrast GROUP of the case
    mean in that group; here re-derived from the same case values so it works for any pair of rungs."""
    pairs, deltas = {}, []
    for li, d in enumerate(ladders):
        keys = d["run_keys"]
        root = d["_path"].parent.parent
        extra = [{"metrics_root": Path(e["metrics_root"]), "run_subdir": e.get("run_subdir")}
                 for e in d.get("extra_ood_sources", [])]
        prev = mpc._labelled_case_values(root, d["ood_contrasts"], extra, keys[mpc.FILL - 1], metric)
        cur = mpc._labelled_case_values(root, d["ood_contrasts"], extra, keys[mpc.FILL], metric)
        gd = []
        for g, mem in d["ood_groups"].items():
            a = [v for l in mem for v in prev.get(l, {}).values() if np.isfinite(v)]
            b = [v for l in mem for v in cur.get(l, {}).values() if np.isfinite(v)]
            if a and b:
                gd.append(np.mean(b) - np.mean(a))
        deltas.append(np.mean(gd) if gd else np.nan)
        members = {l for m in d["ood_groups"].values() for l in m}
        for lbl in members:
            a, b = prev.get(lbl, {}), cur.get(lbl, {})
            for k in sorted(set(a) & set(b)):
                if np.isfinite(a[k]) and np.isfinite(b[k]):
                    pairs[f"{k}§{li}§{lbl}"] = (float(a[k]), float(b[k]))
    if not pairs:
        return np.array([]), np.array([]), np.nan
    x, y = mpc._merge_by(pairs, lambda k: mpc._patient_key(k.split("§", 1)[0]))
    return x, y, float(np.mean(deltas))


def pooled(ladders, metric, hib, pair=None):
    """(p, delta, n_units) for the task-level test on run-key pair `pair(d) -> (prev, cur)` (default: the stored 4 -> 5)."""
    ls = []
    for d in ladders:
        d2 = copy.copy(d); d2["run_keys"] = list(d["run_keys"])
        if pair is not None:
            pk, ck = pair(d)
            if pk is None or ck is None:
                return None
            d2["run_keys"][mpc.FILL - 1], d2["run_keys"][mpc.FILL] = pk, ck
        ls.append(d2)
    grouped = any(d.get("mode") == "grouped_by_contrast" for d in ls)
    scale = 100.0 if metric == "dice" else 1.0
    if grouped:
        x, y, dg = _grouped(ls, metric)
    else:
        x, y, dm = _generic(ls, metric)
        dg = dm if len(ls) == 1 else None      # multi-ladder generic panels: mean over merged patient pairs
    if not len(x):
        return float("nan"), float("nan"), 0
    delta = (dg if dg is not None else float(np.mean(y) - np.mean(x))) * scale * (1 if hib else -1)
    return mpc.wilcoxon_p(x, y), delta, len(x)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--metric", default="dice", choices=["dice", "hd95"]); a = ap.parse_args()
    metric, hib = a.metric, a.metric == "dice"
    bad = 0
    print(f"rung-4.5 RUN_IDs registered: {len(R45)}  ({', '.join(str(p) for p in R45_IDS)})")
    print(f"{'task':9s} {'fill swap 4->5':>22s}   {'noise removal 4->4.5':>22s}   {'texture 4.5->5':>22s}   units  [self-check vs panel_pooled]")
    fmt = lambda r: ("pending (rung 4)" if r is None else "no 4.5 metrics yet" if not r[2]
                     else f"{r[1]:+7.2f} (p={r[0]:.2g})").rjust(22)
    for title, _kind, entries in mpc.PANELS:
        ladders = [mpc.load(rel) for _, rel in entries]
        if any(d is None for d in ladders):
            print(f"{title:9s} ladder JSON missing"); continue
        ref_p, ref_d = mpc.panel_pooled(ladders, metric, hib)
        full = pooled(ladders, metric, hib)
        ok = np.isclose(full[0], ref_p, rtol=1e-9, atol=0) and np.isclose(full[1], ref_d, rtol=1e-6, atol=1e-9)
        bad += not ok
        nr = pooled(ladders, metric, hib, pair=lambda d: (d["run_keys"][mpc.FILL - 1], flat_key(d["run_keys"][mpc.FILL - 1])))
        tx = pooled(ladders, metric, hib, pair=lambda d: (flat_key(d["run_keys"][mpc.FILL - 1]), d["run_keys"][mpc.FILL]))
        print(f"{title:9s} {fmt(full)}   {fmt(nr)}   {fmt(tx)}   {full[2]:5d}  "
              f"[{'ok' if ok else 'MISMATCH'}: panel_pooled {ref_d:+.2f} p={ref_p:.2g}]")
        for (lab, _), d in zip(entries, ladders):
            f1 = pooled([d], metric, hib)
            n1 = pooled([d], metric, hib, pair=lambda d: (d["run_keys"][mpc.FILL - 1], flat_key(d["run_keys"][mpc.FILL - 1])))
            t1 = pooled([d], metric, hib, pair=lambda d: (flat_key(d["run_keys"][mpc.FILL - 1]), d["run_keys"][mpc.FILL]))
            print(f"  {lab:7s} {fmt(f1)}   {fmt(n1)}   {fmt(t1)}   {f1[2]:5d}")
    print("SELF-CHECK", "PASSED" if not bad else f"FAILED on {bad} panel(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
