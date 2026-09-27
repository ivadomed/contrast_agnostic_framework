#!/usr/bin/env python
"""
Per-REGION significance of the ladder's fill-swap step (rung4 Voronoi noise-fill -> rung5
v26_6_2 real-fill), BraTS2024-glioma, for every (train contrast, eval contrast, region) cell.

The ladder engine (00_commun_scripts/00_03_evaluate/ladder_ood_common.py,
_fill_swap_significance) only tests per (train, eval) direction, regions averaged per case.
This splits that test by tumor region with the same statistic (stat_tests.wilcoxon_p: two-sided
paired Wilcoxon, patient = unit, per-patient value = mean over folds 0-2) and stat_tests.holm.

Per-region population = patients whose GROUND TRUTH contains the region (voxel count > 0 in
labelsTr). Reason: eval_metrics.dice_score is NaN when GT and prediction are both empty and 0
when only one is, so for a region absent from GT a rung that hallucinates it scores 0 while a
rung that correctly predicts nothing scores NaN. Pairing region-by-region with a NaN on one side
cannot be done without inventing a value; restricting to GT-present patients makes Dice defined
in both rungs and the question clean ("given the region exists, is it segmented better").
False positives on absent regions are therefore NOT part of this table (the ladder's own
per-case mean does count them, as 0s on the hallucinating rung only).

Gate: from the same raw eval_all.csv rows, reproducing the engine's exact per-case convention
(mean over every finite fold x label value, each rung independently; Holm within the 3 OOD
contrasts) must match ladder_series.json — that validates run dirs, folds and case sets.

Holm families: the 36 cross-contrast (OOD) cells are corrected together; the 12 in-domain
cells (train == eval) are reported as a separate family so they don't dilute the OOD question.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "datasets" / "00_commun_scripts" / "00_00_utils"))
sys.path.insert(0, str(THIS_DIR))
from stat_tests import holm, wilcoxon_p  # noqa: E402
from compute_cross_contrast_ngf import (  # noqa: E402
    FOLDS, LABELS_DIR, LABEL_IDS, METRICS_ROOT, RUNG_DIRS, load_rung_eval,
)

DATA_DIR = THIS_DIR.parent / "outputs" / "data"
TRAINS = ("t1n", "t2w", "t2f")
CONTRASTS = ("t1n", "t1c", "t2w", "t2f")
REGIONS = ("NCR", "SNFH", "ET", "RC")


def gt_presence(cases) -> pd.DataFrame:
    rows = []
    for case in cases:
        arr = np.asarray(nib.load(str(LABELS_DIR / f"{case}.nii.gz")).dataobj)
        for name in REGIONS:
            rows.append(dict(case=case, region=name, gt_vox=int((arr == LABEL_IDS[name]).sum())))
    return pd.DataFrame(rows)


def load_engine_folds(train_contrast: str, rung: str) -> pd.DataFrame:
    """Every fold*/ dir, as the ladder engine's load_case_means globs it (no fold cap) — used
    ONLY by the gate. The t2w real-fill run dir is a legacy 4-fold run, so the published t2w
    ladder silently includes fold3 for that rung; the analysis itself uses FOLDS (0-2)."""
    run_dir = METRICS_ROOT / train_contrast / RUNG_DIRS[train_contrast][rung]
    frames = [pd.read_csv(f / "eval_all.csv").assign(fold=f.name)
              for f in sorted(run_dir.glob("fold*")) if (f / "eval_all.csv").exists()]
    return pd.concat(frames, ignore_index=True)


def gate() -> bool:
    ok = True
    for tr in TRAINS:
        published = json.loads((METRICS_ROOT / tr / "ablations" / "ladder_series.json").read_text()
                               )["fill_swap_significance"]["dice"]["per_contrast"]
        per_rung = {}
        for rung in ("voronoi", "realfill"):
            df = load_engine_folds(tr, rung)
            extra = sorted(set(df["fold"]) - set(FOLDS))
            if extra:
                print(f"  NOTE {tr}/{rung}: engine also reads {extra} (outside the 3-fold policy)")
            per_rung[rung] = df[np.isfinite(df["dice"])].groupby(["group", "case"])["dice"].mean()
        evals = list(published)
        p_raw = []
        for ev in evals:
            v, r = per_rung["voronoi"].loc[ev], per_rung["realfill"].loc[ev]
            common = v.index.intersection(r.index)
            p_raw.append(wilcoxon_p(r.loc[common].to_numpy(), v.loc[common].to_numpy()))
        mine = dict(zip(evals, holm(p_raw)))
        for ev in evals:
            a, b = mine[ev], published[ev]
            match = bool(np.isclose(a, b, rtol=1e-6, atol=0))
            ok &= match
            print(f"  GATE {tr}->{ev}: this script p={a:.4g}  ladder_series p={b:.4g}  "
                  f"{'MATCH' if match else 'MISMATCH'}")
    return ok


def region_table(raw: dict, presence: pd.DataFrame) -> pd.DataFrame:
    present = presence[presence["gt_vox"] > 0][["case", "region"]]
    rows = []
    for tr in TRAINS:
        means = {}
        for rung in ("voronoi", "realfill"):
            df = raw[(tr, rung)]
            df = df[df["label"].isin(REGIONS)].rename(columns={"group": "eval", "label": "region"})
            df = df.merge(present, on=["case", "region"])
            means[rung] = df.groupby(["eval", "case", "region"])["dice"].mean()
        paired = pd.concat({"voronoi": means["voronoi"], "realfill": means["realfill"]}, axis=1).dropna()
        n_dropped = len(pd.concat([means["voronoi"], means["realfill"]], axis=1)) - len(paired)
        if n_dropped:
            print(f"  NOTE train={tr}: {n_dropped} GT-present (eval,case,region) rows lacked a value "
                  f"in one rung and were dropped (expected 0)")
        for (ev, region), g in paired.reset_index().groupby(["eval", "region"]):
            d = (g["realfill"] - g["voronoi"]).to_numpy()
            rows.append(dict(train=tr, eval=ev, region=region, n=len(g),
                             mean_delta_pts=100 * d.mean(), median_delta_pts=100 * np.median(d),
                             frac_improved=float((d > 0).mean()), frac_worse=float((d < 0).mean()),
                             p_raw=wilcoxon_p(g["realfill"].to_numpy(), g["voronoi"].to_numpy())))
    out = pd.DataFrame(rows)
    out["family"] = np.where(out["train"] == out["eval"], "in-domain", "OOD")
    out["p_holm"] = np.nan
    for fam, idx in out.groupby("family").groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p_raw"].tolist())
    out["significant"] = out["p_holm"] < 0.05
    out["direction"] = np.where(~out["significant"], "n.s.",
                                 np.where(out["mean_delta_pts"] > 0, "HELPS", "HURTS"))
    return out


def main():
    raw = {(tr, rung): load_rung_eval(tr, rung) for tr in TRAINS for rung in ("voronoi", "realfill")}
    for (tr, rung), df in raw.items():
        assert set(df["fold"]) <= set(FOLDS), (tr, rung, set(df["fold"]))

    print("Gate: reproduce ladder engine per-direction fill-swap p-values (engine's own convention)")
    gate_ok = gate()
    print(f"  GATE {'PASSED' if gate_ok else 'FAILED'}")
    if not gate_ok:
        sys.exit("Gate failed — refusing to report per-region results built on unvalidated inputs")

    cases = sorted(set().union(*[set(df["case"]) for df in raw.values()]))
    presence = gt_presence(cases)
    presence.to_csv(DATA_DIR / "gt_region_presence.csv", index=False)
    print(f"\nGT region presence over {len(cases)} patients:")
    print((presence.assign(present=presence["gt_vox"] > 0).groupby("region")["present"].sum()).to_string())

    out = region_table(raw, presence)
    out_csv = DATA_DIR / "region_fill_swap_significance.csv"
    out.to_csv(out_csv, index=False)

    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:.3g}"  # noqa: E731
    cols = ["train", "eval", "region", "n", "mean_delta_pts", "median_delta_pts",
            "frac_improved", "frac_worse", "p_raw", "p_holm", "direction"]
    for fam in ("OOD", "in-domain"):
        sub = out[out["family"] == fam].sort_values(["train", "eval", "region"])
        print(f"\n=== {fam} cells (Holm within this family, {len(sub)} cells) ===")
        print(sub[cols].to_string(index=False, float_format=fmt))
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
