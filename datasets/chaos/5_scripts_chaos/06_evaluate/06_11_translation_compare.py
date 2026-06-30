#!/usr/bin/env python3
"""
Translation-robustness comparison for the exp_translation_NNN experiment.

Compares each chaos model's in-domain Dice ORIGINAL vs TRANSLATED, the right way:
matched on the (case,label) pairs whose TRANSLATED GT is non-empty.

WHY MATCHED / GT-PRESENT-ONLY (important):
A 50% shift CROPS organs out of the GT. The shared evaluator scores
"GT empty but prediction non-empty" as Dice 0.0 — correct for normal eval (a false
positive / hallucinated organ), but WRONG here: "GT empty" means "cropped out of
frame", not "absent". Counting those 0.0s makes every model look like it ~halves
(−30..−45), which is a metric artifact, not a translation failure. Scoring only the
organs actually present in the (cropped) translated GT — and comparing the ORIGINAL
on the same (case,label) set — isolates genuine translation-equivariance.

Reads predictions + nnUNet GT directly (the per-fold eval_all.csv cannot distinguish
the two kinds of 0.0). Writes a Markdown table to the experiment metrics dir.

Usage:
  python 06_11_translation_compare.py [--exp 050]
"""
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
PRED = DATASET_ROOT / "8_results_chaos" / "01_predictions" / "chaos_model"
RAW  = DATASET_ROOT / "2_nnUNet_chaos" / "raw"
OUT  = DATASET_ROOT / "8_results_chaos" / "02_metrics" / "chaos_model"
LABELS = [1, 2, 3, 4]   # liver, right_kidney, left_kidney, spleen (MR scoreable)

RUNS = {
    "t1in": ("Dataset060_CHAOS_MR_T1in", [
        ("nnUNet", "baseline_20260614_153230"),
        ("nnUNet", "v26_6_2_train050_val100_20260615_213615"),
        ("auglab", "synthseg_EM_train100_val000_20260611_120000"),
        ("auglab", "synthseg_noEM_train100_val000_20260611_120000"),
        ("auglab", "auglab_default_20260611_120000"),
        ("auglab", "auglabAug_v26_6_2_train025_val100_20260616_200514")]),
    "t2spir": ("Dataset061_CHAOS_MR_T2spir", [
        ("nnUNet", "baseline_20260620_111146"),
        ("nnUNet", "v26_6_2_train050_val100_20260620_112122"),
        ("auglab", "synthseg_EM_20260620_112357"),
        ("auglab", "synthseg_noEM_20260620_112515"),
        ("auglab", "auglab_default_20260620_112240"),
        ("auglab", "auglabAug_v26_6_2_train025_val100_20260620_112633")]),
}


def dice(pb, gb):
    p, g = pb.sum(), gb.sum()
    if p == 0 and g == 0:
        return None
    if p == 0 or g == 0:
        return 0.0
    return 2 * np.logical_and(pb, gb).sum() / (p + g)


def load(ds, sub):
    d = RAW / ds / sub
    return {f.name: np.asanyarray(nib.load(f).dataobj) for f in d.glob("*.nii.gz")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="050", help="translation tag NNN")
    args = ap.parse_args()
    exp = f"exp_translation_{args.exp}"
    lines = [f"# CHAOS translation robustness — {args.exp[:-1] if args.exp.endswith('0') else args.exp}% LEFT shift",
             "",
             "In-domain mean Dice, **matched on (case,label) pairs present in the translated GT** "
             "(organs cropped off-frame are excluded from BOTH original and translated — see header of "
             "06_11_translation_compare.py for why a naive average massively overstates the drop).", ""]
    for contrast, (ds, methlist) in RUNS.items():
        gO = load(ds, f"labelsTs_{contrast}")
        gT = load(ds, f"labelsTs_{contrast}_{exp}")
        lines += [f"## {contrast} models (in-domain {contrast})", "",
                  "| method | original | translated | Δ |", "|---|---|---|---|"]
        for cat, suf in methlist:
            run = f"chaos_{contrast}_{suf}"
            o, t = [], []
            for fold in range(4):
                pO = PRED / contrast / cat / run / f"fold{fold}/{contrast}"
                pT = PRED / contrast / cat / run / f"fold{fold}/{exp}/{contrast}"
                for pf in pT.glob("*.nii.gz"):
                    nm = pf.name
                    if nm not in gO or nm not in gT or not (pO / nm).exists():
                        continue
                    aO = np.asanyarray(nib.load(pO / nm).dataobj)
                    aT = np.asanyarray(nib.load(pf).dataobj)
                    for lab in LABELS:
                        if (gT[nm] == lab).sum() == 0:      # cropped out → skip both
                            continue
                        dO = dice(aO == lab, gO[nm] == lab)
                        dT = dice(aT == lab, gT[nm] == lab)
                        if dO is not None: o.append(dO)
                        if dT is not None: t.append(dT)
            mo, mt = np.mean(o) * 100, np.mean(t) * 100
            lines.append(f"| {suf.split('_2026')[0]} | {mo:.1f} | {mt:.1f} | {mt-mo:+.1f} |")
        lines.append("")
    out = OUT / f"{exp}_comparison.md"
    out.write_text("\n".join(lines))
    print(f"wrote → {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
