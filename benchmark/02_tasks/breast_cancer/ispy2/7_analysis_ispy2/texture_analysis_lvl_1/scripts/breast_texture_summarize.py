#!/usr/bin/env python
"""
Reduce step for breast_texture_analysis.py's sharded output: merges
outputs/data/{delta,vis,missing}_shard*.csv into the fill-swap significance table,
error-signature breakdown, per-contrast visibility table, and the Dice-vs-visibility
comparison + one plot -- same content/format as BraTS FINDINGS.md sections 1/3/4.

Run AFTER all worker shards finish (submit with RUN_JOB_DEPENDENCY=afterok:<job ids>).
Usage: .venv/bin/python breast_texture_summarize.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(PROJECT_ROOT / "datasets" / "00_commun_scripts" / "00_00_utils"))
from stat_tests import wilcoxon_p  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"
DATA_DIR, TABLES_DIR, PLOTS_DIR = OUT / "data", OUT / "tables", OUT / "plots"


def load_shards(pattern):
    files = sorted(DATA_DIR.glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_shards("delta_shard*.csv")
    vis = load_shards("vis_shard*.csv")
    missing = []
    for f in sorted(DATA_DIR.glob("missing_shard*.txt")):
        missing += [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]

    df.to_csv(DATA_DIR / "fill_swap_all.csv", index=False)
    vis.to_csv(DATA_DIR / "visibility_all.csv", index=False)

    lines = ["# Breast texture analysis (ispy2/duke-breast-mri) -- port of BraTS lvl-1 texture ladder mechanism", "",
             "Reused: stat_tests.wilcoxon_p. step_d/step_auc/ramp_R re-implemented (identical formulas) from "
             "BraTS compute_step_affine.py / internal_ramp_vs_fill_swap.py. Single label (tumour). Per-image "
             "only -- breast contrasts not verified co-registered, so no NGF / cross-contrast voxel measures. "
             f"Subsampled to <=40 cases per row (seed 0). {len(df)} delta rows, {len(vis)} visibility rows.", ""]
    if missing:
        lines += ["## Data-availability notes", ""] + [f"- {m}" for m in missing[:60]]
        if len(missing) > 60:
            lines.append(f"- ... ({len(missing) - 60} more, see outputs/data/missing_shard*.txt)")
        lines.append("")

    summary_rows = []
    if not df.empty:
        per_case = df.groupby(["train", "eval_source", "eval_item", "case", "rung",
                                "in_domain", "same_contrast_cross_dataset"]).mean(numeric_only=True).reset_index()
        metric_cols = ["dice", "recall", "precision", "vol_ratio", "fp_share_ring"]
        lines += ["## 1+2. Fill-swap Delta Dice + error signature (patient unit, real vs noise, mean over folds 0-2)", "",
                   "| train | eval | n | dice noise→real (Δ) | p | recall Δ | precision Δ | vol_ratio Δ | fp_share_ring Δ | domain |",
                   "|---|---|--:|---|---|---|---|---|---|---|"]
        for (tr, es, ei), g in per_case.groupby(["train", "eval_source", "eval_item"]):
            w = g.pivot(index="case", columns="rung", values=metric_cols)
            if ("dice", "noise") not in w.columns or ("dice", "real") not in w.columns:
                continue
            noise_d, real_d = w[("dice", "noise")], w[("dice", "real")]
            ok = noise_d.notna() & real_d.notna()
            n = int(ok.sum())
            if n == 0:
                continue
            p_val = wilcoxon_p(real_d[ok].to_numpy(), noise_d[ok].to_numpy())
            domain = ("in-domain" if g["in_domain"].iloc[0]
                      else "cross-dataset-only (same contrast)" if g["same_contrast_cross_dataset"].iloc[0]
                      else "OOD")
            deltas = {}
            for m in metric_cols:
                a, b = w[(m, "real")], w[(m, "noise")]
                ok2 = a.notna() & b.notna()
                deltas[m] = float((a[ok2] - b[ok2]).median()) if ok2.any() else float("nan")
            lines.append(f"| {tr} | {es}/{ei} | {n} | {noise_d[ok].median():.3f}→{real_d[ok].median():.3f} "
                          f"({deltas['dice']:+.3f}) | {p_val:.3g} | {deltas['recall']:+.3f} | "
                          f"{deltas['precision']:+.3f} | {deltas['vol_ratio']:+.3f} | "
                          f"{deltas['fp_share_ring']:+.3f} | {domain} |")
            summary_rows.append(dict(train=tr, eval_source=es, eval_item=ei, n=n,
                                       dice_noise=float(noise_d[ok].median()), dice_real=float(real_d[ok].median()),
                                       delta_dice=deltas["dice"], p=p_val, domain=domain,
                                       delta_recall=deltas["recall"], delta_precision=deltas["precision"],
                                       delta_vol_ratio=deltas["vol_ratio"], delta_fp_share_ring=deltas["fp_share_ring"]))
        lines.append("")
        lines.append("Under-call = recall Δ<0 with precision flat/held (misses -> background); over-call = "
                      "precision Δ<0 with vol_ratio Δ>0 and/or fp_share_ring Δ>0 (false positives spill into "
                      "the surrounding ring). Compare each row's sign/shape to FINDINGS.md §3/§7's BraTS pattern "
                      "(every BraTS failure was an under-call, never a ring spillover).")
        pd.DataFrame(summary_rows).to_csv(DATA_DIR / "fill_swap_summary.csv", index=False)

    if not vis.empty:
        vc = vis.groupby(["eval_source", "eval_item"])[["step_d", "step_auc", "ramp_R"]].mean()
        counts = vis.groupby(["eval_source", "eval_item"]).size()
        lines += ["", "## 3. Visibility per eval contrast (region-vs-ring step; internal ramp R)", "",
                   "| eval contrast | n cases | step_d | step_auc | ramp_R |", "|---|--:|--:|--:|--:|"]
        for (es, ei), r in vc.iterrows():
            lines.append(f"| {es}/{ei} | {int(counts[(es, ei)])} | {r['step_d']:.3f} | {r['step_auc']:.3f} | {r['ramp_R']:.3f} |")
        vc.to_csv(DATA_DIR / "visibility_summary.csv")

        lines += ["", "## 4. Does noise-fill / real-fill Dice track visibility across eval contrasts?", ""]
        if summary_rows:
            sdf = pd.DataFrame(summary_rows)
            vc_flat = vc.reset_index()
            sdf = sdf.merge(vc_flat, on=["eval_source", "eval_item"], how="left")
            sdf.to_csv(DATA_DIR / "dice_vs_visibility.csv", index=False)
            lines += ["| train | eval | domain | step_auc | ramp_R | dice noise | dice real |",
                      "|---|---|---|--:|--:|--:|--:|"]
            for r in sdf.itertuples():
                lines.append(f"| {r.train} | {r.eval_source}/{r.eval_item} | {r.domain} | {r.step_auc:.3f} | "
                              f"{r.ramp_R:.3f} | {r.dice_noise:.3f} | {r.dice_real:.3f} |")
            for tr, g in sdf[sdf["domain"] != "in-domain"].groupby("train"):
                g = g.sort_values("step_auc")
                mono_noise = g["dice_noise"].is_monotonic_increasing or g["dice_noise"].is_monotonic_decreasing
                mono_real = g["dice_real"].is_monotonic_increasing or g["dice_real"].is_monotonic_decreasing
                lines.append(f"\n- train={tr}: sorted by step_auc, noise-fill Dice monotone: {mono_noise}; "
                              f"real-fill Dice monotone: {mono_real} (only {len(g)} OOD items -- too few for a "
                              "within-row rank test like BraTS's Kendall tau; descriptive only).")
        else:
            lines.append("(no fill-swap summary rows available)")

    (TABLES_DIR / "breast_texture_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))

    if summary_rows and not vis.empty:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sdf = pd.DataFrame(summary_rows)
        vc_flat = vis.groupby(["eval_source", "eval_item"])[["step_auc"]].mean().reset_index()
        sdf = sdf.merge(vc_flat, on=["eval_source", "eval_item"], how="left")
        fig, ax = plt.subplots(figsize=(6.5, 5))
        colors = {"t1wce": "#c0392b", "t2w": "#2f7d6b"}
        markers = {"OOD": "o", "in-domain": "s", "cross-dataset-only (same contrast)": "^"}
        for r in sdf.itertuples():
            ax.scatter(r.step_auc, 100 * r.delta_dice, color=colors.get(r.train, "#555"),
                       marker=markers.get(r.domain, "x"), s=90, edgecolors="white", zorder=3)
            ax.annotate(f"{r.train}->{r.eval_source.split('-')[0]}/{r.eval_item}",
                        (r.step_auc, 100 * r.delta_dice), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.axhline(0, color="#777", ls="--", lw=1)
        ax.set_xlabel("region-vs-ring visibility (step_auc)")
        ax.set_ylabel("real-fill gain Δ Dice (pts, real - noise)")
        ax.set_title("Breast: does real-fill's cross-contrast gain track tumour visibility?")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        handles = [plt.Line2D([], [], marker="o", color=c, ls="", label=t) for t, c in colors.items()] + \
                  [plt.Line2D([], [], marker=m, color="#555", ls="", label=k) for k, m in markers.items()]
        ax.legend(handles=handles, fontsize=7, loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "delta_dice_vs_visibility.png", dpi=150, bbox_inches="tight")

    print("DONE")


if __name__ == "__main__":
    main()
