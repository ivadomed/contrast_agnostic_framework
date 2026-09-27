#!/usr/bin/env python
"""
Summarizes outputs/data/anisotropy_per_patient.csv (see compute_anisotropy.py) and links it
to the fill-swap ablation outcome (region_fill_swap_significance.csv / ngf_vs_ladder_per_patient.csv)
to test the hypothesis that t1n's real-fill asymmetry is explained by t1n itself having low
fine-texture energy / high anisotropy (thick-slice-like), not by cross-contrast dissimilarity.

Part 1: per contrast x region, median [IQR] anisotropy and E_mean, frac(anisotropy<0.7), most
common argmin_axis. Key contrast: t1n vs t1c (and t2w, t2f) in brain and SNFH.

Part 2: pooled link to the ablation. For every OOD cell in region_fill_swap_significance.csv,
attach eval-contrast and train-contrast median anisotropy/E_mean (that region, across patients)
and eval-train differences. Writes outputs/tables/anisotropy_vs_fill_swap.md (SNFH first) +
outputs/data/anisotropy_vs_fill_swap.csv.

Part 3: per-patient, for 5 requested cells, Spearman of eval-contrast BRAIN-level anisotropy
(and separately E_mean) vs that patient's delta_dice (from ngf_vs_ladder_per_patient.csv, same
patient population already used for the NGF correlation — assert n matches the sig CSV's n for
those cells). 5 cells x 2 metrics = 10 tests, Holm-corrected together.

Part 4: plots — anisotropy_by_contrast.png, anisotropy_vs_delta.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.resolve().parents[6]
OUT = THIS_DIR.parent / "outputs"
DATA, TABLES, PLOTS = OUT / "data", OUT / "tables", OUT / "plots"

sys.path.insert(0, str(PROJECT_ROOT / "benchmark" / "00_commun_scripts" / "00_00_utils"))
from stat_tests import holm  # noqa: E402

CONTRASTS = ("t1n", "t1c", "t2w", "t2f")
TRAINS = ("t1n", "t2w", "t2f")
REGIONS = ("SNFH", "RC", "ET", "NCR")  # tumor regions, matches sig csv
ANIS_THRESH = 0.7

# Requested per-patient cells: (train, eval, region)
PER_PATIENT_CELLS = [
    ("t2w", "t1n", "SNFH"),
    ("t2f", "t1n", "SNFH"),
    ("t2w", "t1n", "RC"),
    ("t2w", "t1c", "SNFH"),
    ("t1n", "t2w", "SNFH"),
]


def load_anisotropy() -> pd.DataFrame:
    f = DATA / "anisotropy_per_patient.csv"
    if not f.exists():
        sys.exit(f"{f} not found — run compute_anisotropy.py (shards + --merge) first")
    return pd.read_csv(f)


def iqr_str(x: pd.Series) -> str:
    x = x.dropna()
    if len(x) == 0:
        return "n/a"
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return f"{med:.3f} [{q1:.3f}, {q3:.3f}]"


def mode_axis(x: pd.Series):
    x = x.dropna()
    if len(x) == 0:
        return "n/a"
    return int(x.mode().iloc[0])


# ───────────────────────── Part 1 ─────────────────────────
def part1(df: pd.DataFrame):
    lines = ["# Anisotropy summary by contrast x region (BraTS2024-glioma)", "",
             "median [IQR] anisotropy (min(E)/max(E) across the 3 array axes; 1=isotropic fine "
             f"texture, ->0=one axis much smoother) and E_mean (mean squared first-difference "
             f"energy, z-scored intensity units); frac(anisotropy<{ANIS_THRESH}); most common "
             "argmin axis (array axis index, see axcodes printed by --axcodes-only).", ""]
    key_rows = []
    for region in ("brain", "healthy", "SNFH", "RC", "ET", "NCR"):
        sub_r = df[df["region"] == region]
        if len(sub_r) == 0:
            continue
        lines += [f"## {region}", "",
                  "| contrast | n | anisotropy med [IQR] | frac<0.7 | E_mean med [IQR] | mode argmin_axis |",
                  "|---|--:|---|--:|---|--:|"]
        for c in CONTRASTS:
            s = sub_r[sub_r["contrast"] == c]
            if len(s) == 0:
                continue
            frac_low = float((s["anisotropy"] < ANIS_THRESH).mean())
            lines.append(f"| {c} | {len(s)} | {iqr_str(s['anisotropy'])} | {frac_low:.2f} | "
                         f"{iqr_str(s['E_mean'])} | {mode_axis(s['argmin_axis'])} |")
            if region in ("brain", "SNFH"):
                key_rows.append(dict(region=region, contrast=c, n=len(s),
                                      anisotropy_median=s["anisotropy"].median(),
                                      E_mean_median=s["E_mean"].median(),
                                      frac_low=frac_low))
        lines.append("")
    (TABLES / "anisotropy_by_contrast.md").write_text("\n".join(lines))
    print(f"Wrote {TABLES / 'anisotropy_by_contrast.md'}")

    key_df = pd.DataFrame(key_rows)
    key_df.to_csv(DATA / "anisotropy_key_contrasts.csv", index=False)
    print("\n=== Key comparison: t1n vs t1c (and t2w, t2f) in brain / SNFH ===")
    pd.set_option("display.width", 200)
    print(key_df.pivot(index="contrast", columns="region",
                        values=["anisotropy_median", "E_mean_median"]).to_string(float_format=lambda v: f"{v:.4f}"))
    return key_df


# ───────────────────────── Part 2 ─────────────────────────
def part2(df: pd.DataFrame):
    sig = pd.read_csv(DATA / "region_fill_swap_significance.csv")
    sig = sig[sig["family"] == "OOD"].copy()

    med = df.groupby(["contrast", "region"]).agg(
        anisotropy_median=("anisotropy", "median"), E_mean_median=("E_mean", "median")
    ).reset_index()

    def lookup(contrast, region, col):
        s = med[(med["contrast"] == contrast) & (med["region"] == region)]
        return float(s[col].iloc[0]) if len(s) else float("nan")

    rows = []
    for _, r in sig.iterrows():
        tr, ev, region = r["train"], r["eval"], r["region"]
        anis_eval = lookup(ev, region, "anisotropy_median")
        anis_train = lookup(tr, region, "anisotropy_median")
        e_eval = lookup(ev, region, "E_mean_median")
        e_train = lookup(tr, region, "E_mean_median")
        rows.append(dict(train=tr, eval=ev, region=region, n=r["n"],
                          mean_delta_pts=r["mean_delta_pts"], p_holm=r["p_holm"],
                          direction=r["direction"],
                          anisotropy_eval=anis_eval, anisotropy_train=anis_train,
                          anisotropy_eval_minus_train=anis_eval - anis_train,
                          E_mean_eval=e_eval, E_mean_train=e_train,
                          E_mean_eval_minus_train=e_eval - e_train))
    out = pd.DataFrame(rows)
    out["region"] = pd.Categorical(out["region"], ["SNFH", "RC", "ET", "NCR"], ordered=True)
    out = out.sort_values(["region", "train", "eval"]).reset_index(drop=True)
    out.to_csv(DATA / "anisotropy_vs_fill_swap.csv", index=False)

    lines = ["# Anisotropy vs. fill-swap ablation outcome (BraTS2024-glioma, OOD cells)", "",
             "Per (train->eval, region) OOD cell: rung4->5 (noise-fill -> real-fill) Delta Dice "
             "points from region_fill_swap_significance.csv, alongside median anisotropy/E_mean "
             "of the eval and train contrast in that region (this patient population, not "
             "case-matched to the ablation's own patient set). eval_minus_train < 0 means the "
             "eval contrast has LESS fine-texture-energy / MORE anisotropy than the train "
             "contrast.", ""]
    for region in ("SNFH", "RC", "ET", "NCR"):
        sub = out[out["region"] == region]
        if len(sub) == 0:
            continue
        lines += [f"## {region}", "",
                  "| train->eval | Delta Dice | p_holm | outcome | anis(eval) | anis(train) | anis(eval-train) | E_mean(eval) | E_mean(train) | E_mean(eval-train) |",
                  "|---|--:|--:|---|--:|--:|--:|--:|--:|--:|"]
        for _, r in sub.iterrows():
            lines.append(f"| {r['train']}->{r['eval']} | {r['mean_delta_pts']:+.2f} | {r['p_holm']:.2g} | "
                         f"{r['direction']} | {r['anisotropy_eval']:.3f} | {r['anisotropy_train']:.3f} | "
                         f"{r['anisotropy_eval_minus_train']:+.3f} | {r['E_mean_eval']:.3f} | "
                         f"{r['E_mean_train']:.3f} | {r['E_mean_eval_minus_train']:+.3f} |")
        lines.append("")
    (TABLES / "anisotropy_vs_fill_swap.md").write_text("\n".join(lines))
    print(f"\nWrote {TABLES / 'anisotropy_vs_fill_swap.md'}")
    print(out[out["region"] == "SNFH"].to_string(index=False))
    return out


# ───────────────────────── Part 3 ─────────────────────────
def part3(df: pd.DataFrame):
    ladder_pp = pd.read_csv(DATA / "ngf_vs_ladder_per_patient.csv")
    sig = pd.read_csv(DATA / "region_fill_swap_significance.csv")
    sig = sig[sig["family"] == "OOD"].set_index(["train", "eval", "region"])

    # brain-level anisotropy/E_mean per patient x contrast (eval-side metric)
    brain = df[df["region"] == "brain"].set_index(["patient", "contrast"])[["anisotropy", "E_mean"]]

    rows, p_raw_list = [], []
    for train, ev, region in PER_PATIENT_CELLS:
        cell = ladder_pp[(ladder_pp["train"] == train) & (ladder_pp["eval"] == ev)
                          & (ladder_pp["region"] == region)].copy()
        expected_n = int(sig.loc[(train, ev, region), "n"]) if (train, ev, region) in sig.index else None
        merged = cell
        # attach eval-contrast brain anisotropy/E_mean per patient
        merged["anisotropy_eval"] = [
            brain.loc[(c, ev), "anisotropy"] if (c, ev) in brain.index else np.nan for c in merged["case"]
        ]
        merged["E_mean_eval"] = [
            brain.loc[(c, ev), "E_mean"] if (c, ev) in brain.index else np.nan for c in merged["case"]
        ]
        merged = merged.dropna(subset=["anisotropy_eval", "E_mean_eval", "delta_dice"])
        n = len(merged)
        if expected_n is not None and n != expected_n:
            print(f"WARNING: cell ({train},{ev},{region}) n={n} but sig csv n={expected_n} "
                  f"— patient population mismatch, treat with caution")
        for metric_col, metric_name in (("anisotropy_eval", "anisotropy"), ("E_mean_eval", "E_mean")):
            if n >= 8:
                rho, p = spearmanr(merged[metric_col], merged["delta_dice"])
            else:
                rho, p = float("nan"), float("nan")
            rows.append(dict(train=train, eval=ev, region=region, metric=metric_name, n=n,
                              spearman_rho=rho, p_raw=p))
            p_raw_list.append(p)

    out = pd.DataFrame(rows)
    out["p_holm"] = holm(list(out["p_raw"]))
    out.to_csv(DATA / "anisotropy_per_patient_spearman.csv", index=False)
    print("\n=== Per-patient Spearman: eval-contrast brain anisotropy/E_mean vs delta_dice (real-fill - noise-fill) ===")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return out


# ───────────────────────── Part 4: plots ─────────────────────────
def plot_by_contrast(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    brain = df[df["region"] == "brain"]
    data = [brain[brain["contrast"] == c]["anisotropy"].dropna().values for c in CONTRASTS]
    bp = ax.boxplot(data, labels=CONTRASTS, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfe3f7")
    for i, d in enumerate(data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.06, size=len(d))
        ax.scatter(np.full(len(d), i) + jitter, d, s=8, alpha=0.35, color="#2f5f8a", zorder=3)
    ax.axhline(ANIS_THRESH, color="#c0392b", linestyle="--", linewidth=1, label=f"threshold {ANIS_THRESH}")
    ax.set_ylabel("brain-level anisotropy (min(E)/max(E))")
    ax.set_title("Brain-level anisotropy per contrast")
    ax.legend(fontsize=8)
    ax.grid(axis="y", color="#eee")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax = axes[1]
    width = 0.35
    x = np.arange(len(CONTRASTS))
    for offset, region, color in ((-width / 2, "SNFH", "#c0392b"), (width / 2, "healthy", "#2f7d6b")):
        meds = [df[(df["contrast"] == c) & (df["region"] == region)]["E_mean"].median() for c in CONTRASTS]
        iqrs = [np.percentile(df[(df["contrast"] == c) & (df["region"] == region)]["E_mean"].dropna(), [25, 75])
                if len(df[(df["contrast"] == c) & (df["region"] == region)]) else [np.nan, np.nan]
                for c in CONTRASTS]
        lo = [m - iqr[0] for m, iqr in zip(meds, iqrs)]
        hi = [iqr[1] - m for m, iqr in zip(meds, iqrs)]
        ax.bar(x + offset, meds, width=width, yerr=[lo, hi], capsize=3, color=color, alpha=0.85,
               label=region)
    ax.set_xticks(x)
    ax.set_xticklabels(CONTRASTS)
    ax.set_ylabel("E_mean (fine-texture energy, z-scored units)")
    ax.set_title("E_mean per contrast: SNFH vs healthy")
    ax.legend(fontsize=8)
    ax.grid(axis="y", color="#eee")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle("Fine-texture anisotropy / energy by contrast (BraTS2024-glioma, n=70 patients)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_vs_delta(spearman_df: pd.DataFrame, ladder_pp: pd.DataFrame, brain: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 5, figsize=(22, 8.5))
    for col_i, (train, ev, region) in enumerate(PER_PATIENT_CELLS):
        cell = ladder_pp[(ladder_pp["train"] == train) & (ladder_pp["eval"] == ev)
                          & (ladder_pp["region"] == region)].copy()
        cell["anisotropy_eval"] = [
            brain.loc[(c, ev), "anisotropy"] if (c, ev) in brain.index else np.nan for c in cell["case"]
        ]
        cell["E_mean_eval"] = [
            brain.loc[(c, ev), "E_mean"] if (c, ev) in brain.index else np.nan for c in cell["case"]
        ]
        for row_i, (metric_col, metric_label) in enumerate((("anisotropy_eval", "anisotropy"),
                                                              ("E_mean_eval", "E_mean"))):
            ax = axes[row_i, col_i]
            sub = cell.dropna(subset=[metric_col, "delta_dice"])
            x, y = sub[metric_col].values, sub["delta_dice"].values
            ax.scatter(x, y, s=18, alpha=0.6, color="#2f5f8a")
            if len(x) >= 2 and np.std(x) > 0:
                b, a = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, a + b * xs, color="#c0392b", linewidth=1.5)
            info = spearman_df[(spearman_df["train"] == train) & (spearman_df["eval"] == ev)
                                & (spearman_df["region"] == region) & (spearman_df["metric"] == metric_label)]
            if len(info):
                rho, p, p_h, n = (info["spearman_rho"].iloc[0], info["p_raw"].iloc[0],
                                   info["p_holm"].iloc[0], info["n"].iloc[0])
                ax.set_title(f"{train}->{ev} {region}\nn={n} rho={rho:.2f} p={p:.3f} (Holm {p_h:.3f})",
                             fontsize=8.5)
            ax.set_xlabel(f"eval-contrast brain {metric_label}", fontsize=8)
            if col_i == 0:
                ax.set_ylabel("delta_dice (real-fill - noise-fill)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(color="#eee")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.suptitle("Per-patient: eval-contrast fine-texture metric vs. real-fill Dice delta", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    for d in (DATA, TABLES, PLOTS):
        d.mkdir(parents=True, exist_ok=True)
    df = load_anisotropy()
    print(f"Loaded {len(df)} anisotropy rows, {df['patient'].nunique()} patients")

    part1(df)
    part2(df)
    spearman_df = part3(df)

    plot_by_contrast(df, PLOTS / "anisotropy_by_contrast.png")
    brain = df[df["region"] == "brain"].set_index(["patient", "contrast"])[["anisotropy", "E_mean"]]
    ladder_pp = pd.read_csv(DATA / "ngf_vs_ladder_per_patient.csv")
    plot_vs_delta(spearman_df, ladder_pp, brain, PLOTS / "anisotropy_vs_delta.png")


if __name__ == "__main__":
    main()
