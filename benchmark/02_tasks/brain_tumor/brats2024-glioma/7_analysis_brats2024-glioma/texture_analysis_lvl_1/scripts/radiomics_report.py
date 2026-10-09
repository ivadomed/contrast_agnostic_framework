#!/usr/bin/env python
"""Compose outputs/tables/radiomics_fillswap.md and outputs/plots/radiomics_fillswap.png from A/B pickles."""
from __future__ import annotations

import glob
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from radiomics_common import DATA, PLOTS, TABLES  # noqa: E402

SCH = ["LOCO(36 cells)", "LO-train-contrast(3)", "LO-EVAL-contrast(4)"]
SETS = ["hand_raw", "hand_deid", "rad_tex_raw", "rad_tex_deid", "rad_ts_raw", "rad_ts_deid"]


def ver():
    d = glob.glob(str(Path(sys.executable).resolve().parents[1] / "lib/python3.*/site-packages/pyradiomics*"))
    return "3.1.0 (wheelhouse)"


def main():
    L = ["# IBSI radiomics vs hand-made features for the fill-swap effect", "",
         "PyRadiomics 3.1.0 (installed per job in $SLURM_TMPDIR), z-scored within brain/foreground, binCount=32, Original + LoG(sigma 1,2,3 mm) + Wavelet, "
         "firstorder/glcm/glrlm/glszm/gldm/ngtdm (93 x 12 = 1116 per mask); masks = target region eroded 1 vox (R) and 1-5 vox ring; "
         "row features R, ring, R-ring; shape (R) kept as a separate set. BraTS 1 mm iso native; other datasets resampled to 1 mm iso "
         "(linear image / NN mask, native spacing in the shard CSVs).", ""]
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    # ---------------- A
    if (DATA / "radiomics_A_meta.pkl").exists():
        meta = pickle.load(open(DATA / "radiomics_A_meta.pkl", "rb"))
        S = pd.read_csv(DATA / "radiomics_A_summary.csv")
        B = pd.read_csv(DATA / "radiomics_A_best.csv")
        G = pd.read_csv(DATA / "radiomics_A_importance_groups.csv")
        I = pd.read_csv(DATA / "radiomics_A_importance_features.csv")
        nm = pickle.load(open(DATA / "radiomics_A_nullmax.pkl", "rb"))
        info = meta["info"]
        L += ["## A. Within-BraTS (bigfeat_v2 design)", "",
              f"{meta['nrows']} rows (patient x OOD cell), {meta['npat']} patients (all available; no subsampling), same rows/splits/permutation seeds for "
              f"hand-made and radiomics sets. Label-free |Spearman|>0.95 pruning of base features: tex {info['rad_tex'][0]} -> {info['rad_tex'][1]}, "
              f"tex+shape {info['rad_ts'][0]} -> {info['rad_ts'][1]}; features per set = [E, T, E-T]. Top-30 |corr| selection + mean-imputation + scaling inside folds; "
              f"{meta['N']} cell-level permutations of the whole pipeline. Note: radiomics sets contain NO whole-brain features, "
              f"so rad_*_raw is already closer to the hand-made de-identified set than hand_raw is.", "",
              "Best-of-3-models (HGB/RF/ENet) cell Spearman; p vs null of the same max over models:", "",
              "| set | " + " | ".join(SCH) + " |", "|---|" + "--:|" * len(SCH)]
        for fs in SETS:
            row = []
            for sn in SCH:
                r = B[(B.set == fs) & (B.scheme == sn)].iloc[0]
                row.append(f"{r.max_rho:+.2f} (p={r.p:.3f})")
            L.append(f"| {fs} | " + " | ".join(row) + " |")
        ref = S[(S.scheme == "GroupKFold-patient(5) [ref]")].groupby("set").rho.max()
        L += ["", "Reference (patient GroupKFold, observed only, best model): " + ", ".join(f"{k} {v:+.2f}" for k, v in ref.items()), "",
              "Sign accuracy / failures caught (LOEO, per model, 14 significant cells; trivial always-helps = 0.71, 0/4):", ""]
        e = S[S.scheme == "LO-EVAL-contrast(4)"]
        for r in e.itertuples():
            L.append(f"- {r.set}/{r.model}: cell rho {r.rho:+.2f} (p={r.p_rho:.3f}), sign-acc {r.acc:.2f} (p={r.p_acc:.3f}), failures {r.fc}/4 (p={r.p_fc:.3f})")
        fs, mn, surv = meta["interp"]
        L += ["", f"### Importance: {fs}/{mn} ({'survives' if surv else 'NO radiomics model survives'} LOEO p<0.05; model chosen by best LOEO rho). "
              "Grouped permutation importance on held-out leave-eval-contrast-out folds (MSE increase):", ""]
        for t in ("part", "itype", "cls", "src"):
            g = G[G.type == t].sort_values("mse_inc", ascending=False)
            L.append(f"- {t}: " + ", ".join(f"{r.group} {r.mse_inc:+.2f}" for r in g.itertuples()))
        L += ["", "Top features: " + "; ".join(f"{r.feature} ({r.mse_inc:+.2f}, {r.folds}/4)" for r in I.head(8).itertuples()), ""]
        w = 0.13
        for j, fs_ in enumerate(SETS):
            xs = np.arange(len(SCH)) + (j - 2.5) * w
            ob = [nm[(fs_, s)][0] for s in SCH]
            ax[0].bar(xs, ob, w, label=fs_, color=plt.cm.tab10(j if j < 2 else j + 1))
            ax[0].scatter(xs, [np.percentile(nm[(fs_, s)][1], 95) for s in SCH], marker="_", s=120, c="k", zorder=3)
        ax[0].set_xticks(range(len(SCH)))
        ax[0].set_xticklabels(SCH, fontsize=7)
        ax[0].set_ylabel("best-of-3-models cell Spearman")
        ax[0].set_title("A. within-BraTS: observed (bars) vs permutation 95th pct (black ticks)")
        ax[0].axhline(0, c="grey", lw=.5)
        ax[0].legend(fontsize=6, ncol=2)
    # ---------------- B
    if (DATA / "radiomics_B_fit.pkl").exists():
        F = pickle.load(open(DATA / "radiomics_B_fit.pkl", "rb"))
        h = F["hand"]
        L += ["## B. Cross-dataset relative features, leave-one-dataset-out", "",
              f"{F['n_cells']} dev cells ({F['n_sig']} with p<0.05; always-helps sign acc {F['always_helps']:.2f}); features after label-free pruning: "
              f"tex {F['sizes']['tex']}, tex+shape {F['sizes']['ts']} (of {F['sizes']['base']}). Candidates (set x model), selection inside folds, "
              f"{F['N']} selection-aware permutations (max over all 6 candidates re-run per shuffle). On-harmony uses <=6 labels/case "
              f"(evenly spaced by volume), <=20 cases per key (hand-made features used all labels/<=40 cases): a feature-definition difference.", "",
              "| candidate | LODO rho | sign-acc (sig cells) |", "|---|--:|--:|"]
        for c, v in sorted(F["cand"].items(), key=lambda kv: -np.nan_to_num(kv[1]["rho"], nan=-9)):
            L.append(f"| {c[0]}/{c[1]} | {v['rho']:+.3f} | {v['sa']:.2f} |")
        L += ["", f"**Frozen: {F['best'][0]}/{F['best'][1]}**, LODO rho {F['rho']:+.3f}, selection-aware perm p = {F['p_rho']:.3f} "
              f"(null 95th pct {F['null95']:+.3f}); sign-acc {F['sa']:.2f} vs always-helps {F['always_helps']:.2f} (perm p={F['p_sa']:.3f}). "
              f"Per held-out dataset rho: " + ", ".join(f"{g} {v:+.2f}" for g, v in F["pergrp"].items()),
              f"Hand-made 7-feature family, same cells + same shuffles: best {h['best']}, rho {h['rho']:+.3f} (p={h['p_rho']:.3f}), sign-acc {h['sa']:.2f} (p={h['p_sa']:.3f}).", "",
              "Frozen model features: " + ", ".join(F["frozen"]["features"][:10]) + ". Most-selected across LODO folds: "
              + "; ".join(f"{k} ({v}/5)" for k, v in F["freq"].head(6).items()), ""]
        ax[1].hist(F["null_rho"], bins=22, color="lightgrey", label="radiomics null (max over 6 cands)")
        ax[1].hist(h["null_rho"], bins=22, color="tan", alpha=.5, label="hand-made null")
        ax[1].axvline(F["rho"], c="r", label=f"radiomics obs {F['rho']:+.2f}")
        ax[1].axvline(h["rho"], c="saddlebrown", ls="--", label=f"hand-made obs {h['rho']:+.2f}")
        ax[1].set_xlabel("pooled LODO Spearman")
        ax[1].set_title("B. cross-dataset LODO vs selection-aware null")
        ax[1].legend(fontsize=7)
    if (DATA / "radiomics_B_openms.pkl").exists():
        O = pickle.load(open(DATA / "radiomics_B_openms.pkl", "rb"))
        L += ["### Open-MS, frozen model (SECOND LOOK: Open-MS was already scored once by the earlier hand-made model)", "",
              "| cell | score | pred | delta | p | hit |", "|---|--:|:-:|--:|--:|:-:|"]
        for r in O["cells"].itertuples():
            L.append(f"| {r.name} | {r.score:+.2f} | {r.pred} | {r.delta:+.2f} | {r.p:.3g} | {'HIT' if r.hit else 'miss'} |")
        L += ["", f"Hits {O['hits']}/{O['n']} (significant cells {O['sig_hits']}/{O['n_sig']}, always-helps {O['always']}/{O['n_sig']}); "
              f"Spearman {O['rho']:+.2f}. Previous hand-made model: 2/4, Spearman 0.00.", ""]
    fig.tight_layout()
    fig.savefig(PLOTS / "radiomics_fillswap.png", dpi=130)
    (TABLES / "radiomics_fillswap.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
