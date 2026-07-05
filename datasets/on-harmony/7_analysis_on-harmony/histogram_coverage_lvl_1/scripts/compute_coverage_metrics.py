#!/usr/bin/env python
"""
Histogram-manifold coverage metrics — Pillar 2 of the PALETTE paper (per-group design).

The real ON-Harmony data is NOT one manifold — it is a union of well-separated
(scanner × contrast) clusters. A single pooled k-NN sets the coverage radius from the
dominant cluster (GRE, ~81%), mismeasuring sparse clusters. So Coverage is computed
PER (contrast × scanner) group at that group's OWN local scale, then macro-averaged
(equal weight per regime — the principled replacement for downsampling/balancing).

Metrics (only two axes, no redundancy — see GROUNDING_AUDIT.md):
  Coverage   — Naeem et al. ICML 2020, per group → macro-average.
               "Does synthesis reach each real regime?" The claim-aligned metric.
  Vendi      — Friedman & Dieng TMLR 2023, global on the synthetic cloud.
               "How diverse is the augmentation?"

DROPPED (with reasons in GROUNDING_AUDIT.md): Precision/Recall (non-robust duplicates of
Density/Coverage, Naeem 2020; Recall inflated by domain-randomisation outliers) and Density
(fidelity is not the objective; global density is GRE-confounded and redundant here).

Reference geometry (scaler + PCA) fit on ALL real once → one fixed feature space shared by
every method and group. Groups with < k+1 real are reported N/A (never silently dropped).

Usage:
  run_job --gpus 0 --cpus 8 --mem 32G --wait -- .venv/bin/python compute_coverage_metrics.py \\
    --real-csv real_regional_hist31.csv --synth-csv synth_regional_hist31.csv --output-dir outputs
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from prdc import compute_prdc
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from vendi_score import vendi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SEED = 42
PRDC_K = 5
VENDI_N = 256
GROUP_REAL_CAP = 300          # subsample per group for tractable, reproducible coverage
MIN_GROUP_REAL = PRDC_K + 1   # need >= k+1 real to estimate coverage at k
METHOD_ORDER = ["palette", "synthseg_em", "synthseg_noem", "auglab_default"]
REF_METHOD, CMP_METHOD = "palette", "synthseg_em"

REAL_META = {"subject", "session", "modality_id", "acq_tag", "scanner_model",
             "scanner_vendor", "cohort_category", "image_path", "mask_path", "label_map_path"}
SYNTH_META = {"method", "subject", "session", "key", "run"}


def feat_cols(df, meta):
    return [c for c in df.columns if c not in meta and not c.startswith("diagnostics_")]


def vendi_score(X, n, rng):
    if len(X) == 0:
        return float("nan")
    Xs = X if len(X) <= n else X[rng.choice(len(X), n, replace=False)]
    return float(vendi.score_X(Xs))


def coverage_of(real_g, fake, k):
    """Naeem-2020 Coverage of a single real group by the fake cloud (0 if group too small)."""
    if len(real_g) < k + 1 or len(fake) < k + 1:
        return float("nan")
    return float(compute_prdc(real_features=real_g, fake_features=fake, nearest_k=k)["coverage"])


def macro_coverage(real_groups, fake, k):
    """Mean coverage over viable groups; returns (macro, {group: coverage})."""
    per = {}
    for name, rg in real_groups.items():
        c = coverage_of(rg, fake, k)
        if not np.isnan(c):
            per[name] = c
    macro = float(np.mean(list(per.values()))) if per else float("nan")
    return macro, per


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real-csv", type=Path, required=True)
    p.add_argument("--synth-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--pca-variance", type=float, default=0.90)
    p.add_argument("--n-boot", type=int, default=500)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df_real = pd.read_csv(args.real_csv)
    df_synth = pd.read_csv(args.synth_csv)
    common = sorted(set(feat_cols(df_real, REAL_META)) & set(feat_cols(df_synth, SYNTH_META)))
    if not common:
        raise SystemExit("No common feature columns.")
    log.info("Feature dims: %d | real rows %d | synth rows %d", len(common), len(df_real), len(df_synth))

    Xr = df_real[common].to_numpy(np.float64)
    keep = (~np.isnan(Xr).all(0)) & (np.nanvar(Xr, 0) > 0)
    common = [c for c, kp in zip(common, keep) if kp]

    # Reference geometry fit on ALL real (one fixed space; no balancing).
    imp = SimpleImputer(strategy="mean").fit(df_real[common].to_numpy(np.float64))
    scal = StandardScaler().fit(imp.transform(df_real[common].to_numpy(np.float64)))
    pca = PCA(n_components=args.pca_variance, svd_solver="full", random_state=SEED)
    pca.fit(scal.transform(imp.transform(df_real[common].to_numpy(np.float64))))
    log.info("PCA: %d → %d dims (%.0f%% var, fit on all real)", len(common), pca.n_components_, args.pca_variance * 100)

    def embed(df):
        return pca.transform(scal.transform(imp.transform(df[common].to_numpy(np.float64)))).astype(np.float64)

    # Real (scanner × contrast) groups, each subsampled to <= cap (fixed).
    df_real = df_real.copy()
    df_real["_mod"] = df_real["modality_id"].astype(str).str.split("_").str[0]
    df_real["_sc"] = df_real.get("scanner_model", "unknown").astype(str)
    real_groups, skipped = {}, []
    for (mod, sc), g in df_real.groupby(["_mod", "_sc"]):
        if len(g) < MIN_GROUP_REAL:
            skipped.append((f"{mod}×{sc}", len(g))); continue
        gg = g if len(g) <= GROUP_REAL_CAP else g.sample(GROUP_REAL_CAP, random_state=SEED)
        real_groups[f"{mod}×{sc}"] = embed(gg)
    log.info("Viable groups: %d | skipped (<%d real): %s", len(real_groups), MIN_GROUP_REAL,
             ", ".join(f"{n}({c})" for n, c in skipped) or "none")

    methods = [m for m in METHOD_ORDER if m in set(df_synth["method"])]
    synth_emb = {m: embed(df_synth[df_synth["method"] == m]) for m in methods}
    vendi_real = vendi_score(np.vstack(list(real_groups.values())), VENDI_N, np.random.default_rng(SEED))

    # ── Point estimates ───────────────────────────────────────────────────────
    per_group_rows, point = [], {}
    for m in methods:
        macro, per = macro_coverage(real_groups, synth_emb[m], PRDC_K)
        v = vendi_score(synth_emb[m], VENDI_N, np.random.default_rng(SEED))
        point[m] = {"macro_coverage": macro, "vendi": v, "n_synth": len(synth_emb[m])}
        for gname, cov in per.items():
            per_group_rows.append({"method": m, "group": gname, "coverage": round(cov, 6)})
        log.info("[point] %-16s macro_coverage=%.4f  vendi=%.2f", m, macro, v)

    # ── Subject-level bootstrap (resample synth side; real groups fixed) ───────
    subj_idx = {}
    for m in methods:
        sm = df_synth[df_synth["method"] == m].reset_index(drop=True)
        subj_idx[m] = {s: g.index.to_numpy() for s, g in sm.groupby("subject")}
    subjects = sorted(set(df_synth["subject"]))
    log.info("Bootstrap: %d iters over %d subjects", args.n_boot, len(subjects))

    boot = {m: {"macro_coverage": [], "vendi": []} for m in methods}
    paired = {"macro_coverage": [], "vendi": []}
    for b in range(args.n_boot):
        draw = rng.choice(subjects, len(subjects), replace=True)
        cur = {}
        for m in methods:
            idx = np.concatenate([subj_idx[m][s] for s in draw if s in subj_idx[m]])
            fake = synth_emb[m][idx]
            macro, _ = macro_coverage(real_groups, fake, PRDC_K)
            v = vendi_score(fake, VENDI_N, rng)
            boot[m]["macro_coverage"].append(macro); boot[m]["vendi"].append(v)
            cur[m] = {"macro_coverage": macro, "vendi": v}
        if REF_METHOD in cur and CMP_METHOD in cur:
            for k in ("macro_coverage", "vendi"):
                paired[k].append(cur[REF_METHOD][k] - cur[CMP_METHOD][k])
        if (b + 1) % 100 == 0:
            log.info("  boot %d / %d", b + 1, args.n_boot)

    def ci(v):
        a = np.asarray(v, float); a = a[~np.isnan(a)]
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (float("nan"),) * 2

    # ── Outputs ───────────────────────────────────────────────────────────────
    rows = []
    for m in methods:
        r = {"method": m, "n_synth": point[m]["n_synth"]}
        for k in ("macro_coverage", "vendi"):
            lo, hi = ci(boot[m][k])
            r[k] = round(point[m][k], 6); r[f"{k}_ci_lo"] = round(lo, 6); r[f"{k}_ci_hi"] = round(hi, 6)
        rows.append(r)
    pd.DataFrame(rows).to_csv(args.output_dir / "coverage_metrics.csv", index=False)
    pd.DataFrame(per_group_rows).to_csv(args.output_dir / "coverage_per_group.csv", index=False)

    prows = []
    for k in ("macro_coverage", "vendi"):
        d = np.asarray(paired[k], float); d = d[~np.isnan(d)]
        lo, hi = ci(paired[k])
        pval = 2 * min((d <= 0).mean(), (d >= 0).mean()) if len(d) else float("nan")
        prows.append({"metric": k, f"{REF_METHOD}_minus_{CMP_METHOD}": round(float(np.median(d)), 6) if len(d) else float("nan"),
                      "ci_lo": round(lo, 6), "ci_hi": round(hi, 6), "boot_p_two_sided": round(float(pval), 6)})
    pd.DataFrame(prows).to_csv(args.output_dir / "coverage_paired.csv", index=False)

    summary = {"config": {"design": "per-group coverage (macro-avg) + global vendi",
                          "pca_variance": args.pca_variance, "pca_dims": int(pca.n_components_),
                          "prdc_k": PRDC_K, "vendi_n": VENDI_N, "group_real_cap": GROUP_REAL_CAP,
                          "n_boot": args.n_boot, "n_viable_groups": len(real_groups),
                          "skipped_groups": {n: c for n, c in skipped}, "seed": SEED,
                          "vendi_real_reference": round(vendi_real, 4)},
               "point": {m: {k: round(point[m][k], 6) for k in ("macro_coverage", "vendi")} for m in methods},
               "methods": methods}
    (args.output_dir / "coverage_summary.json").write_text(json.dumps(summary, indent=2))

    log.info("\n%s", pd.DataFrame(rows).to_string(index=False))
    log.info("\nPaired %s − %s:\n%s", REF_METHOD, CMP_METHOD, pd.DataFrame(prows).to_string(index=False))
    log.info("Vendi real reference: %.3f | Saved → %s", vendi_real, args.output_dir)


if __name__ == "__main__":
    main()
