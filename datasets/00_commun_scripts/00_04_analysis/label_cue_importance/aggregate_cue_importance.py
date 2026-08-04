#!/usr/bin/env python
"""
Turn per-(subject, label) cue files from compute_label_cues.py into the relative-importance table.

Protocol = Martin, Fowlkes & Malik (TPAMI 2004): score each cue as a standalone boundary detector
by average precision, then combine the cues with logistic regression. Two importance readings are
reported for every cue, deliberately (LITERATURE_REVIEW.md sec.5):

  standalone AP   how far this cue alone gets you. OVER-credits a cue that is correlated with
                  another, since it gets full credit for shared evidence.
  marginal dAP    drop in the combined model's AP when this cue is removed -- its UNIQUE
                  contribution. UNDER-credits a correlated cue, for the same reason permutation
                  importance does (arxiv 2312.10858).

The truth is bracketed by the pair; a cue high on BOTH is unambiguously important. We report the
pair rather than fabricating a single "importance %". AP has a chance baseline of exactly 0.5
(compute_label_cues.py balances positives against negatives), so 0.5 is the no-information line
for the standalone columns and 0.0 for the marginal ones.

The combined model is fitted with LEAVE-ONE-SUBJECT-OUT cross-validation, so no subject
contributes to both fit and evaluation.

Usage:
  python aggregate_cue_importance.py --in-dir DIR [DIR ...] --out-dir DIR
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cue_metrics import CUE_NAMES, average_precision  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FAMILY = {"bg": "intensity", "eg_coh": "boundary", "eg_sharp": "boundary", "tg": "texture"}


def _binom_sf(k: int, n: int, p: float) -> float:
    """P(X > k) for X ~ Binomial(n, p) -- exact, no scipy dependency in the analysis path."""
    from math import comb
    if k >= n:
        return 0.0
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1, n + 1)))


def _fit_logistic(X: np.ndarray, y: np.ndarray, iters: int = 300, lr: float = 0.5) -> np.ndarray:
    """Plain L2-regularised logistic regression by gradient descent on standardised features.
    Implemented here to keep the analysis dependency-free and fully inspectable; the model is a
    4-feature linear one, so there is nothing an optimiser library would add."""
    Xb = np.c_[X, np.ones(len(X))]
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        grad = Xb.T @ (p - y) / len(y) + 1e-3 * np.r_[w[:-1], 0.0]
        w -= lr * grad
    return w


MAX_FIT_POINTS = 400   # per subject, for FITTING the combined model only


def _prep_fit_data(per_subj: dict, seed: int = 0) -> dict:
    """Draw the fitting subsample ONCE per subject, for all cues at once.

    Doing it inside the fold x cue-subset loops instead re-drew it ~280k times for a full run and
    made the aggregation slower than the GPU job that produced the data."""
    out = {}
    for i, (sub, rec) in enumerate(sorted(per_subj.items())):
        n = len(rec["y"])
        idx = (np.arange(n) if n <= MAX_FIT_POINTS
               else np.random.default_rng(seed + i).choice(n, MAX_FIT_POINTS, replace=False))
        out[sub] = (np.c_[tuple(rec[c][idx] for c in CUE_NAMES)], rec["y"][idx])
    return out


N_FOLDS = 5   # grouped K-fold over subjects, NOT leave-one-subject-out


def _loso_ap(per_subj: dict, prepped: dict, cols: list[int]) -> float:
    """Subject-grouped K-fold AP of the logistic combination of the cue columns `cols`.

    Grouped by SUBJECT, so no subject contributes to both fit and evaluation -- the property that
    matters. It was leave-one-subject-OUT until 2026-08-02; with 40-subject groups that is 40 fits
    x 7 cue subsets x 20 groups = 5600 fits of a 6-feature model, which pinned ~6 cores of a LOGIN
    NODE for 9+ minutes (~54 CPU-min, past the Alliance login-node allowance) for no statistical
    gain over K-fold. K=5 gives the same estimand at 8x less compute.

    Fit uses the capped subsample (MAX_FIT_POINTS per subject); EVALUATION uses every point of the
    held-out subjects. The standalone per-cue APs elsewhere always use every point."""
    subs = sorted(per_subj)
    if len(subs) < 3:
        return float("nan")
    k = min(N_FOLDS, len(subs))
    folds = [subs[i::k] for i in range(k)]
    scores, ys = [], []
    for fold in folds:
        tr = [s for s in subs if s not in fold]
        if not tr:
            continue
        Xtr = np.vstack([prepped[s][0][:, cols] for s in tr])
        ytr = np.concatenate([prepped[s][1] for s in tr])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        w = _fit_logistic((Xtr - mu) / sd, ytr)
        for held in fold:
            Xte = np.c_[tuple(per_subj[held][CUE_NAMES[c]] for c in cols)]
            scores.append(((Xte - mu) / sd) @ w[:-1] + w[-1])
            ys.append(per_subj[held]["y"])
    return average_precision(np.concatenate(scores), np.concatenate(ys))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, type=Path, nargs="+")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple, dict] = defaultdict(dict)
    for d in args.in_dir:
        for f in sorted(d.glob("*.npz")):
            ds, mod, case, label = f.stem.split("__")
            z = np.load(f)
            rec = {k: z[k].astype(np.float64) for k in (*CUE_NAMES, "y", "stratum")}
            for k in z.files:
                if k.startswith(("ngf_", "reg_")):
                    rec[k] = float(z[k])
            groups[(ds, mod, label)][case] = rec
    if not groups:
        log.error("no .npz found under %s", args.in_dir)
        return 1

    rows = []
    for (ds, mod, label), per_subj in sorted(groups.items()):
        n_sub = len(per_subj)
        standalone = {c: float(np.mean([average_precision(v[c], v["y"]) for v in per_subj.values()]))
                      for c in CUE_NAMES}
        prepped = _prep_fit_data(per_subj)
        allc = list(range(len(CUE_NAMES)))
        full = _loso_ap(per_subj, prepped, allc)
        marginal = {c: full - _loso_ap(per_subj, prepped, [j for j in allc if j != i])
                    for i, c in enumerate(CUE_NAMES)}
        frac_interior = float(np.mean([np.mean(v["stratum"] == 1) for v in per_subj.values()]))
        # Descriptive NGF(image,label) surface-alignment index, averaged over subjects. Reported
        # next to the detector cues but NOT scored like them -- see ngf_label_alignment().
        ngf_obs = float(np.nanmean([v.get("ngf_observed", np.nan) for v in per_subj.values()]))
        ngf_null = float(np.nanmean([v.get("ngf_null_mean", np.nan) for v in per_subj.values()]))
        ngf_z = float(np.nanmean([v.get("ngf_z", np.nan) for v in per_subj.values()]))
        # Regional texture contrast (ROI vs surrounding shell, multiscale HOG, placement null).
        # HOG is the trustworthy descriptor here -- contrast-normalised, so blind to a pure
        # intensity offset; the ordinal one leaks at regional scale (see cue_metrics --sanity).
        regk = sorted({k for v in per_subj.values() for k in v if k.startswith("reg_") and k.endswith("_z")})
        reg = {k: float(np.nanmean([v.get(k, np.nan) for v in per_subj.values()])) for k in regk}
        rows.append(dict(dataset=ds, modality=mod, label=label, n_subjects=n_sub,
                         combined_ap=full, frac_neg_interior=frac_interior,
                         ngf_observed=ngf_obs, ngf_null=ngf_null, ngf_z=ngf_z, **reg,
                         **{f"ap_{c}": standalone[c] for c in CUE_NAMES},
                         **{f"dap_{c}": marginal[c] for c in CUE_NAMES}))
        log.info("%s/%s/%s  n=%d  combined=%.3f  standalone=%s", ds, mod, label, n_sub, full,
                 {k: round(v, 3) for k, v in standalone.items()})

    # ── the two claims, tested directly ──────────────────────────────────────────────────────
    # The wide per-label table below answers "what is the cue mix", but the paper makes two
    # narrower claims that deserve their own explicit tests rather than being read out of it:
    #   (a) brats/open-ms labels DO have a texture signal   -> is tg above the 0.5 chance line,
    #       within each dataset? (Chaos also having texture is irrelevant to this claim.)
    #   (b) brats/open-ms labels LACK the sharp boundary / strong intensity gradient that chaos
    #       organs have -> compare the boundary cues ACROSS datasets.
    claim = []
    by_ds: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for (ds, mod, label), per_subj in sorted(groups.items()):
        for c in CUE_NAMES:
            by_ds[ds][c] += [average_precision(v[c], v["y"]) for v in per_subj.values()]
    for ds in sorted(by_ds):
        n = len(by_ds[ds]["tg"])
        m = {c: float(np.mean(by_ds[ds][c])) for c in CUE_NAMES}
        # (a) one-sided sign test of per-(subject,label) tg AP against the 0.5 chance line.
        tg = np.asarray(by_ds[ds]["tg"])
        k = int((tg > 0.5).sum())
        p_a = float(_binom_sf(k - 1, n, 0.5)) if n else float("nan")
        claim.append(dict(dataset=ds, n=n, tg_mean=m["tg"], tg_frac_above=k / max(n, 1), tg_p=p_a,
                          bg=m["bg"], eg_coh=m["eg_coh"], eg_sharp=m["eg_sharp"]))
    hdr = list(rows[0])
    csv = args.out_dir / "cue_importance.csv"
    csv.write_text(",".join(hdr) + "\n" + "\n".join(
        ",".join(f"{r[h]:.4f}" if isinstance(r[h], float) else str(r[h]) for h in hdr)
        for r in rows) + "\n")

    cl = ["## The two claims, tested directly", "",
          "**(a) Do these labels carry a texture signal at all?** `tg` average precision against",
          "its 0.5 chance line, pooled over every (subject, label) of the dataset; `p` is a",
          "one-sided sign test. This is a WITHIN-dataset claim -- chaos also having texture does",
          "not weaken it.", "",
          "**(b) Do they have the sharp boundary / strong intensity gradient that organs have?**",
          "The boundary cues, compared ACROSS datasets. `eg_coh` ('is there a clean wall here')",
          "is the decisive one; at or below 0.500 the outline is no more wall-like than the tissue",
          "beside it.", "",
          "| dataset | n(subj,label) | tg mean | frac tg>0.5 | p (tg>chance) | bg | eg_coh | eg_sharp |",
          "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for c in claim:
        cl.append("| {dataset} | {n} | {tg_mean:.3f} | {tg_frac_above:.2f} | {tg_p:.2g} | "
                  "{bg:.3f} | {eg_coh:.3f} | {eg_sharp:.3f} |".format(**c))
    cl += ["", "---", ""]

    md = ["# Label boundary-cue importance",
          "",
          *cl,
          "Relative importance of three local cue families at each label's surface, following",
          "Martin, Fowlkes & Malik (TPAMI 2004). Chance = 0.500 for standalone AP columns",
          "(positives and negatives are balanced), 0.000 for marginal dAP columns.",
          "**Read the pair, not one column**: standalone AP over-credits correlated cues, marginal",
          "dAP under-credits them; the truth is bracketed. A standalone AP **below 0.5 is not",
          "'uninformative'** -- it means the cue is informative with the OPPOSITE sign (e.g. label",
          "surfaces that are systematically LESS sharp than nearby non-surface locations, i.e.",
          "gradual transitions). Such a cue can still carry a large marginal dAP; the logistic",
          "combination is free to use a negative weight. Grounding and limits:",
          "`00_commun_scripts/00_04_analysis/label_cue_importance/LITERATURE_REVIEW.md`.",
          "",
          "| dataset | modality | label | n | combined AP | AP bg (intens.) | AP eg_coh | AP eg_sharp | AP tg (texture) | dAP bg | dAP eg_coh | dAP eg_sharp | dAP tg | NGF idx | NGF null | NGF z |",
          "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        md.append("| {dataset} | {modality} | {label} | {n_subjects} | {combined_ap:.3f} | "
                  "{ap_bg:.3f} | {ap_eg_coh:.3f} | {ap_eg_sharp:.3f} | {ap_tg:.3f} | "
                  "{dap_bg:+.3f} | {dap_eg_coh:+.3f} | {dap_eg_sharp:+.3f} | {dap_tg:+.3f} | "
                  "{ngf_observed:.3f} | {ngf_null:.3f} | {ngf_z:+.1f} |".format(**r))
    md += ["",
           "`NGF idx` is the ablation's own NGF formula (Haber & Modersitzki 2006) applied between",
           "the image gradient field and the label's gradient field on the label surface, against a",
           "rotation/translation permutation null (`NGF null`, which sits at the 1/3 chance floor",
           "for 3-D gradient orientations). It is DESCRIPTIVE, not one of the scored cues: used as a",
           "detector it is circular, since |grad L| is nonzero only on the surface. Read `NGF z`.",
           "",
           "`frac_neg_interior` (in the CSV) records what fraction of each row's negative points",
           "came from the label INTERIOR rather than its surroundings. Small labels (MS lesions)",
           "have no interior left after the window-exclusion, so their negatives are all",
           "surroundings; large organs have both. Check it before comparing rows with very",
           "different values -- the estimand is the same but the negative pool is not."]
    (args.out_dir / "cue_importance_summary.md").write_text("\n".join(md) + "\n")
    log.info("wrote %s and %s", csv, args.out_dir / "cue_importance_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
