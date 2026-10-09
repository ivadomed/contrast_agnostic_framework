#!/usr/bin/env python3
"""
Self-checks for the macroΔ significance machinery in stat_tests.py (design/tail
primitives) and aggregate_from_config.py (build_design, and the per-case flat-pool
that feeds it). Synthetic data only, no dataset needed:

    .venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/test_significance_units.py

What each test proves (3 and 6 are the ones that distinguish a working fix from a
plausible-looking one):
  1. flat, all-unique-unit data  -> same macroΔ and same null variance as the legacy `macro_perm`
  2. nested organ group          -> analytic sign-flip variance == Monte-Carlo sign-flip null
  2b. p-value TAIL               -> exact when few units; matches Monte-Carlo when 3 heavy units
                                    dominate (where a plain normal approximation is 10x+ off)
  3. more cases in a nested group-> its sign-flip SD SHRINKS (~1/sqrt(n)); a per-organ collapse can't
  4. same patient scored by two trained models / in two contrasts -> ONE unit
  5. the point estimate (macroΔ) is untouched by any of this
  6. a case's per-case value FLAT-POOLS labels+folds (cross_fold_class_mean's own
     convention), not an equal-weight-per-label average of already-fold-averaged
     numbers — the 2026-09-21 label-averaging regression: a label with fewer valid
     folds than another must NOT get the same weight as a fully-covered one.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "00_00_utils"))
from stat_tests import macro_perm, macro_perm_design, macro_ci_design, stratum_arrays  # noqa: E402
from scipy import stats  # noqa: E402

_spec = importlib.util.spec_from_file_location("aggregate_from_config", HERE / "aggregate_from_config.py")
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)


def _raw(col_case_labels: dict) -> dict:
    """{(col, case_tag): {label: {fold: value}}} -> col -> case -> label -> fold -> value
    (load_run_cases_raw's return shape — what build_design/resolve_group/
    _case_flat_value consume)."""
    out = {}
    for (col, tag), labs in col_case_labels.items():
        out.setdefault(col, {})[tag] = labs
    return out


def _pair(rng, tags, delta, sd=0.05, col="c", label="x", n_folds=3):
    """(ref, comp) raw dicts: one column/label, `n_folds` folds per case (all equal —
    the case where equal-weight-per-label and flat-pooling agree; test 6 covers the
    unequal-fold-count case that tells them apart)."""
    comp, ref = {}, {}
    for t in tags:
        base = rng.uniform(0.5, 0.9)
        comp[(col, t)] = {label: {f"fold{f}": base + rng.normal(0, sd) for f in range(n_folds)}}
        ref[(col, t)] = {label: {f"fold{f}": base + delta + rng.normal(0, sd) for f in range(n_folds)}}
    return _raw(ref), _raw(comp)


def test_flat_matches_legacy():
    rng = np.random.default_rng(1)
    arrs, groups, R, C = [], {}, {}, {}
    for j, (n, delta) in enumerate([(8, 0.01), (20, 0.03), (40, -0.005)]):
        tags = [f"ds|p{j}_{i}" for i in range(n)]
        r, c = _pair(rng, tags, delta, col=f"col{j}")
        R.update(r); C.update(c); groups[f"g{j}"] = f"col{j}"
        arrs.append(np.array([agg._case_flat_value(r[f"col{j}"][t]) - agg._case_flat_value(c[f"col{j}"][t])
                              for t in tags]))
    entries, K = agg.build_design(groups, R, C, "patient")
    new, old = macro_perm_design(entries, K, True), macro_perm(arrs, True)
    assert np.isclose(new[0], old[0], rtol=1e-12), (new, old)                       # macroΔ identical
    sd_design = _sd(entries, K)
    sd_legacy = np.sqrt(sum((a ** 2).sum() / a.size ** 2 for a in arrs) / len(arrs) ** 2)
    assert np.isclose(sd_design, sd_legacy, rtol=1e-12), (sd_design, sd_legacy)     # null variance identical
    assert 1 / 3 < new[2] / old[2] < 3, (new, old)           # diffuse weights: tail ≈ the normal approximation
    lo, hi = macro_ci_design(entries, K, b_boot=4000)
    assert lo < old[0] < hi
    print("ok 1  flat design: same macroΔ + null variance as legacy macro_perm; p", round(new[2], 5), "vs normal", round(old[2], 5))


def _merge_raw(dst: dict, src: dict) -> None:
    """dst[col][case] gets src[col][case]'s labels MERGED in (not overwritten) — a raw
    dict's per-case value is itself a dict keyed by label, so a plain .update() at the
    case level would drop whichever organ was already there for that case."""
    for col, cases in src.items():
        d = dst.setdefault(col, {})
        for case, labs in cases.items():
            d.setdefault(case, {}).update(labs)


def _ct_like(rng, n_amos, n_chaos, delta=(0.04, 0.02, 0.03)):
    """Nested group: liver pools two sources, spleen/kidney from one; AMOS patients carry all 3 organs."""
    R, C = {}, {}
    for organ, d in zip(("liver", "spleen", "kidney"), delta):
        tags = [f"amos|a{i}" for i in range(n_amos)]
        r, c = _pair(rng, tags, d, col="amos_ct", label=organ)
        _merge_raw(R, r); _merge_raw(C, c)
    r, c = _pair(rng, [f"chaos|c{i}" for i in range(n_chaos)], delta[0], col="chaos_ct", label="liver")
    _merge_raw(R, r); _merge_raw(C, c)
    group = {"liver": [{"column": "amos_ct", "label": "liver"}, {"column": "chaos_ct", "label": "liver"}],
             "spleen": {"column": "amos_ct", "label": "spleen"},
             "kidney": {"column": "amos_ct", "label": "kidney"}}
    return R, C, group


def _sd(entries, K):
    tot = {}
    for _, _, u, c in entries:
        tot[u] = tot.get(u, 0.0) + c / K
    return float(np.sqrt(sum(v * v for v in tot.values())))


def test_variance_matches_monte_carlo():
    rng = np.random.default_rng(2)
    R, C, group = _ct_like(rng, 60, 20)
    entries, K = agg.build_design({"CT": group}, R, C, "patient")
    units = sorted({e[2] for e in entries}); ix = {u: i for i, u in enumerate(units)}
    idx = np.array([ix[e[2]] for e in entries]); c = np.array([e[3] for e in entries])
    sg = np.random.default_rng(0).choice([-1, 1], size=(40000, len(units)))
    mc = (sg[:, idx] * c).sum(1) / K
    assert abs(mc.std() - _sd(entries, K)) / _sd(entries, K) < 0.02, (mc.std(), _sd(entries, K))
    print("ok 2  analytic sign-flip SD vs Monte-Carlo:", round(_sd(entries, K), 5), round(mc.std(), 5))


def _entries(ts):
    return [(0, "l", f"u{i}", float(t)) for i, t in enumerate(ts)], 1


def test_tail_exact_and_monte_carlo():
    # (i) few units -> exactly the brute-force sign-flip test
    ts = [0.5, 0.3, -0.1, 0.2, 0.05, 0.4, -0.02, 0.15]
    e, K = _entries(ts)
    sg = np.array([[1 if (b >> i) & 1 else -1 for i in range(len(ts))] for b in range(2 ** len(ts))])
    T = sg @ np.array(ts)
    exact = (T >= sum(ts) - 1e-12).mean()
    assert np.isclose(macro_perm_design(e, K, True)[2], exact, rtol=1e-12)
    # (ii) 3 dominant units + 100 light ones: the tail must match Monte-Carlo, and the plain
    # normal approximation must be visibly wrong here (that is why it is not used)
    ts = [0.5, 0.3, 0.2] + [0.004] * 60 + [-0.004] * 40
    e, K = _entries(ts)
    p_design = macro_perm_design(e, K, True)[2]
    t = np.array(ts)
    hits = 0
    rng = np.random.default_rng(0)
    for _ in range(8):
        hits += ((rng.choice([-1.0, 1.0], size=(50000, t.size)) @ t) >= t.sum() - 1e-12).sum()
    p_mc, se = hits / 400000, np.sqrt(hits) / 400000
    p_normal = stats.norm.sf(t.sum() / np.sqrt((t ** 2).sum()))
    # the light units enter as a (continuous) normal term, so a purely +-0.004 discrete remainder
    # leaves ~20% tail error here; the plain normal approximation is >10x off (asserted below)
    assert abs(p_design - p_mc) < 0.3 * p_mc, (p_design, p_mc, se)
    assert p_normal > 5 * p_mc, (p_normal, p_mc)
    # (iii) many equal units -> the tail agrees with the normal approximation
    ts = [1.0] * 400 + [-1.0] * 300
    e, K = _entries(ts)
    t = np.array(ts)
    assert 0.9 < macro_perm_design(e, K, True)[2] / stats.norm.sf(t.sum() / np.sqrt((t ** 2).sum())) < 1.1
    print("ok 2b tail: exact when few units;", f"3-heavy-unit case design={p_design:.4f} MC={p_mc:.4f} normal={p_normal:.4f}")


def test_more_cases_shrink_variance():
    sds = {}
    for n in (25, 100, 400):
        rng = np.random.default_rng(3)
        R, C, group = _ct_like(rng, n, n // 3)
        sds[n] = _sd(*agg.build_design({"CT": group}, R, C, "patient"))
        # the array view the legacy flat callers use must show the same thing (one element per patient)
        arr = agg.resolve_group(group, R, C)
        assert arr.size > 3, "nested group collapsed back to per-organ elements"
    assert sds[100] < 0.65 * sds[25] and sds[400] < 0.65 * sds[100], sds
    print("ok 3  sign-flip SD shrinks as cases are added:", {k: round(v, 5) for k, v in sds.items()})


def test_patient_units_merge():
    rng = np.random.default_rng(4)
    pats = [f"chaos|MR{i}" for i in range(4)]
    R, C = {}, {}
    for col in ("t1in", "t1out"):                        # same 4 patients, two test contrasts
        for mod, d in (("m1", 0.02), ("m2", -0.01)):     # scored by two trained models
            r, c = _pair(rng, [f"{mod}::{p}" for p in pats], d, col=col)
            R.setdefault(col, {}).update(r[col]); C.setdefault(col, {}).update(c[col])
    groups = {"t1in": "t1in", "t1out": "t1out"}
    e_pat, K = agg.build_design(groups, R, C, "patient")
    e_str, _ = agg.build_design(groups, R, C, "stratum")
    assert len({e[2] for e in e_pat}) == 4, "patient scope: 4 patients must be 4 units"
    assert len({e[2] for e in e_str}) == 8, "stratum scope: 4 patients x 2 strata must be 8 units"
    # merging the two trained models: 2 arrays (one per contrast) of 4 elements, not 8
    assert [a.size for a in stratum_arrays(e_pat)] == [4, 4]
    expect = np.mean([agg._case_flat_value(R["t1in"][t]) - agg._case_flat_value(C["t1in"][t]) for t in R["t1in"]])
    assert np.isclose(agg.resolve_group("t1in", R, C).mean(), expect)
    print("ok 4  8 model-tagged cases -> 4 patient units; 4 patients x 2 contrasts -> 4 (patient) / 8 (stratum) units")


def test_point_estimate_untouched():
    rng = np.random.default_rng(5)
    R, C, group = _ct_like(rng, 50, 20)
    entries, K = agg.build_design({"CT": group}, R, C, "patient")
    d = lambda col, lab: np.array([agg._case_flat_value(R[col][t], lab) - agg._case_flat_value(C[col][t], lab)
                                   for t in R[col]])
    liver = np.concatenate([d("amos_ct", "liver"), d("chaos_ct", "liver")])   # a list POOLS cases
    expect = np.mean([liver.mean(), d("amos_ct", "spleen").mean(), d("amos_ct", "kidney").mean()])
    got = macro_perm_design(entries, K, True)[0]
    assert np.isclose(got, expect, rtol=1e-12), (got, expect)
    print("ok 5  macroΔ = per-organ-then-cross-organ mean, unchanged:", round(got, 6))


def test_label_flat_pool_not_equal_weight():
    """The 2026-09-21 regression this closed: a label=None leaf (e.g. a plain contrast
    column, or brats2024-glioma's flat non-grouped configs) must FLAT-POOL every
    applicable label's raw fold-level values — same convention as
    cross_fold_class_mean/resolve_group_value for the displayed table cell — not average
    each label's own already-fold-averaged mean with equal weight regardless of how many
    folds backed it. Construct a case where label A has 3 valid folds and label B has
    only 1 (a tumor sub-region absent — Dice undefined/dropped — in 2 of 3 folds'
    predictions): flat-pool weights A's 3 samples over B's 1 (4:1); equal-weight-per-
    label would give A and B equal say (1:1) regardless."""
    case = {"A": {"fold0": 0.10, "fold1": 0.20, "fold2": 0.30}, "B": {"fold0": 0.90}}
    flat = agg._case_flat_value(case)                    # label=None -> pool all labels
    # cross_fold_class_mean's convention: mean of PER-FOLD means (each fold's mean pools
    # whichever labels have a value in that fold) — fold0 mixes A and B (mean 0.5), fold1/2
    # are A-only (0.2, 0.3); NOT a naive flat mean of the 4 raw numbers (0.375).
    flat_pool_expected = np.mean([np.mean([0.10, 0.90]), 0.20, 0.30])
    equal_weight_wrong = np.mean([np.mean([0.10, 0.20, 0.30]), 0.90])  # the regression's answer
    assert np.isclose(flat, flat_pool_expected), (flat, flat_pool_expected)
    assert not np.isclose(flat, equal_weight_wrong), "still averaging per-label instead of flat-pooling"
    # explicit label is untouched either way (one value per fold, degenerates to a plain mean)
    assert np.isclose(agg._case_flat_value(case, "A"), np.mean([0.10, 0.20, 0.30]))
    # end-to-end via _leaf_entries / resolve_group_entries on a label=None leaf
    R = {"col": {"p1": case}}
    C = {"col": {"p1": {"A": {"fold0": 0.0, "fold1": 0.0, "fold2": 0.0}, "B": {"fold0": 0.0}}}}
    ents = agg.resolve_group_entries("col", R, C)
    assert len(ents) == 1 and np.isclose(ents[0][2], flat_pool_expected), ents
    print(f"ok 6  label=None leaf flat-pools raw values ({flat:.4f}) not per-label-equal-weight ({equal_weight_wrong:.4f})")


if __name__ == "__main__":
    for t in (test_flat_matches_legacy, test_variance_matches_monte_carlo, test_tail_exact_and_monte_carlo,
              test_more_cases_shrink_variance, test_patient_units_merge, test_point_estimate_untouched,
              test_label_flat_pool_not_equal_weight):
        t()
    print("all significance-unit checks passed")
