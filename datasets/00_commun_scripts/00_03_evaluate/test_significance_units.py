#!/usr/bin/env python3
"""
Self-checks for the unit-level (patient) sign-flip design used by every macroΔ
significance test — stat_tests.py's design primitives + aggregate_from_config.build_design.
Synthetic data only, no dataset needed:

    .venv/bin/python datasets/00_commun_scripts/00_03_evaluate/test_significance_units.py

What each test proves (the third is the one that distinguishes a working fix from a
plausible-looking one — before 2026-09-21 a nested organ group collapsed to one element
per ORGAN, so no amount of extra cases could shrink its variance):
  1. flat, all-unique-unit data  -> same macroΔ and same null variance as the legacy `macro_perm`
  2. nested organ group          -> analytic sign-flip variance == Monte-Carlo sign-flip null
  2b. p-value TAIL               -> exact when few units; matches Monte-Carlo when 3 heavy units
                                    dominate (where a plain normal approximation is 10x+ off)
  3. more cases in a nested group-> its sign-flip SD SHRINKS (~1/sqrt(n)); a per-organ collapse can't
  4. same patient scored by two trained models / in two contrasts -> ONE unit
  5. the point estimate (macroΔ) is untouched by any of this
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


def _by_label(col_label_cases: dict) -> dict:
    """{(col, label): {tag: value}} -> the loaders' col -> label -> {tag: value} shape."""
    out = {}
    for (col, lab), cases in col_label_cases.items():
        out.setdefault(col, {})[lab] = dict(cases)
    return out


def _pair(rng, tags, delta, sd=0.05, col_lab=("c", "x")):
    """(ref, comp) by-label dicts with ref = comp + delta + noise over `tags`."""
    comp = {t: rng.uniform(0.5, 0.9) for t in tags}
    ref = {t: comp[t] + delta + rng.normal(0, sd) for t in tags}
    return _by_label({col_lab: ref}), _by_label({col_lab: comp})


def test_flat_matches_legacy():
    rng = np.random.default_rng(1)
    arrs, groups, R, C = [], {}, {}, {}
    for j, (n, delta) in enumerate([(8, 0.01), (20, 0.03), (40, -0.005)]):
        tags = [f"ds|p{j}_{i}" for i in range(n)]
        r, c = _pair(rng, tags, delta, col_lab=(f"col{j}", "x"))
        R.update(r); C.update(c); groups[f"g{j}"] = f"col{j}"
        arrs.append(np.array([r[f"col{j}"]["x"][t] - c[f"col{j}"]["x"][t] for t in tags]))
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


def _ct_like(rng, n_amos, n_chaos, delta=(0.04, 0.02, 0.03)):
    """Nested group: liver pools two sources, spleen/kidney from one; AMOS patients carry all 3 organs."""
    R, C = {}, {}
    for organ, d in zip(("liver", "spleen", "kidney"), delta):
        tags = [f"amos|a{i}" for i in range(n_amos)]
        r, c = _pair(rng, tags, d, col_lab=("amos_ct", organ)); R.setdefault("amos_ct", {}).update(r["amos_ct"]); C.setdefault("amos_ct", {}).update(c["amos_ct"])
    r, c = _pair(rng, [f"chaos|c{i}" for i in range(n_chaos)], delta[0], col_lab=("chaos_ct", "liver"))
    R.update(r); C.update(c)
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
            r, c = _pair(rng, [f"{mod}::{p}" for p in pats], d, col_lab=(col, "x"))
            R.setdefault(col, {}).setdefault("x", {}).update(r[col]["x"]); C.setdefault(col, {}).setdefault("x", {}).update(c[col]["x"])
    groups = {"t1in": "t1in", "t1out": "t1out"}
    e_pat, K = agg.build_design(groups, R, C, "patient")
    e_str, _ = agg.build_design(groups, R, C, "stratum")
    assert len({e[2] for e in e_pat}) == 4, "patient scope: 4 patients must be 4 units"
    assert len({e[2] for e in e_str}) == 8, "stratum scope: 4 patients x 2 strata must be 8 units"
    # merging the two trained models: 2 arrays (one per contrast) of 4 elements, not 8
    assert [a.size for a in stratum_arrays(e_pat)] == [4, 4]
    assert np.isclose(agg.resolve_group("t1in", R, C).mean(),
                      np.mean([R["t1in"]["x"][t] - C["t1in"]["x"][t] for t in R["t1in"]["x"]]))
    print("ok 4  8 model-tagged cases -> 4 patient units; 4 patients x 2 contrasts -> 4 (patient) / 8 (stratum) units")


def test_point_estimate_untouched():
    rng = np.random.default_rng(5)
    R, C, group = _ct_like(rng, 50, 20)
    entries, K = agg.build_design({"CT": group}, R, C, "patient")
    d = lambda col, lab, pool=None: np.array([R[col][lab][t] - C[col][lab][t] for t in R[col][lab]])
    liver = np.concatenate([d("amos_ct", "liver"), d("chaos_ct", "liver")])   # a list POOLS cases
    expect = np.mean([liver.mean(), d("amos_ct", "spleen").mean(), d("amos_ct", "kidney").mean()])
    got = macro_perm_design(entries, K, True)[0]
    assert np.isclose(got, expect, rtol=1e-12), (got, expect)
    print("ok 5  macroΔ = per-organ-then-cross-organ mean, unchanged:", round(got, 6))


if __name__ == "__main__":
    for t in (test_flat_matches_legacy, test_variance_matches_monte_carlo, test_tail_exact_and_monte_carlo,
              test_more_cases_shrink_variance,
              test_patient_units_merge, test_point_estimate_untouched):
        t()
    print("all significance-unit checks passed")
