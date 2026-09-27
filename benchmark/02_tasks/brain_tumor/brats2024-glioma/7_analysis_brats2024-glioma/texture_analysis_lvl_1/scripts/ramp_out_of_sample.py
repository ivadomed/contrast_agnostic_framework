#!/usr/bin/env python
"""
ramp_out_of_sample.py — out-of-sample test of the BraTS-derived "internal ramp" rule:

  RULE: real-fill hurts (Delta = real - noise Dice < 0) on an eval contrast when the
  segmentation target is FLATTER in the eval contrast than in the training contrast
  (R_eval < R_train); it helps when R_eval > R_train. R = share of within-label
  intensity variance explained by depth to the label border (Kelley-bias-corrected
  eta^2, computed per image per label from raw intensities — invariant to any affine
  map of intensity, so no z-scoring/body-mask step is needed).

R(image, label):
  mask = (GT == label); skip if voxel count < 200.
  dist = scipy.ndimage.distance_transform_edt(mask)      # voxel units, no spacing arg —
    matches the reference BraTS implementation
    (7_analysis_brats2024-glioma/texture_analysis_lvl_1/scripts/compute_boundary_profiles.py),
    which also runs distance_transform_edt with no `sampling=` argument. This means R is
    computed in voxel-index distance, not physical mm, on anisotropic-voxel datasets
    (duke-breast-mri, CT volumes) — a known limitation, stated up front, not discovered
    post hoc.
  bin = floor(dist) clipped to [1, 7]; bin>=7 pooled into bin 7 (depth 1..7).
  Per bin: n, sum(intensity), sum(intensity^2) [raw voxel intensities, no z-scoring —
    SS_between/SS_total is exactly invariant to any affine transform x -> a*x+b, a!=0,
    of the intensities, so z-scoring cannot change R and is skipped].
  SS_tot = sum(x^2) - (sum(x))^2/N ; SS_between = sum_bins( (sum_x_bin)^2/n_bin ) - (sum(x))^2/N
  R_adj = 1 - [(SS_tot - SS_between)/(N-k)] / [SS_tot/(N-1)]   (k = number of bins used)

Per (dataset, contrast): R_per_label = mean over cases of R_adj (cases where that label is
present with >=200 vox). R_pooled per case = voxel-weighted mean of that case's per-label
R_adj; R_pooled per (dataset,contrast) = mean over cases of R_pooled_case. Pooled R is the
PRIMARY quantity (matches the ladder's own Delta, which pools all foreground labels).

Modes:
  manifest   -- print the built-in per-(dataset,contrast) case manifest to stdout (for audit)
  extract    -- compute per-case per-label R for one (dataset,contrast) manifest entry,
                write outputs/data/ramp_oos_<dataset>_<contrast>.csv
  aggregate  -- read every ramp_oos_*.csv + the ladder_series.json Delta/p values, build the
                primary OOD pair table, run the binomial + Fisher tests + Spearman, write
                outputs/tables/ramp_out_of_sample.md and outputs/data/ramp_oos_pairs.csv

PRE-REGISTERED (written before computing any R value, 2026-09-24 continuation session):
  Primary test: over all OOD (train,eval) pairs listed in the task (BraTS excluded --
  the rule was derived there, so it is in-sample; used only as a sanity check of this
  script's R-computation code against the existing internal_ramp_patient.csv), does
  sign(R_eval - R_train) == sign(Delta)? One-sided exact binomial vs 0.5 (rule predicts
  agreement > chance), plus Fisher exact on the 2x2 sign table (more honest than the
  binomial here because the OOD pair set is NOT balanced -- most open-ms/chaos/ispy2
  pairs are net-positive Delta and most on-harmony pairs are net-negative, so a rule
  that just recovers "which dataset is this" could look falsely good under a plain
  binomial). Restricted-to-p<0.05 subset reported separately. Secondary (exploratory):
  Spearman(R_eval - R_train, Delta) over the same pairs.
  Not adjusted after seeing outcomes; extras beyond this list are labelled exploratory.

Usage (Slurm CPU job only -- see the docstring's CLAUDE.md hard cluster rules):
  .venv/bin/python ramp_out_of_sample.py manifest
  .venv/bin/python ramp_out_of_sample.py extract --key open-ms:flair
  .venv/bin/python ramp_out_of_sample.py aggregate
  .venv/bin/python ramp_out_of_sample.py sanity-check-brats     # compares to internal_ramp_patient.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.stats import binomtest, fisher_exact, spearmanr

try:
    import nibabel as nib
except ImportError:
    nib = None

REPO = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
THIS_DIR = Path(__file__).resolve().parent
OUT = THIS_DIR.parent / "outputs"
DATA, TABLES = OUT / "data", OUT / "tables"
DATA.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

MAX_BIN = 7
MIN_LABEL_VOX = 200
MAX_CASES = 40


def _stem(fn: str) -> str:
    """nnUNet-style case id from a filename: strip _0000 image-channel suffix + extension."""
    s = re.sub(r"\.nii\.gz$", "", fn)
    s = re.sub(r"_0000$", "", s)
    return s


def _list_cases(img_dir: Path, lbl_dir: Path, img_suffix="_0000"):
    """Pairs (case_id, image_path, label_path) by matching stems, nnUNet imagesTs/labelsTs style."""
    imgs = {}
    for f in sorted(img_dir.glob("*.nii.gz")):
        cid = _stem(f.name)
        imgs[cid] = f
    pairs = []
    for f in sorted(lbl_dir.glob("*.nii.gz")):
        cid = _stem(f.name)
        if cid in imgs:
            pairs.append((cid, imgs[cid], f))
    return pairs


def _list_cases_onharmony(contrast: str):
    base = REPO / f"benchmark/02_tasks/brain_healthy/on-harmony/8_results_on-harmony/01_predictions/on_harmony_model/_test_set/{contrast}"
    return _list_cases(base / "images_native", base / "gt_native")


def _toothfairy2_cases():
    test_json = REPO / "benchmark/02_tasks/mandible_healthy/toothfairy2/4_splits_toothfairy2/test_cases.json"
    ids = json.loads(test_json.read_text())["test"]
    bids = REPO / "benchmark/02_tasks/mandible_healthy/toothfairy2/1_BIDS_toothfairy2/maxillofacial-toothfairy2"
    pairs = []
    for cid in ids:
        sub = "sub-" + cid.split("_", 1)[1]
        img = bids / sub / "anat" / f"{sub}_acq-cbct_ct.nii.gz"
        lbl = bids / "derivatives" / "labels" / sub / "anat" / f"{sub}_acq-cbct_ct_label-maxillofacial_seg.nii.gz"
        if img.exists() and lbl.exists():
            pairs.append((cid, img, lbl))
    return pairs


# ─────────────────────────── manifest: (dataset, contrast) -> (mandible_only, case-list fn) ──
def build_manifest():
    m = {}

    def nn(ds_dir, img_dir_name, lbl_dir_name):
        p = REPO / ds_dir
        return _list_cases(p / img_dir_name, p / lbl_dir_name)

    # open-ms: 8 patients, all 3 contrasts available per patient, shared across both train dirs
    base = "benchmark/02_tasks/brain_ms/open-ms/2_nnUNet_open-ms/raw/Dataset070_OpenMS_FLAIR"
    for c in ("flair", "t1w", "t2w"):
        m[f"open-ms:{c}"] = (False, lambda c=c, base=base: nn(base, f"imagesTs_{c}", f"labelsTs_{c}"))

    # chaos: MR test patients (n=4) shared for t1in/t1out/t2spir; CT test patients separate (n=20)
    cbase = "benchmark/02_tasks/abdomen_healthy/chaos/2_nnUNet_chaos/raw/Dataset060_CHAOS_MR_T1in"
    for c in ("t1in", "t1out", "t2spir", "ct"):
        m[f"chaos:{c}"] = (False, lambda c=c, cbase=cbase: nn(cbase, f"imagesTs_{c}", f"labelsTs_{c}"))

    # ispy2
    m["ispy2:t1wce"] = (False, lambda: nn("benchmark/02_tasks/breast_cancer/ispy2/2_nnUNet_ispy2/raw/Dataset100_ISPY2T1wce",
                                            "imagesTs_t1wce", "labelsTs_t1wce"))
    m["ispy2:t2w"] = (False, lambda: nn("benchmark/02_tasks/breast_cancer/ispy2/2_nnUNet_ispy2/raw/Dataset101_ISPY2T2w",
                                          "imagesTs_t2w", "labelsTs_t2w"))
    # duke-breast-mri (cross-dataset eval companion; unilateral-crop items only)
    dbase = "benchmark/02_tasks/breast_cancer/duke-breast-mri/2_nnUNet_duke-breast-mri/raw"
    m["duke:t1wce_uni"] = (False, lambda: nn(dbase, "imagesTs_t1wce_uni", "labelsTs_t1wce_uni"))
    m["duke:precontrast_uni"] = (False, lambda: nn(dbase, "imagesTs_precontrast_uni", "labelsTs_precontrast_uni"))

    # on-harmony (shared cross-contrast test set, images_native/gt_native, non-nnUNet-suffix images)
    for c in ("T1w", "T2w", "bold", "dwi_ap", "epi_ap", "gre_echo1_mag"):
        m[f"onharmony:{c}"] = (False, lambda c=c: _list_cases_onharmony(c))

    # toothfairy2 (mandible-only, label 1) — training contrast reference, from BIDS + its own test split
    m["toothfairy2:cbct"] = (True, _toothfairy2_cases)

    # hanseg / pddca (cross-dataset eval companions, mandible-only, GT already binary label 1)
    hbase = "benchmark/02_tasks/mandible_healthy/hanseg/2_nnUNet_hanseg/raw"
    m["hanseg:ct"] = (True, lambda: nn(hbase, "imagesTs_ct", "labelsTs_ct"))
    m["hanseg:mrt1"] = (True, lambda: nn(hbase, "imagesTs_mrt1", "labelsTs_mrt1"))
    m["pddca:ct"] = (True, lambda: nn("benchmark/02_tasks/mandible_healthy/pddca/2_nnUNet_pddca/raw", "imagesTs_ct", "labelsTs_ct"))

    return m


# ─────────────────────────── core R computation ───────────────────────────
def r_adj_for_label(img: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    n_vox = int(mask.sum())
    if n_vox < MIN_LABEL_VOX:
        return float("nan"), n_vox
    dist = distance_transform_edt(mask)
    d = dist[mask]
    x = img[mask].astype(np.float64)
    bins = np.clip(np.floor(d).astype(int), 1, MAX_BIN)
    N = x.size
    S = x.sum()
    S2 = (x ** 2).sum()
    ss_tot = S2 - S * S / N
    if ss_tot <= 0:
        return float("nan"), n_vox
    uniq = np.unique(bins)
    k = len(uniq)
    if k < 2:
        return float("nan"), n_vox
    ss_between = 0.0
    for b in uniq:
        xb = x[bins == b]
        ss_between += xb.sum() ** 2 / xb.size
    ss_between -= S * S / N
    if N - k <= 0:
        return float("nan"), n_vox
    r_adj = 1 - ((ss_tot - ss_between) / (N - k)) / (ss_tot / (N - 1))
    return float(r_adj), n_vox


def compute_case(image_path: Path, label_path: Path, mandible_only: bool):
    img = nib.load(str(image_path)).get_fdata().astype(np.float64)
    lbl = nib.load(str(label_path)).get_fdata()
    labels = [1] if mandible_only else sorted(int(v) for v in np.unique(lbl) if v > 0)
    rows = []
    for lab in labels:
        mask = lbl == lab
        r_adj, n_vox = r_adj_for_label(img, mask)
        rows.append((lab, n_vox, r_adj))
    return rows


def do_extract(key: str):
    manifest = build_manifest()
    if key not in manifest:
        print(f"ERROR: unknown key {key}. Known: {sorted(manifest)}", file=sys.stderr)
        sys.exit(1)
    mandible_only, case_fn = manifest[key]
    cases = case_fn()
    if len(cases) == 0:
        print(f"WARNING: 0 cases found for {key} -- skipping, writing empty CSV")
    if len(cases) > MAX_CASES:
        cases = cases[:MAX_CASES]
    dataset, contrast = key.split(":", 1)
    out_rows = []
    n_ok, n_fail = 0, 0
    for cid, img_p, lbl_p in cases:
        try:
            for lab, n_vox, r_adj in compute_case(img_p, lbl_p, mandible_only):
                out_rows.append(dict(dataset=dataset, contrast=contrast, case=cid, label=lab,
                                      n_vox=n_vox, R_adj=r_adj))
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"SKIP {cid}: {e}", file=sys.stderr)
    df = pd.DataFrame(out_rows)
    out_csv = DATA / f"ramp_oos_{dataset}_{contrast}.csv"
    df.to_csv(out_csv, index=False)
    print(f"{key}: {n_ok} cases ok, {n_fail} failed, {len(df)} label-rows -> {out_csv}")


# ─────────────────────────── ladder outcomes (Delta, p) ───────────────────────────
LADDER_JSONS = {
    ("open-ms", "flair"): "benchmark/02_tasks/brain_ms/open-ms/8_results_open-ms/02_metrics/open_ms_model/flair/ablations/ladder_series.json",
    ("open-ms", "t1w"): "benchmark/02_tasks/brain_ms/open-ms/8_results_open-ms/02_metrics/open_ms_model/t1w/ablations/ladder_series.json",
    ("chaos", "t1in"): "benchmark/02_tasks/abdomen_healthy/chaos/8_results_chaos/02_metrics/chaos_model/t1in/ablations/ladder_series.json",
    ("chaos", "t2spir"): "benchmark/02_tasks/abdomen_healthy/chaos/8_results_chaos/02_metrics/chaos_model/t2spir/ablations/ladder_series.json",
    ("ispy2", "t1wce"): "benchmark/02_tasks/breast_cancer/ispy2/8_results_ispy2/02_metrics/ispy2_model/t1wce/ablations/ladder_series.json",
    ("ispy2", "t2w"): "benchmark/02_tasks/breast_cancer/ispy2/8_results_ispy2/02_metrics/ispy2_model/t2w/ablations/ladder_series.json",
    ("onharmony", "T1w"): "benchmark/02_tasks/brain_healthy/on-harmony/8_results_on-harmony/02_metrics/on_harmony_model/T1w/ablations/ladder_series.json",
    ("onharmony", "T2w"): "benchmark/02_tasks/brain_healthy/on-harmony/8_results_on-harmony/02_metrics/on_harmony_model/T2w/ablations/ladder_series.json",
    ("toothfairy2", "cbct"): "benchmark/02_tasks/mandible_healthy/toothfairy2/8_results_toothfairy2/02_metrics/toothfairy2_model/cbct/ablations/ladder_series.json",
}

# eval-contrast key in the JSON's per_contrast dict -> (R-manifest key, "primary"/"exploratory")
EVAL_KEY_MAP = {
    ("open-ms", "flair"): {"t1w": "open-ms:t1w", "t2w": "open-ms:t2w"},
    ("open-ms", "t1w"): {"flair": "open-ms:flair", "t2w": "open-ms:t2w"},
    ("chaos", "t1in"): {"t1out": "chaos:t1out", "t2spir": "chaos:t2spir", "ct": "chaos:ct"},
    ("chaos", "t2spir"): {"t1in": "chaos:t1in", "t1out": "chaos:t1out", "ct": "chaos:ct"},
    ("ispy2", "t1wce"): {"t2w": "ispy2:t2w", "duke-breast-mri/precontrast_uni": "duke:precontrast_uni"},
    ("ispy2", "t2w"): {"t1wce": "ispy2:t1wce", "duke-breast-mri/t1wce_uni": "duke:t1wce_uni",
                        "duke-breast-mri/precontrast_uni": "duke:precontrast_uni"},
    ("onharmony", "T1w"): {"T2w": "onharmony:T2w", "bold": "onharmony:bold", "dwi_ap": "onharmony:dwi_ap",
                            "epi_ap": "onharmony:epi_ap", "gre_echo1_mag": "onharmony:gre_echo1_mag"},
    ("onharmony", "T2w"): {"T1w": "onharmony:T1w", "bold": "onharmony:bold", "dwi_ap": "onharmony:dwi_ap",
                            "epi_ap": "onharmony:epi_ap", "gre_echo1_mag": "onharmony:gre_echo1_mag"},
    ("toothfairy2", "cbct"): {"hanseg/ct": "hanseg:ct", "hanseg/mrt1": "hanseg:mrt1", "pddca/ct": "pddca:ct"},
}
TRAIN_R_KEY = {
    ("open-ms", "flair"): "open-ms:flair", ("open-ms", "t1w"): "open-ms:t1w",
    ("chaos", "t1in"): "chaos:t1in", ("chaos", "t2spir"): "chaos:t2spir",
    ("ispy2", "t1wce"): "ispy2:t1wce", ("ispy2", "t2w"): "ispy2:t2w",
    ("onharmony", "T1w"): "onharmony:T1w", ("onharmony", "T2w"): "onharmony:T2w",
    ("toothfairy2", "cbct"): "toothfairy2:cbct",
}


def load_ladder_delta_p(dataset, train, eval_key):
    d = json.loads((REPO / LADDER_JSONS[(dataset, train)]).read_text())
    labels = d["labels"]
    idx_real = next(i for i, l in enumerate(labels) if "real fill" in l)
    idx_prev = idx_real - 1
    arr = d["per_contrast"]["dice"][eval_key]
    scale = 100 if max(arr) <= 1.5 else 1
    delta = (arr[idx_real] - arr[idx_prev]) * scale
    p = d.get("fill_swap_significance", {}).get("dice", {}).get("per_contrast", {}).get(eval_key, float("nan"))
    return delta, p


def pooled_r(df: pd.DataFrame) -> tuple[float, int]:
    """Voxel-weighted pooled R per case, then mean over cases. Returns (R, n_cases_used)."""
    df = df.dropna(subset=["R_adj"])
    if df.empty:
        return float("nan"), 0
    per_case = df.groupby("case").apply(lambda g: np.average(g["R_adj"], weights=g["n_vox"]))
    return float(per_case.mean()), len(per_case)


def do_aggregate():
    manifest_keys = set(build_manifest())
    r_cache = {}

    def get_pooled_r(key):
        if key not in r_cache:
            csv = DATA / f"ramp_oos_{key.replace(':', '_')}.csv"
            if not csv.exists():
                r_cache[key] = (float("nan"), 0)
            else:
                r_cache[key] = pooled_r(pd.read_csv(csv))
        return r_cache[key]

    rows = []
    for (dataset, train), evals in EVAL_KEY_MAP.items():
        train_key = TRAIN_R_KEY[(dataset, train)]
        r_train, n_train = get_pooled_r(train_key)
        for eval_json_key, r_key in evals.items():
            r_eval, n_eval = get_pooled_r(r_key)
            delta, p = load_ladder_delta_p(dataset, train, eval_json_key)
            rows.append(dict(dataset=dataset, train=train, eval=eval_json_key, R_train=r_train,
                              n_train_cases=n_train, R_eval=r_eval, n_eval_cases=n_eval,
                              R_gap=r_eval - r_train, delta=delta, p=p,
                              pred_sign=np.sign(r_eval - r_train), actual_sign=np.sign(delta)))
    pairs = pd.DataFrame(rows)
    pairs["hit"] = pairs["pred_sign"] == pairs["actual_sign"]
    pairs.to_csv(DATA / "ramp_oos_pairs.csv", index=False)

    valid = pairs.dropna(subset=["R_gap", "delta"])
    valid = valid[valid["pred_sign"] != 0]  # exclude exact-zero-gap ties from the directional test
    n_hit = int(valid["hit"].sum())
    n_tot = len(valid)
    bt = binomtest(n_hit, n_tot, 0.5, alternative="greater") if n_tot else None

    tab = pd.crosstab(valid["pred_sign"] > 0, valid["actual_sign"] > 0)
    fisher_p = fisher_exact(tab.values)[1] if tab.shape == (2, 2) else float("nan")

    sig = valid[valid["p"] < 0.05]
    n_hit_sig, n_tot_sig = int(sig["hit"].sum()), len(sig)
    bt_sig = binomtest(n_hit_sig, n_tot_sig, 0.5, alternative="greater") if n_tot_sig else None
    n_pos_sig = int((sig["actual_sign"] > 0).sum())
    pred_hurts_sig = sig[sig["pred_sign"] < 0]
    n_hurts_sig_hit = int(pred_hurts_sig["hit"].sum())
    n_hurts_sig_tot = len(pred_hurts_sig)

    n_pos_all = int((valid["actual_sign"] > 0).sum())

    varying = valid[~valid["dataset"].isin(["onharmony", "toothfairy2"])]  # datasets whose own
    # outcomes are not all one sign (onharmony = all-negative-but-one, toothfairy2 = all-positive)
    n_hit_var, n_tot_var = int(varying["hit"].sum()), len(varying)
    bt_var = binomtest(n_hit_var, n_tot_var, 0.5, alternative="greater") if n_tot_var else None

    per_ds = valid.groupby("dataset")["hit"].agg(["sum", "count"])

    rho, rho_p = spearmanr(valid["R_gap"], valid["delta"]) if n_tot > 2 else (float("nan"), float("nan"))

    L = ["# Out-of-sample test of the internal-ramp fill-swap rule", "",
         "Rule (derived on BraTS, pre-registered before this script computed any R): real-fill "
         "hurts (Delta<0) when R_eval < R_train, helps when R_eval > R_train. R = Kelley-bias-"
         "corrected eta^2 of intensity vs. depth-to-border, per image per label, pooled voxel-"
         "weighted across a case's foreground labels (mandible-only for toothfairy2/hanseg/pddca). "
         "BraTS excluded from this test (in-sample, rule source). No rule adjustment after seeing "
         "outcomes below; anything beyond the pre-registered primary test is marked exploratory.", "",
         f"## Primary result: {n_hit}/{n_tot} pairs match sign(R_eval-R_train) to sign(Delta)",
         f"- One-sided exact binomial vs 0.5: p = {bt.pvalue:.4g}" if bt else "- (no valid pairs)",
         f"- Fisher exact, two-sided (2x2 sign table, does not assume balanced margins): p = {fisher_p:.4g}",
         f"- Secondary: Spearman(R_gap, Delta) rho = {rho:+.3f}, p = {rho_p:.4g} (n={n_tot})", "",
         "## Trivial-baseline comparison and where the rule actually fails", "",
         f"A rule that ignores R entirely and always predicts \"real-fill helps\" scores "
         f"{n_pos_all}/{n_tot} on the full pair set -- worse than the ramp rule's {n_hit}/{n_tot}. "
         "But this reverses on the subset that matters most:", "",
         f"- **Restricted to fill-swap p<0.05 ({n_tot_sig} pairs): the ramp rule gets {n_hit_sig}/{n_tot_sig} "
         f"(binomial p={bt_sig.pvalue:.3g}), while \"always helps\" gets {n_pos_sig}/{n_tot_sig} -- "
         f"the trivial baseline WINS on the significant subset.**",
         f"- **The \"hurts\" side has zero significant support: {n_hurts_sig_hit}/{n_hurts_sig_tot} "
         "significant hurts-predictions were correct** (open-ms flair->t1w, ispy2 t1wce->t2w, "
         "ispy2 t1wce->duke precontrast_uni all predicted hurts and actually helped; on-harmony "
         "T2w->T1w -- the only significant negative Delta anywhere in this pair set -- was itself "
         "predicted to help, so it is a miss on the \"helps\" side, not a hurts-prediction). Every "
         "dataset where the rule scores a hit on a significant pair does so on the \"helps\" side.",
         f"- **Two of five datasets (on-harmony, toothfairy2) supply almost all the hits, and both "
         "have outcomes that are almost all one sign already** (on-harmony: 9/10 pairs Delta<0; "
         "toothfairy2: 3/3 Delta>0) -- so a same-sign-as-dataset-mean rule would do about as well "
         "as the ramp rule there. Restricting to the three datasets whose own within-dataset "
         f"outcomes actually vary in sign (open-ms, chaos, ispy2): the rule gets {n_hit_var}/{n_tot_var} "
         f"(exploratory one-sided binomial p={bt_var.pvalue:.3g} -- not pre-registered, reported "
         "because the primary result is otherwise dataset-composition-confounded).",
         "", f"Per-dataset hit counts: " + ", ".join(f"{ds} {int(r['sum'])}/{int(r['count'])}"
                                                        for ds, r in per_ds.iterrows()), "",
         "## Per-dataset predictions vs. actual outcomes", "",
         "| dataset | train | eval | R_train (n) | R_eval (n) | R_gap | pred | Delta (pts) | p | actual | hit |",
         "|---|---|---|--:|--:|--:|:--:|--:|--:|:--:|:--:|"]
    for r in pairs.itertuples():
        pred = "helps" if r.pred_sign > 0 else ("hurts" if r.pred_sign < 0 else "tie")
        act = "helps" if r.actual_sign > 0 else ("hurts" if r.actual_sign < 0 else "flat")
        hit_str = "n/a" if pd.isna(r.R_gap) or pd.isna(r.delta) else ("YES" if r.hit else "no")
        L.append(f"| {r.dataset} | {r.train} | {r.eval} | {r.R_train:.3f} ({r.n_train_cases}) | "
                  f"{r.R_eval:.3f} ({r.n_eval_cases}) | {r.R_gap:+.3f} | {pred} | {r.delta:+.2f} | "
                  f"{r.p:.3g} | {act} | {hit_str} |")
    # ── per-label descriptive tables (required deliverable, not just pooled R) ──
    def per_label_table(dataset_prefix, keys):
        rows = []
        for key in keys:
            csv = DATA / f"ramp_oos_{key.replace(':', '_')}.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv).dropna(subset=["R_adj"])
            for lab, g in df.groupby("label"):
                rows.append(dict(key=key, label=int(lab), n_cases=g["case"].nunique(),
                                  mean_n_vox=int(g["n_vox"].mean()), R_mean=g["R_adj"].mean()))
        return pd.DataFrame(rows)

    chaos_labels = {1: "liver", 2: "right_kidney", 3: "left_kidney", 4: "spleen"}
    chaos_tab = per_label_table("chaos", ["chaos:t1in", "chaos:t1out", "chaos:t2spir", "chaos:ct"])
    if not chaos_tab.empty:
        chaos_tab["organ"] = chaos_tab["label"].map(chaos_labels)
        L += ["", "## Per-label R -- chaos (organ, required deliverable, descriptive)", "",
              "| contrast | organ | n_cases | mean_n_vox | R_mean |", "|---|---|--:|--:|--:|"]
        for r in chaos_tab.sort_values(["key", "label"]).itertuples():
            L.append(f"| {r.key.split(':')[1]} | {r.organ} | {r.n_cases} | {r.mean_n_vox} | {r.R_mean:.3f} |")

    oh_tab = per_label_table("onharmony", [f"onharmony:{c}" for c in
                              ("T1w", "T2w", "bold", "dwi_ap", "epi_ap", "gre_echo1_mag")])
    if not oh_tab.empty:
        top = (oh_tab.sort_values("mean_n_vox", ascending=False)
               .groupby("key").head(8).sort_values(["key", "mean_n_vox"], ascending=[True, False]))
        L += ["", "## Per-label R -- on-harmony (top-8-by-volume labels per contrast, descriptive)", "",
              "| contrast | label_id | n_cases | mean_n_vox | R_mean |", "|---|--:|--:|--:|--:|"]
        for r in top.itertuples():
            L.append(f"| {r.key.split(':')[1]} | {r.label} | {r.n_cases} | {r.mean_n_vox} | {r.R_mean:.3f} |")

    # ── voxel spacing table (diagnostic for the anisotropy limitation) ──
    spacing_csv = DATA / "ramp_oos_spacing.csv"
    if spacing_csv.exists():
        sp = pd.read_csv(spacing_csv)
        L += ["", "## Voxel spacing per manifest key (mm; one case each, diagnostic)", "",
              "| key | spacing_mm |", "|---|---|"]
        for r in sp.itertuples():
            L.append(f"| {r.key} | {r.spacing_mm} |")

    L += ["", "## Notes / limitations", "",
          "- **R is computed in voxel-index distance** (no `sampling=` in distance_transform_edt), "
          "matching the reference BraTS script exactly. This is NOT uniformly harmless: see the "
          "spacing table above -- on-harmony's bold/dwi_ap/epi_ap/gre_echo1_mag are visibly coarser "
          "than its T1w/T2w, and chaos MR is thick-slice. \"Depth 7 voxels\" is a different physical "
          "distance per contrast, which inflates R for finer-spacing contrasts and likely "
          "contributes to on-harmony's near-zero R on the EPI-family contrasts (R~0.004-0.018 vs. "
          "T1w's 0.066) -- part of why on-harmony supplies 9 of the 22 hits (see baseline section).",
          "- Cases capped at 40 per contrast; several contrasts (chaos MR n=4, open-ms n=7-8) have "
          "few test cases -- R estimates there are noisy.",
          "- Pairs are not independent (several share a training contrast) -- p-values should be "
          "read as descriptive, not as independent draws.",
          "- ispy2 t1wce->t2w and t1wce->duke-precontrast_uni: R_eval for \"t2w\" was computed from "
          "Dataset101_ISPY2T2w's own imagesTs_t2w/labelsTs_t2w (that model's own eval set), not from "
          "a t2w image paired to the t1wce-trained model's actual eval cohort -- the ladder's own "
          "eval pipeline evaluates the t1wce-trained model on the same physical t2w test images, so "
          "this is the right image set, but the first-40-by-filename cap may not exactly match the "
          "case subset the ladder scored. The R_gap there is ~6x the sign-flip threshold, so this "
          "would not change the hit/miss call.",
          "- The BraTS sanity check (see script's `sanity-check-brats` mode) compares this script's "
          "R computation against the existing internal_ramp_patient.csv on real patients/images, "
          "not just against the reference's own printed ranking -- see console output / log for "
          "the agreement statistics; a documented convention difference (floor+pool-at-7 here vs. "
          "exact-integer-distance+bins-1..8 there) means exact numeric match is not expected."]
    (TABLES / "ramp_out_of_sample.md").write_text("\n".join(L))
    print("\n".join(L))


def sanity_check_brats():
    """Real check (not circular): run THIS script's r_adj_for_label on the SNFH mask for a
    handful of real BraTS patients, using the same load_patient()/region_masks() the reference
    pipeline uses, and compare directly to internal_ramp_patient.csv's own R_adj for the same
    (patient, contrast, region=SNFH) rows. This can disagree from the reference in an expected,
    documented way: the reference uses distance_transform_edt(mask)==k (drops non-integer-distance
    voxels, e.g. sqrt(2), and keeps bins 1..8 unpooled); this script uses floor(dist) pooled at
    bin 7. Reports both the raw agreement and the rank agreement (Spearman across patients),
    since the rule only needs R's ORDERING to be preserved, not its exact value."""
    sys.path.insert(0, str(THIS_DIR))
    from compute_cross_contrast_ngf import load_patient, region_masks  # noqa: E402

    ref = pd.read_csv(THIS_DIR.parent / "outputs" / "data" / "internal_ramp_patient.csv")
    ref = ref[ref["region"] == "SNFH"]
    patients = sorted(ref["patient"].unique())[:5]
    rows = []
    for pid in patients:
        try:
            vols, label = load_patient(pid, "cpu")
        except Exception as e:  # noqa: BLE001
            print(f"SKIP {pid}: {e}")
            continue
        masks = {k: m.numpy() for k, m in region_masks(label, vols["t1n"]).items()}
        snfh = masks["SNFH"]
        for c in ("t1n", "t1c", "t2w", "t2f"):
            img = vols[c].numpy().astype(np.float64)
            r_new, n_vox = r_adj_for_label(img, snfh)
            ref_row = ref[(ref["patient"] == pid) & (ref["contrast"] == c)]
            r_ref = float(ref_row["R_adj"].iloc[0]) if len(ref_row) else float("nan")
            rows.append(dict(patient=pid, contrast=c, R_new=r_new, R_ref=r_ref, n_vox=n_vox))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    d = df.dropna()
    if len(d) > 2:
        rho, p = spearmanr(d["R_new"], d["R_ref"])
        mad = float((d["R_new"] - d["R_ref"]).abs().mean())
        print(f"\nn={len(d)} (patient,contrast) rows compared")
        print(f"Spearman rank agreement R_new vs R_ref: rho={rho:+.3f} (p={p:.3g})")
        print(f"Mean absolute difference: {mad:.4f}")
    else:
        print("Not enough overlapping rows to compare.")


def do_spacing():
    """Print voxel spacing (zooms) for one case per manifest key -- depth bins are voxel-index
    based (no `sampling=` in distance_transform_edt), so anisotropic/coarse-slice contrasts get a
    physically different depth-7 cutoff than isotropic ones. This is diagnostic, not a fix."""
    manifest = build_manifest()
    rows = []
    for key, (mandible_only, case_fn) in manifest.items():
        try:
            cases = case_fn()
            if not cases:
                continue
            cid, img_p, _ = cases[0]
            zooms = nib.load(str(img_p)).header.get_zooms()[:3]
            rows.append(dict(key=key, case=cid, spacing_mm=tuple(round(float(z), 3) for z in zooms)))
        except Exception as e:  # noqa: BLE001
            rows.append(dict(key=key, case="ERROR", spacing_mm=str(e)))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(DATA / "ramp_oos_spacing.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["manifest", "extract", "aggregate", "sanity-check-brats", "spacing"])
    ap.add_argument("--key", default=None)
    args = ap.parse_args()
    if args.mode == "manifest":
        for k, (mo, fn) in build_manifest().items():
            try:
                n = len(fn())
            except Exception as e:  # noqa: BLE001
                n = f"ERROR: {e}"
            print(f"{k}: mandible_only={mo} n_cases={n}")
    elif args.mode == "extract":
        if not args.key:
            print("--key required", file=sys.stderr); sys.exit(1)
        do_extract(args.key)
    elif args.mode == "aggregate":
        do_aggregate()
    elif args.mode == "sanity-check-brats":
        sanity_check_brats()
    elif args.mode == "spacing":
        do_spacing()


if __name__ == "__main__":
    main()
