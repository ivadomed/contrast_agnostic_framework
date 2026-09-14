#!/usr/bin/env python3
"""
Create the ispy2 train/test partition and CV splits — PATIENT-LEVEL and
FOV-STRATIFIED.

ispy2 is the PRIMARY TRAINING set for the breast-cancer task as of the 2026-09-04
pivot (it replaces ambl, whose 99 patients overfit). Two training modalities
(t1wce -> Dataset100, t2w -> Dataset101) share ONE patient-level partition, exactly
as ambl/chaos/brats2024-glioma do.

WHAT IS DIFFERENT HERE: THE FOV AXIS
------------------------------------
Every patient contributes MORE THAN ONE nnU-Net case, because 01_02_derive_fov_
variants.py builds both field-of-view variants of each contrast (see
4_splits_ispy2/fov_variants.json, the single source of truth this script reads —
it never re-derives the case list):

  natively-BILATERAL patient (122)   t1wce: bil + uni   t2w: bil + uni   -> 4 cases
  natively-UNILATERAL patient (438)  t1wce: uni only    t2w: bil + uni   -> 3 cases
                                     (no bilateral DCE exists in TCIA for them)

Case ids are `ispy2_<PID>_<bil|uni>`. THE HARD REQUIREMENT is that all of a
patient's cases — both FOVs, both contrasts — land in the SAME split. Splitting is
therefore done over PATIENTS and only then expanded to case ids; a `straddle` assert
at the end re-checks it from the emitted case lists themselves rather than trusting
the construction.

STRATIFICATION (two keys)
-------------------------
1. LATERALITY CLASS (uni / bil). This is not cosmetic: the 122 bilateral patients
   are the ONLY source of `t1wce/bil` cases. If they clustered into one fold, other
   folds would contain no bilateral T1wce at all and the "each split receives both
   FOVs" requirement would silently fail. Patients are split within each class and
   the per-class folds are then merged, so every fold and the test set get a
   proportional share.
2. TUMOUR BURDEN, within each class — lesion volume in mm^3 from the patient's own
   T1wce mask (the collection's real DICOM-SEG; mm^3 rather than voxel count because
   voxel sizes vary widely across this multi-site cohort). Test patients are taken at
   evenly-spaced burden ranks so the held-out set spans low->high burden, same
   approach as ambl/atlas-liver-hcc.

FOLDS: 3, not 4. CLAUDE.md's fold policy is "folds 0 1 2 only, permanently"; the
shared drivers (`train_common.sh` TRAIN_FOLDS, `predict_common.sh` PREDICT_FOLDS,
`eval_folds.py`) all default to `0 1 2`. Older datasets carry a 4-fold file whose
fold 3 is never trained — for a NEW split there is no reason to emit a fold nobody
will ever run, so this writes exactly 3.

OUTPUTS (4_splits_ispy2/)
  partition.json      {"train_pool_patients", "test_patients", per-patient laterality
                       + burden, and the expanded per-dataset case lists}
  splits_final.json   3-fold CV over the TRAIN-POOL CASES of each nnUNet dataset
                      (written once per dataset id, via splits_lib.write_splits_final)
  test_cases.json     {"test": [...case ids...], "test_patients": [...]}
  splits_report.md    fold sizes, FOV balance per split, straddle check

Run:  bash 01_create_splits/01_03_create_splits.sh
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib

DATASET_ROOT = Path(__file__).resolve().parents[2]              # datasets/ispy2
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_00_utils"))
from splits_lib import kfold_splits, write_splits_final          # noqa: E402

SEED = 1337
TEST_FRACTION = 0.15
N_FOLDS = 3

BIDS_ROOT = DATASET_ROOT / "1_BIDS_ispy2" / "breast-ispy2"
DERIV_ROOT = BIDS_ROOT / "derivatives" / "labels"
SPLITS = DATASET_ROOT / "4_splits_ispy2"
NNUNET_PRE = DATASET_ROOT / "2_nnUNet_ispy2" / "preprocessed"
NNUNET_DATASETS = {"t1wce": "Dataset100_ISPY2T1wce", "t2w": "Dataset101_ISPY2T2w"}
N_WORKERS = int(os.environ.get("ISPY2_FOV_WORKERS", "16"))


def case_id(pid: str, fov: str) -> str:
    return f"ispy2_{pid}_{fov}"


def _burden(sub: str) -> tuple[str, float]:
    """Lesion volume in mm^3 from the patient's native T1wce DICOM-SEG mask."""
    p = DERIV_ROOT / sub / "anat" / f"{sub}_acq-firstpost_T1w_label-lesion_seg.nii.gz"
    img = nib.load(str(p))
    n = int((np.asanyarray(img.dataobj) > 0).sum())
    vox = float(abs(np.linalg.det(np.asarray(img.affine)[:3, :3])))
    return sub, round(n * vox, 2)


def stratified_pick(pool: list[str], burden: dict[str, float], k: int) -> list[str]:
    """k patients at evenly-spaced tumour-burden ranks (deterministic; tops up on ties)."""
    by = sorted(pool, key=lambda s: (burden[s], s))
    idx = np.linspace(0, len(by) - 1, k).round().astype(int)
    picked = {by[i] for i in idx}
    i = 0
    while len(picked) < k and i < len(by):
        picked.add(by[i]); i += 1
    return sorted(picked)


def main() -> None:
    fov = json.loads((SPLITS / "fov_variants.json").read_text())
    patients = fov["patients"]
    subs = sorted(patients)
    print(f"{len(subs)} patients from fov_variants.json", flush=True)

    lat = {s: patients[s]["native_t1wce_fov"] for s in subs}       # 'uni' | 'bil'
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        burden = dict(ex.map(_burden, subs, chunksize=8))

    # ── patient-level test hold-out, stratified by class then burden ──────────
    rng = np.random.default_rng(SEED)
    test_patients: list[str] = []
    fold_val_patients: list[list[str]] = [[] for _ in range(N_FOLDS)]
    per_class = {}
    for cls in ("bil", "uni"):
        pool = [s for s in subs if lat[s] == cls]
        n_test = round(len(pool) * TEST_FRACTION)
        t = stratified_pick(pool, burden, n_test)
        test_patients += t
        rest = [s for s in pool if s not in set(t)]
        shuffled = list(rest); rng.shuffle(shuffled)
        # shared CV chunking (splits_lib) — applied WITHIN the class, then merged,
        # so every fold gets a proportional share of the 122 bilateral patients.
        cls_splits = kfold_splits(shuffled, N_FOLDS)
        for k in range(N_FOLDS):
            fold_val_patients[k] += cls_splits[k]["val"]
        per_class[cls] = {"n": len(pool), "n_test": len(t), "n_train_pool": len(rest)}
    test_patients = sorted(test_patients)
    train_pool_patients = sorted(s for s in subs if s not in set(test_patients))
    assert not (set(test_patients) & set(train_pool_patients))
    assert sorted(p for f in fold_val_patients for p in f) == train_pool_patients

    # ── expand patients -> nnU-Net case ids, per dataset ─────────────────────
    def cases_for(sub: str, contrast: str) -> list[str]:
        v = patients[sub]["variants"]
        return [case_id(patients[sub]["pid"], f)
                for f in ("bil", "uni") if f"{contrast}/{f}" in v]

    out_splits_paths = {}
    per_dataset = {}
    for contrast, ds_name in NNUNET_DATASETS.items():
        train_cases = sorted(c for s in train_pool_patients for c in cases_for(s, contrast))
        test_cases = sorted(c for s in test_patients for c in cases_for(s, contrast))
        splits = []
        for k in range(N_FOLDS):
            val_p = set(fold_val_patients[k])
            val = sorted(c for s in val_p for c in cases_for(s, contrast))
            train = sorted(c for s in train_pool_patients if s not in val_p
                           for c in cases_for(s, contrast))
            assert not (set(train) & set(val)), f"{ds_name} fold {k}: train/val overlap"
            assert not (set(train + val) & set(test_cases)), f"{ds_name} fold {k}: test contamination"
            # HARD REQUIREMENT: no patient's FOV variants straddle train/val — re-checked
            # from the emitted case ids, not assumed from how they were built.
            pat = lambda c: c.rsplit("_", 1)[0]                      # noqa: E731
            assert not ({pat(c) for c in train} & {pat(c) for c in val}), \
                f"{ds_name} fold {k}: a patient straddles train/val"
            splits.append({"train": train, "val": val})
        assert not ({c.rsplit('_', 1)[0] for c in train_cases}
                    & {c.rsplit('_', 1)[0] for c in test_cases}), \
            f"{ds_name}: a patient straddles train_pool/test"
        out_splits_paths[contrast] = write_splits_final(splits, SPLITS, NNUNET_PRE, ds_name)
        per_dataset[contrast] = {"dataset": ds_name, "train_cases": train_cases,
                                 "test_cases": test_cases,
                                 "splits": [{"n_train": len(s["train"]), "n_val": len(s["val"])}
                                            for s in splits]}
        # splits_final.json is per-dataset; keep a named copy so the second write
        # does not overwrite the first in 4_splits_ispy2/.
        (SPLITS / f"splits_final_{contrast}.json").write_text(json.dumps(splits, indent=2))

    SPLITS.mkdir(parents=True, exist_ok=True)
    (SPLITS / "partition.json").write_text(json.dumps({
        "seed": SEED, "n_folds": N_FOLDS, "test_fraction": TEST_FRACTION,
        "train_pool_patients": train_pool_patients, "test_patients": test_patients,
        "fold_val_patients": fold_val_patients,
        "laterality": lat, "tumour_volume_mm3": burden,
        "per_class": per_class,
        "nnunet_datasets": {c: d["dataset"] for c, d in per_dataset.items()},
        "cases": {c: {"train_pool": d["train_cases"], "test": d["test_cases"]}
                  for c, d in per_dataset.items()},
    }, indent=2))
    (SPLITS / "test_cases.json").write_text(json.dumps({
        "test_patients": test_patients,
        "test": {c: d["test_cases"] for c, d in per_dataset.items()},
    }, indent=2))

    _report(patients, lat, burden, train_pool_patients, test_patients,
            fold_val_patients, per_dataset)
    for c, p in out_splits_paths.items():
        print(f"-> {p} (+ copied into {NNUNET_PRE}/{NNUNET_DATASETS[c]})")


def _report(patients, lat, burden, train_pool, test, fold_val, per_dataset) -> None:
    def fovmix(subs, contrast):
        n_bil = sum(1 for s in subs if f"{contrast}/bil" in patients[s]["variants"])
        n_uni = sum(1 for s in subs if f"{contrast}/uni" in patients[s]["variants"])
        return n_bil, n_uni

    L = ["# ispy2 splits — patient-level, FOV-stratified\n",
         f"Patients: {len(train_pool)} train pool + {len(test)} held-out internal test "
         f"= {len(train_pool)+len(test)}\n",
         "Laterality class balance (the 122 bilateral patients are the ONLY source of "
         "`t1wce/bil` cases, so they are stratified across every split):\n",
         "| split | patients | natively bilateral | natively unilateral |",
         "|---|---|---|---|"]
    L.append(f"| test | {len(test)} | {sum(lat[s]=='bil' for s in test)} | "
             f"{sum(lat[s]=='uni' for s in test)} |")
    for k, fv in enumerate(fold_val):
        L.append(f"| fold{k} val | {len(fv)} | {sum(lat[s]=='bil' for s in fv)} | "
                 f"{sum(lat[s]=='uni' for s in fv)} |")
    L.append("")
    for contrast, d in per_dataset.items():
        L += [f"## {d['dataset']} ({contrast}) — nnU-Net cases\n",
              "| split | cases | bilateral-FOV | unilateral-FOV |", "|---|---|---|---|"]
        b, u = fovmix(test, contrast)
        L.append(f"| test | {len(d['test_cases'])} | {b} | {u} |")
        for k, s in enumerate(d["splits"]):
            vb, vu = fovmix(fold_val[k], contrast)
            tb, tu = fovmix([p for p in train_pool if p not in set(fold_val[k])], contrast)
            L.append(f"| fold{k} train | {s['n_train']} | {tb} | {tu} |")
            L.append(f"| fold{k} val | {s['n_val']} | {vb} | {vu} |")
        L.append("")
    bs = [burden[s] for s in test]; ba = [burden[s] for s in train_pool]
    L += ["## Tumour-burden balance (T1wce DICOM-SEG lesion volume, mm^3)\n",
          f"- test: median {np.median(bs):.0f}, p10 {np.percentile(bs,10):.0f}, "
          f"p90 {np.percentile(bs,90):.0f}",
          f"- train pool: median {np.median(ba):.0f}, p10 {np.percentile(ba,10):.0f}, "
          f"p90 {np.percentile(ba,90):.0f}\n",
          "## Straddle check\n",
          "Every patient's FOV variants (and both contrasts) are assigned as a unit; "
          "asserted from the emitted case lists for every fold and for train-pool vs "
          "test in both nnU-Net datasets. **No patient straddles a split.**\n"]
    (SPLITS / "splits_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
