#!/usr/bin/env python3
"""
Generate EM (GMM) pseudo-label maps for SynthSeg Mode B.

For each training case: run GMM with N_COMPONENTS on the raw T1w brain voxels
(masked by the FreeSurfer label mask), save as integer NIfTI in the same space.
Then create fold-specific symlink directories matching the existing synthseg_labels layout.

Usage:
    run_job --gpus 0 --slot 0 --wait -- .venv/bin/python scripts/nnunet_onharmony/00b_generate_modeB_labels.py
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.mixture import GaussianMixture

import os

DATASET_ROOT = Path(__file__).resolve().parents[3]   # datasets/on-harmony/
PROJECT_ROOT = DATASET_ROOT.parents[1]               # project root

BIDS = Path(os.environ.get("BIDS_ROOT", str(DATASET_ROOT / "1_BIDS_on-harmony")))
_nnunet_pre = Path(os.environ.get("nnUNet_preprocessed",
                                  str(DATASET_ROOT / "2_nnUNet_on-harmony/preprocessed")))
SPLITS_JSON = _nnunet_pre / "Dataset030_OnHarmonyT1w/splits_final.json"
FREESURFER_LABELS_ROOT = BIDS / "derivatives/synthseg_segs"
OUT_DERIVATIVES = BIDS / "derivatives/synthseg_modeB_labels"
SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", str(DATASET_ROOT / "4_splits_on-harmony")))

N_COMPONENTS = 32
N_JOBS = 8
RANDOM_STATE = 42


def get_t1w_path(sub: str, ses: str) -> Path:
    return BIDS / sub / ses / "anat" / f"{sub}_{ses}_T1w.nii.gz"


def get_freesurfer_label_path(sub: str, ses: str) -> Path:
    # synthseg_segs/{sub}/{ses}/{sub}_{ses}_T1w.nii.gz
    direct = FREESURFER_LABELS_ROOT / sub / ses / f"{sub}_{ses}_T1w.nii.gz"
    if direct.exists():
        return direct
    # Fallback: look in data/splits/synthseg_labels (any fold)
    for fold_dir in (ROOT / "data/splits/synthseg_labels").glob("fold_*"):
        for f in fold_dir.glob(f"{sub}_{ses}_*.nii.gz"):
            return f
    return None


def generate_em_labels(t1w_path: Path, fs_label_path: Path, out_path: Path) -> bool:
    if out_path.exists():
        return True

    t1w_img = nib.load(str(t1w_path))
    t1w = np.asarray(t1w_img.dataobj, dtype=np.float32)
    fs_labels = np.asarray(nib.load(str(fs_label_path)).dataobj, dtype=np.int32)

    brain_mask = fs_labels > 0
    if brain_mask.sum() < 1000:
        print(f"  WARNING: very few brain voxels for {out_path.name}")
        return False

    brain_voxels = t1w[brain_mask].reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type="full",
        max_iter=200,
        random_state=RANDOM_STATE,
        n_init=1,
    )
    gmm.fit(brain_voxels)
    labels_flat = gmm.predict(brain_voxels).astype(np.int32) + 1  # 1..N_COMPONENTS

    em_labels = np.zeros_like(fs_labels)
    em_labels[brain_mask] = labels_flat

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(em_labels, t1w_img.affine, t1w_img.header), str(out_path))
    return True


def main():
    splits = json.loads(SPLITS_JSON.read_text())

    # Collect all unique cases across all folds
    all_cases = set()
    for fold in splits:
        all_cases.update(fold["train"])
        all_cases.update(fold["val"])

    print(f"Generating EM labels for {len(all_cases)} cases with N={N_COMPONENTS} GMM components")

    generated = []
    for case_id in sorted(all_cases):
        # case_id = sub-XXXXX_ses-YYYYYY_T1w
        parts = case_id.split("_ses-")
        sub = parts[0]
        ses = "ses-" + parts[1].replace("_T1w", "")

        t1w_path = get_t1w_path(sub, ses)
        if not t1w_path.exists():
            print(f"  SKIP {case_id}: T1w not found at {t1w_path}")
            continue

        fs_label_path = get_freesurfer_label_path(sub, ses)
        if fs_label_path is None:
            print(f"  SKIP {case_id}: FreeSurfer labels not found")
            continue

        out_fname = f"{sub}_{ses}_T1w_emlabels.nii.gz"
        out_path = OUT_DERIVATIVES / sub / ses / out_fname

        print(f"  {case_id} ... ", end="", flush=True)
        ok = generate_em_labels(t1w_path, fs_label_path, out_path)
        if ok:
            print("done" if not out_path.exists() else "cached")
            generated.append((case_id, out_path))
        else:
            print("FAILED")

    print(f"\nGenerated {len(generated)} EM label maps → {OUT_DERIVATIVES}")

    # Create fold-specific symlink directories (same structure as synthseg_labels)
    print("\nCreating fold symlinks...")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for val_fold, fold_data in enumerate(splits):
        train_cases = fold_data["train"]
        link_dir = SPLITS_DIR / f"fold_{val_fold}"
        link_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for case_id in train_cases:
            parts = case_id.split("_ses-")
            sub = parts[0]
            ses = "ses-" + parts[1].replace("_T1w", "")
            out_fname = f"{sub}_{ses}_T1w_emlabels.nii.gz"
            target = OUT_DERIVATIVES / sub / ses / out_fname
            if not target.exists():
                continue
            link = link_dir / out_fname
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(target)
            n += 1
        print(f"  Fold {val_fold}: {n} EM label symlinks → {link_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
