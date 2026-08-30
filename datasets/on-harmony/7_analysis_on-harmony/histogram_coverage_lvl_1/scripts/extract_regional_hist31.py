#!/usr/bin/env python
"""
Extract 31-class regional histogram features (31 anatomical regions × 64 intensity bins
= 1984-dim) for the histogram-coverage analysis (Pillar 2), on THIS machine, fresh.

Uses the SAME 31-class parcellation as Pillar 1 (texture): the lateralised SynthSeg/
FreeSurfer classes of `Dataset031_OnHarmonyT1w31` (WM_L…VentralDC_R). This fixes the
inconsistency where the old regional_hist_64 used only 7 macro-regions.

Two modes:
  --mode synth : generated volumes  <gen>/<method>/<key>/<key>_run-NN.nii.gz
                 label = Dataset031 labelsTr/<key>.nii.gz  (already 1..31, used directly —
                 identical labels to Pillar 1; volumes are voxel-aligned to it).
  --mode real  : all real ON-Harmony scans (pooled, every modality), discovered via the
                 grounded native discover_scans; label = each scan's OWN per-modality
                 SynthSeg mask (FreeSurfer labels) remapped FreeSurfer→31. (The old code
                 wrongly used the T1w synthseg for every modality.)

Per-region histograms use one global p1–p99 normalisation over the brain (union of the 31
regions), so the inter-region intensity ordering that defines MRI contrast is preserved —
same recipe as the real regional_hist extractor. CPU-only, parallel.

Usage:
  # synth (all 4 methods)
  run_job --gpus 0 --cpus 16 --mem 32G --wait -- .venv/bin/python extract_regional_hist31.py \\
      --mode synth --output-csv .../outputs/synth_regional_hist31.csv --n-workers 16
  # real (pooled)
  run_job --gpus 0 --cpus 32 --mem 64G --wait -- .venv/bin/python extract_regional_hist31.py \\
      --mode real  --output-csv .../outputs/real_regional_hist31.csv  --n-workers 32
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS = Path(__file__).resolve()
ANALYSIS_ROOT = THIS.parents[2]                       # 7_analysis_on-harmony
ONHARMONY_ROOT = THIS.parents[3]                      # datasets/on-harmony
GENERATED_ROOT = ANALYSIS_ROOT / "texture_analysis_lvl_1" / "data" / "generated"
DS031 = ONHARMONY_ROOT / "2_nnUNet_on-harmony" / "raw" / "Dataset031_OnHarmonyT1w31"
DS031_LABELS = DS031 / "labelsTr"
BIDS_ROOT = Path(os.environ["BIDS_ROOT"])
DERIV_ROOT = BIDS_ROOT / "derivatives"
SYNTHSEG_DERIV = DERIV_ROOT / "labels"
METHODS = ["palette", "synthseg_em", "synthseg_noem", "auglab_default"]

N_REGIONS = 31
N_BINS_DEFAULT = 64
MIN_VOXELS = 50

# FreeSurfer label → Dataset031 class (1..31). Lateralised; matches Dataset031 dataset.json.
FS_TO_31: dict[int, int] = {
    2: 1, 3: 2, 4: 3, 5: 4, 7: 5, 8: 6, 10: 7, 11: 8, 12: 9, 13: 10,
    14: 11, 15: 12, 16: 13, 17: 14, 18: 15, 26: 16, 28: 17,
    41: 18, 42: 19, 43: 20, 44: 21, 46: 22, 47: 23, 49: 24, 50: 25, 51: 26, 52: 27,
    53: 28, 54: 29, 58: 30, 60: 31,
}

# ── Reuse grounded native discovery (stub unused radiomics import) ─────────────
if "radiomics" not in sys.modules:
    _r = types.ModuleType("radiomics")
    _fe = types.ModuleType("radiomics.featureextractor")
    _fe.RadiomicsFeatureExtractor = object
    _r.featureextractor = _fe
    sys.modules["radiomics"] = _r
    sys.modules["radiomics.featureextractor"] = _fe

_NATIVE = ANALYSIS_ROOT / "contrast_manifold" / "scripts" / "extract_features_native.py"
_spec = importlib.util.spec_from_file_location("_native", _NATIVE)
_native = importlib.util.module_from_spec(_spec)
sys.path.insert(0, str(_NATIVE.parent))
_spec.loader.exec_module(_native)
discover_scans = _native.discover_scans
parse_scanner = _native.parse_scanner


def feature_cols(n_bins: int) -> list[str]:
    return [f"region{r:02d}_hist_{b}" for r in range(1, N_REGIONS + 1) for b in range(n_bins)]


def compute_features_31(nii_path: Path, label_path: Path, is_freesurfer: bool,
                        n_bins: int) -> Optional[np.ndarray]:
    """(31*n_bins,) feature vector or None. Global p1-p99 norm over the 31-region brain."""
    try:
        img = nib.as_closest_canonical(nib.load(str(nii_path)))
        arr = img.get_fdata(dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
        img3d = nib.Nifti1Image(arr, img.affine)

        seg_img = nib.as_closest_canonical(nib.load(str(label_path)))
        if seg_img.shape[:3] != img3d.shape[:3] or not np.allclose(seg_img.affine, img3d.affine, atol=1e-3):
            seg_img = resample_from_to(seg_img, img3d, order=0)
        seg = np.round(seg_img.get_fdata()).astype(np.int32)

        if is_freesurfer:
            remap = np.zeros(int(seg.max()) + 1, dtype=np.int32) if seg.max() > 0 else np.zeros(1, np.int32)
            for fs, r in FS_TO_31.items():
                if fs < remap.shape[0]:
                    remap[fs] = r
            seg = remap[np.clip(seg, 0, remap.shape[0] - 1)]

        region_masks = [seg == r for r in range(1, N_REGIONS + 1)]
        brain_mask = np.zeros(arr.shape, dtype=bool)
        for m in region_masks:
            brain_mask |= m

        brain_vals = arr[brain_mask]
        if brain_vals.size < 500:
            return None
        p1, p99 = np.percentile(brain_vals, 1), np.percentile(brain_vals, 99)
        if p99 <= p1:
            return None
        arr_norm = np.clip((arr - p1) / (p99 - p1), 0.0, 1.0)

        feats = []
        for mask in region_masks:
            if mask.sum() < MIN_VOXELS:
                feats.append(np.zeros(n_bins, dtype=np.float32))
            else:
                h, _ = np.histogram(arr_norm[mask], bins=n_bins, range=(0.0, 1.0))
                s = h.sum()
                feats.append((h / s).astype(np.float32) if s > 0 else np.zeros(n_bins, np.float32))
        return np.concatenate(feats)
    except Exception as exc:
        log.warning("Feature computation failed for %s: %s", Path(nii_path).name, exc)
        return None


def _worker(args: tuple) -> tuple:
    idx, nii, lbl, is_fs, n_bins = args
    return idx, compute_features_31(Path(nii), Path(lbl), is_fs, n_bins)


def discover_synth(methods: list[str]) -> list[dict]:
    tasks, missing = [], set()
    for method in methods:
        mdir = GENERATED_ROOT / method
        if not mdir.is_dir():
            log.warning("Method dir absent: %s", mdir); continue
        for case in sorted(p for p in mdir.iterdir() if p.is_dir()):
            key = case.name
            lbl = DS031_LABELS / f"{key}.nii.gz"
            if not lbl.exists():
                missing.add(key); continue
            for nii in sorted(case.glob(f"{key}_run-*.nii.gz")):
                run = nii.stem.split("_run-")[-1].replace(".nii", "")
                tasks.append({"method": method, "key": key, "subject": key.split("_")[0],
                              "session": key.split("_")[1], "run": run,
                              "path": nii, "label": lbl, "is_fs": False})
    if missing:
        log.warning("No Dataset031 label for %d case(s): %s", len(missing), sorted(missing)[:3])
    return tasks


def _synthseg_for_image(img_path: Path) -> Optional[Path]:
    """Per-modality SynthSeg: <bids>/x.nii.gz → <bids>/derivatives/labels/x_label-synthseg_dseg.nii.gz"""
    rel = img_path.relative_to(BIDS_ROOT)
    seg = SYNTHSEG_DERIV / rel.parent / img_path.name.replace(".nii.gz", "_label-synthseg_dseg.nii.gz")
    return seg if seg.exists() else None


def discover_real() -> list[dict]:
    scans = discover_scans(BIDS_ROOT, DERIV_ROOT, no_mask=True)
    tasks, no_seg = [], 0
    for s in scans:
        seg = _synthseg_for_image(s["path"])
        if seg is None:
            no_seg += 1; continue
        model, vendor = parse_scanner(s.get("ses", ""))
        tasks.append({"path": s["path"], "label": seg, "is_fs": True,
                      "subject": s.get("sub", ""), "session": s.get("ses", ""),
                      "modality_id": s.get("modality_id", ""), "acq_tag": s.get("acq_tag", ""),
                      "scanner_model": model, "scanner_vendor": vendor})
    log.info("Real scans with per-modality synthseg: %d (skipped %d without seg)", len(tasks), no_seg)
    return tasks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["synth", "real"], required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--n-bins", type=int, default=N_BINS_DEFAULT)
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--methods", nargs="+", default=METHODS)
    args = p.parse_args()

    tasks = discover_synth(args.methods) if args.mode == "synth" else discover_real()
    log.info("Discovered %d volumes (mode=%s)", len(tasks), args.mode)
    if not tasks:
        log.error("Nothing to extract."); sys.exit(1)

    fcols = feature_cols(args.n_bins)
    rows: list[Optional[dict]] = [None] * len(tasks)
    n_fail = 0
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futs = {pool.submit(_worker, (i, str(t["path"]), str(t["label"]), t["is_fs"], args.n_bins)): i
                for i, t in enumerate(tasks)}
        for done, fut in enumerate(as_completed(futs), 1):
            i, feat = fut.result()
            t = tasks[i]
            if args.mode == "synth":
                base = {"method": t["method"], "subject": t["subject"], "session": t["session"],
                        "key": t["key"], "run": t["run"]}
            else:
                base = {"subject": t["subject"], "session": t["session"],
                        "modality_id": t["modality_id"], "acq_tag": t["acq_tag"],
                        "scanner_model": t["scanner_model"], "scanner_vendor": t["scanner_vendor"],
                        "image_path": str(t["path"]), "label_map_path": str(t["label"])}
            if feat is None:
                n_fail += 1
                rows[i] = {**base, **{c: float("nan") for c in fcols}}
            else:
                rows[i] = {**base, **dict(zip(fcols, feat.tolist()))}
            if done % 500 == 0:
                log.info("  %d / %d done (%d failed)", done, len(tasks), n_fail)

    df = pd.DataFrame([r for r in rows if r is not None])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    log.info("Saved %d rows × %d cols → %s (%d failed)", len(df), df.shape[1], args.output_csv, n_fail)
    if args.mode == "synth":
        log.info("Per-method:\n%s", df["method"].value_counts().to_string())
    else:
        log.info("By modality:\n%s", df["modality_id"].str.split("_").str[0].value_counts().to_string())


if __name__ == "__main__":
    main()
