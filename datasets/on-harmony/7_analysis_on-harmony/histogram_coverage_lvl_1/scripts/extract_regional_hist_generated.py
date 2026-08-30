#!/usr/bin/env python
"""
Extract regional_hist_64 features (7 SynthSeg regions × 64 intensity bins = 448-dim)
from the Level-1 generated volumes, for the histogram-coverage analysis (Pillar 2).

Reuses the EXACT feature-computation core the real ON-Harmony features were built with
(`compute_features`, the FreeSurfer→7-region map) — only discovery and the SynthSeg
path resolution differ, so real and synthetic live in the same 448-dim space.

Generated volumes (Pillar 1's data, reused for consistency):
  <generated_root>/<method>/<key>/<key>_run-NN.nii.gz     key = sub-XXXX_ses-YYYY_T1w
The volumes are voxel-aligned to their source T1w, so the T1w SynthSeg parcellation of
the same sub/ses applies directly:
  <bids>/derivatives/labels/<sub>/<ses>/anat/<sub>_<ses>_T1w_label-synthseg_dseg.nii.gz

Output: one CSV with meta cols {method, subject, session, key, run} + 448 hist columns.

CPU-only; parallel via ProcessPoolExecutor. Dispatch through run_job --gpus 0.

Usage:
  run_job --gpus 0 --cpus 16 --mem 32G --wait -- .venv/bin/python \\
    datasets/on-harmony/7_analysis_on-harmony/histogram_coverage_lvl_1/scripts/extract_regional_hist_generated.py \\
    --output-csv .../histogram_coverage_lvl_1/outputs/synth_regional_hist_64.csv --n-workers 16
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS = Path(__file__).resolve()
ANALYSIS_ROOT = THIS.parents[2]                       # 7_analysis_on-harmony
GENERATED_ROOT = ANALYSIS_ROOT / "texture_analysis_lvl_1" / "data" / "generated"
BIDS_SYNTHSEG = Path(os.environ["BIDS_ROOT"]) / "derivatives" / "labels"
METHODS = ["palette", "synthseg_em", "synthseg_noem", "auglab_default"]

# ── Reuse the grounded feature core from the contrast_manifold extractor ───────
# That module transitively imports extract_features_native → `from radiomics import
# featureextractor`. pyradiomics is not in the rebuilt venv and is unused for histogram
# features (compute_features needs only nibabel/numpy), so stub it to reuse the REAL,
# unchanged compute_features without pulling a heavy unused dependency.
import types as _types
if "radiomics" not in sys.modules:
    _r = _types.ModuleType("radiomics")
    _fe = _types.ModuleType("radiomics.featureextractor")
    _fe.RadiomicsFeatureExtractor = object   # referenced in a native type annotation at def-time
    _r.featureextractor = _fe
    sys.modules["radiomics"] = _r
    sys.modules["radiomics.featureextractor"] = _fe

_EXTRACTOR = (ANALYSIS_ROOT / "contrast_manifold" / "scripts"
              / "extract_features_regional_hist.py")
sys.path.insert(0, str(_EXTRACTOR.parent))            # so its own `import extract_features_native` resolves
_spec = importlib.util.spec_from_file_location("_rh_extractor", _EXTRACTOR)
_rh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rh)
compute_features = _rh.compute_features
_build_feature_cols = _rh._build_feature_cols


def _synthseg_for_key(key: str) -> Path | None:
    """key = sub-XXXX_ses-YYYY_T1w  →  T1w SynthSeg parcellation for that sub/ses."""
    parts = key.split("_")
    if len(parts) < 2:
        return None
    sub, ses = parts[0], parts[1]
    p = BIDS_SYNTHSEG / sub / ses / "anat" / f"{sub}_{ses}_T1w_label-synthseg_dseg.nii.gz"
    return p if p.exists() else None


def discover(generated_root: Path, methods: list[str]) -> list[dict]:
    tasks: list[dict] = []
    missing_seg: set[str] = set()
    for method in methods:
        mdir = generated_root / method
        if not mdir.is_dir():
            log.warning("Method dir absent: %s", mdir)
            continue
        for case in sorted(mdir.iterdir()):
            if not case.is_dir():
                continue
            key = case.name
            seg = _synthseg_for_key(key)
            if seg is None:
                missing_seg.add(key)
                continue
            for nii in sorted(case.glob(f"{key}_run-*.nii.gz")):
                run = nii.stem.split("_run-")[-1].replace(".nii", "")
                tasks.append({
                    "method": method, "key": key,
                    "subject": key.split("_")[0], "session": key.split("_")[1],
                    "run": run, "path": nii, "synthseg": seg,
                })
    if missing_seg:
        log.warning("No T1w SynthSeg for %d case(s), e.g. %s",
                    len(missing_seg), sorted(missing_seg)[:3])
    return tasks


def _worker(args: tuple) -> tuple:
    idx, path, seg, n_bins = args
    return idx, compute_features(Path(path), Path(seg), n_bins)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generated-root", type=Path, default=GENERATED_ROOT)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--n-bins", type=int, default=64)
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--methods", nargs="+", default=METHODS)
    args = p.parse_args()

    tasks = discover(args.generated_root, args.methods)
    log.info("Discovered %d volumes across %d methods", len(tasks), len(args.methods))
    if not tasks:
        log.error("Nothing to extract."); sys.exit(1)

    feat_cols = _build_feature_cols(args.n_bins)
    rows: list[dict] = [None] * len(tasks)  # type: ignore
    n_fail = 0
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futs = {pool.submit(_worker, (i, str(t["path"]), str(t["synthseg"]), args.n_bins)): i
                for i, t in enumerate(tasks)}
        for done, fut in enumerate(as_completed(futs), 1):
            i, feat = fut.result()
            t = tasks[i]
            base = {"method": t["method"], "subject": t["subject"],
                    "session": t["session"], "key": t["key"], "run": t["run"]}
            if feat is None:
                n_fail += 1
                rows[i] = {**base, **{c: float("nan") for c in feat_cols}}
            else:
                rows[i] = {**base, **dict(zip(feat_cols, feat.tolist()))}
            if done % 500 == 0:
                log.info("  %d / %d done (%d failed)", done, len(tasks), n_fail)

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    log.info("Saved %d rows × %d cols → %s  (%d failed)",
             len(df), df.shape[1], args.output_csv, n_fail)
    log.info("Per-method counts:\n%s", df["method"].value_counts().to_string())


if __name__ == "__main__":
    main()
