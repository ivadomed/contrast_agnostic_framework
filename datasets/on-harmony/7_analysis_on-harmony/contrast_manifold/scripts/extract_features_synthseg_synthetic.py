#!/usr/bin/env python
"""
Extract per-region PyRadiomics features for synthetic ON-Harmony volumes
using the T1w SynthSeg label map of the corresponding original subject/session.

Because synthetic volumes share anatomy with the original T1w, we reuse
the T1w SynthSeg label map directly (already in T1w space = synthetic space).

Output CSV goes under:
  analysis/contrast_manifold/outputs/data/{synthetic_cohort}/synthseg_mask_{N}/

Usage (single slot):
  run_job --gpus 1 --slot 0 --wait -- .venv/bin/python \\
    analysis/contrast_manifold/scripts/extract_features_synthseg_synthetic.py \\
    --synth-root data/ON-Harmony/derivatives/synthetic_v19_c \\
    --output-csv analysis/contrast_manifold/outputs/data/synthetic_v19_c/synthseg_mask_31/synthetic_v19_c_features.csv \\
    --n_workers 8

Usage (4 parallel slots):
  for rank in 0 1 2 3; do
    run_job --gpus 1 --slot $rank --wait -- .venv/bin/python \\
      analysis/contrast_manifold/scripts/extract_features_synthseg_synthetic.py \\
      --synth-root data/ON-Harmony/derivatives/synthetic_v19_c \\
      --output-csv analysis/contrast_manifold/outputs/data/synthetic_v19_c/synthseg_mask_31/synthetic_v19_c_features.csv \\
      --n_workers 4 --rank $rank --world-size 4 \\
      > /tmp/feat_synth_v19c_rank${rank}.log 2>&1 < /dev/null &
  done
  wait && echo "All done."

To restrict to a subset of label IDs:
  --label-subset 2,3,41,42
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

for _lg in ("radiomics", "radiomics.glcm", "pykwalify", "pykwalify.core"):
    logging.getLogger(_lg).setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Import helpers from the native extraction script
_NATIVE_SCRIPT = Path(__file__).parent / "extract_features_native.py"
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("extract_features_native", _NATIVE_SCRIPT)
_native = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_native)

parse_bids = _native.parse_bids

# Import multilabel extractor and label registry from the original synthseg script
_ORIG_SCRIPT = Path(__file__).parent / "extract_features_synthseg_original.py"
_spec2 = _ilu.spec_from_file_location("extract_features_synthseg_original", _ORIG_SCRIPT)
_orig = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_orig)

ALL_LABELS               = _orig.ALL_LABELS
_extract_one_multilabel  = _orig._extract_one_multilabel


# ─── Scan discovery ───────────────────────────────────────────────────────────

def discover_synthetic_scans(
    synth_root: Path,
    synthseg_root: Path,
    cohort: str,
) -> list[dict]:
    """
    Pair each synthetic NIfTI with the T1w SynthSeg label map of the same session.

    Synthetic layout:
      synth_root/sub-X/ses-Y/sub-X_ses-Y_run-NN_syn-T1w.nii.gz

    SynthSeg label map:
      synthseg_root/sub-X/ses-Y/anat/sub-X_ses-Y_T1w_synthseg.nii.gz
    """
    records: list[dict] = []
    for nii in sorted(synth_root.rglob("*_syn-T1w.nii.gz")):
        sub_dir = nii.parent.parent.name   # sub-X
        ses_dir = nii.parent.name          # ses-Y
        sub     = sub_dir.replace("sub-", "")
        ses     = ses_dir.replace("ses-", "")

        label_map = (
            synthseg_root / sub_dir / ses_dir / "anat"
            / f"{sub_dir}_{ses_dir}_T1w_synthseg.nii.gz"
        )
        if not label_map.exists():
            log.debug("No SynthSeg label map for %s/%s — skipping", sub_dir, ses_dir)
            continue

        entities   = parse_bids(nii.name)
        run        = entities.get("run", "??")
        modality_id = f"syn-T1w_run-{run}"
        acq_tag     = f"{cohort}_run-{run}"

        records.append({
            "path":        nii,
            "label_map":   label_map,
            "entities":    entities,
            "sub":         f"sub-{sub}",
            "ses":         f"ses-{ses}",
            "modality_id": modality_id,
            "acq_tag":     acq_tag,
            "cohort":      cohort,
        })
    return records


# ─── Per-scan worker wrapper ──────────────────────────────────────────────────

def _extract_synthetic_one(
    nii_path: Path,
    label_map_path: Path,
    entities: dict,
    sub: str,
    ses: str,
    modality_id: str,
    acq_tag: str,
    cohort: str,
    config_path: Path,
    active_labels: dict[int, str],
) -> dict | None:
    result = _extract_one_multilabel(
        nii_path=nii_path,
        label_map_path=label_map_path,
        entities=entities,
        sub=sub,
        ses=ses,
        modality_id=modality_id,
        acq_tag=acq_tag,
        config_path=config_path,
        active_labels=active_labels,
    )
    if result is not None:
        result["cohort_category"] = cohort
        result["scanner_model"]   = f"{cohort}_generator"
        result["scanner_vendor"]  = "synthetic"
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synth-root", type=Path, required=True,
                   help="Root of synthetic NIfTI files "
                        "(e.g. data/ON-Harmony/derivatives/synthetic_v19_c)")
    p.add_argument("--synthseg-root", type=Path,
                   default=PROJECT_ROOT / "data" / "ON-Harmony"
                           / "derivatives" / "synthseg_masks")
    p.add_argument("--output-csv", type=Path, default=None,
                   help="Output CSV path. Auto-derived from synth-root if not set.")
    p.add_argument("--config", type=Path,
                   default=PROJECT_ROOT / "analysis" / "contrast_manifold"
                           / "config" / "radiomics_config.yaml")
    p.add_argument("--label-subset", type=str, default=None,
                   help="Comma-separated label IDs. Default: all 31 labels. "
                        "Example: --label-subset 2,3,41,42")
    p.add_argument("--n_workers",  type=int, default=4)
    p.add_argument("--append",     action="store_true")
    p.add_argument("--limit",      type=int, default=None)
    p.add_argument("--rank",       type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Derive cohort name from synth-root basename (e.g. synthetic_v19_c)
    cohort = args.synth_root.name

    # Resolve active labels
    if args.label_subset:
        ids = [int(x.strip()) for x in args.label_subset.split(",")]
        active_labels = {i: ALL_LABELS[i] for i in ids if i in ALL_LABELS}
        if not active_labels:
            log.error("No valid label IDs in --label-subset %s", args.label_subset)
            sys.exit(1)
    else:
        active_labels = ALL_LABELS
    n_labels = len(active_labels)

    if args.output_csv is None:
        args.output_csv = (
            PROJECT_ROOT / "analysis" / "contrast_manifold" / "outputs"
            / "data" / cohort / f"synthseg_mask_{n_labels}"
            / f"{cohort}_features.csv"
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    log.info("Cohort: %s  Labels: %d  Output → %s", cohort, n_labels, args.output_csv)

    if not args.config.exists():
        raise FileNotFoundError(f"PyRadiomics config not found: {args.config}")
    if not args.synth_root.exists():
        raise FileNotFoundError(f"Synthetic root not found: {args.synth_root}")

    log.info("Discovering synthetic scans …")
    scans = discover_synthetic_scans(args.synth_root, args.synthseg_root, cohort)
    log.info("%d synthetic scans found with label maps", len(scans))

    if args.world_size > 1:
        scans = scans[args.rank :: args.world_size]
        log.info("Rank %d/%d → %d scans", args.rank, args.world_size, len(scans))

    if args.limit:
        scans = scans[: args.limit]

    done_keys: set[str] = set()
    if args.append and args.output_csv.exists():
        existing = pd.read_csv(args.output_csv, usecols=["image_path"])
        done_keys = set(existing["image_path"].tolist())
        scans = [s for s in scans if str(s["path"]) not in done_keys]
        log.info("%d scans remaining after dedup", len(scans))

    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor, as_completed

    log.info("Submitting %d jobs (%d workers) …", len(scans), args.n_workers)

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {
            pool.submit(
                _extract_synthetic_one,
                s["path"], s["label_map"], s["entities"],
                s["sub"], s["ses"], s["modality_id"], s["acq_tag"],
                s["cohort"], args.config, active_labels,
            ): s["path"]
            for s in scans
        }
        with tqdm(as_completed(futures), total=len(futures),
                  desc="Extracting features", unit="scan",
                  dynamic_ncols=True) as pbar:
            for fut in pbar:
                src = futures[fut]
                try:
                    rec = fut.result()
                except Exception as exc:
                    pbar.write(f"ERROR {src.name}: {exc}")
                    rec = None
                if rec is not None:
                    results.append(rec)
                else:
                    pbar.write(f"WARN  no features: {src.name}")
                pbar.set_postfix(valid=len(results))

    if not results:
        log.error("No features extracted.")
        return

    META_COLS = [
        "subject", "session", "modality_id", "acq_tag",
        "scanner_model", "scanner_vendor", "cohort_category",
        "image_path", "label_map_path",
    ]
    df_new    = pd.DataFrame(results)
    feat_cols = [c for c in df_new.columns if c not in META_COLS]
    df_new    = df_new[META_COLS + feat_cols]

    if args.append and args.output_csv.exists():
        df_existing = pd.read_csv(args.output_csv)
        df_out = pd.concat([df_existing, df_new], axis=0, ignore_index=True, sort=False)
    else:
        df_out = df_new

    df_out.to_csv(args.output_csv, index=False)
    log.info("Saved %d rows × %d columns → %s",
             len(df_out), len(df_out.columns), args.output_csv)


if __name__ == "__main__":
    main()
