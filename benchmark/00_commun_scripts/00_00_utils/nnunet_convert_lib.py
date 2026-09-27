#!/usr/bin/env python3
"""
Shared helpers for the 02_nnunet conversion stage (BIDS → nnUNet raw).

The per-dataset convert scripts differ in the per-case conversion (channel count, BIDS
file naming, segmentation path) and in case discovery / filtering. What they share — the
on-the-fly gzip copy, the threaded conversion driver with failure collection, and writing
dataset.json — lives here. Each script supplies its own ``convert_fn(case) -> case_id``
and its dataset.json fields.
"""
import gzip
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def gzip_copy(src: Path, dst: Path) -> None:
    """Copy src (.nii) to dst (.nii.gz), compressing on the fly. No-op if dst exists."""
    if dst.exists():
        return
    with open(src, "rb") as f_in, gzip.open(dst, "wb", compresslevel=1) as f_out:
        shutil.copyfileobj(f_in, f_out)


def run_threaded_conversion(items, convert_fn, jobs: int, progress_every: int = 0) -> list:
    """Run convert_fn over items in a thread pool. convert_fn(item) -> case_id.

    Collects per-item failures and, if any, prints them and raises SystemExit(1).
    Returns the list of case_ids (completion order). If progress_every > 0, prints an
    ``i/total`` line every that many completions (and on the last).
    """
    items = list(items)
    case_ids, failed = [], []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(convert_fn, it): it for it in items}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                case_ids.append(fut.result())
            except Exception as e:  # noqa: BLE001 — report every failing item, then bail
                failed.append((futs[fut], str(e)))
            if progress_every and (i % progress_every == 0 or i == len(items)):
                print(f"  {i}/{len(items)}")
    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for it, err in failed:
            print(f"  {it}: {err}")
        raise SystemExit(1)
    return case_ids


def write_dataset_json(out_dir: Path, channel_names: dict, labels: dict,
                       num_training: int, **extra) -> Path:
    """Write nnU-Net dataset.json. ``extra`` carries name/description/reference/licence/
    release etc. file_ending defaults to .nii.gz. Returns the written path."""
    payload = dict(extra)
    payload["channel_names"] = channel_names
    payload["labels"] = labels
    payload["numTraining"] = num_training
    payload.setdefault("file_ending", ".nii.gz")
    out = Path(out_dir) / "dataset.json"
    out.write_text(json.dumps(payload, indent=2))
    return out
