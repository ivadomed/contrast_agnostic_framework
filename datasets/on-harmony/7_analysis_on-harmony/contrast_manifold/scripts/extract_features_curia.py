#!/usr/bin/env python
"""
Extract CURIA (DINOv2-based) embeddings for MRI volumes.

For each 3D NIfTI:
  1. Load volume and reorient to RAS (axial slices = last axis).
  2. Take the middle 50% of axial slices (ignores empty superior/inferior slabs).
  3. Process each slice through CuriaImageProcessor (→ 512×512, z-score normalised).
  4. Forward through CURIA backbone in large batched GPU calls → CLS token per slice.
  5. Average CLS tokens across slices → one 768-dim vector per volume.

Outputs a CSV with metadata columns + curia_0 … curia_767.

Usage (original ON-Harmony):
  run_job --gpus 1 --slot 0 --wait -- .venv/bin/python analysis/contrast_manifold/scripts/extract_features_curia.py \\
    --mode original --gpu-id 0 \\
    --output-csv analysis/contrast_manifold/outputs/data/original/curia_embeddings/on_harmony_features.csv

Usage (synthetic v19_c, 3 parallel processes):
  run_job --gpus 1 --slot 0 --wait -- .venv/bin/python ... --mode synthetic --gpu-id 1 --rank 0 --world-size 3 ...
  run_job --gpus 1 --slot 0 --wait -- .venv/bin/python ... --mode synthetic --gpu-id 2 --rank 1 --world-size 3 ...
  run_job --gpus 1 --slot 0 --wait -- .venv/bin/python ... --mode synthetic --gpu-id 3 --rank 2 --world-size 3 ...
"""
from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import threading
import warnings
from pathlib import Path

# Pin GPU before torch import so device_count() reflects only one GPU.
_gpu_arg_idx = next((i for i, a in enumerate(sys.argv) if a == "--gpu-id"), None)
if _gpu_arg_idx is not None and _gpu_arg_idx + 1 < len(sys.argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_gpu_arg_idx + 1]

import nibabel as nib
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

HF_TOKEN         = os.environ.get("HF_TOKEN", "")
MODEL_ID         = "raidium/curia"
N_EMB            = 768   # CURIA CLS-token dimensionality
GPU_BATCH_SLICES = 512   # slices per GPU forward pass (~20 GB activation mem on A6000)
N_IO_WORKERS     = 16    # parallel CPU threads for NIfTI load + preprocess
PREFETCH_SCANS   = 32    # scans queued ahead of GPU


# ─── Re-use scan-discovery helpers ────────────────────────────────────────────
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "extract_features_native",
    Path(__file__).parent / "extract_features_native.py",
)
_native = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_native)

discover_scans   = _native.discover_scans
parse_bids       = _native.parse_bids
make_modality_id = _native.make_modality_id
make_acq_tag     = _native.make_acq_tag
parse_scanner    = _native.parse_scanner


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model_and_processor(device: torch.device):
    from huggingface_hub import login
    from transformers import AutoImageProcessor, AutoModel

    login(token=HF_TOKEN)
    log.info("Loading CuriaImageProcessor …")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    log.info("Loading CURIA backbone …")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval().to(device)
    log.info("Model on %s", device)
    return model, processor


# ─── Volume preprocessing (CPU) ───────────────────────────────────────────────

def volume_to_pixel_values(nii_path: Path, processor) -> torch.Tensor | None:
    """Load NIfTI → preprocess middle-50% axial slices → (N, 1, 512, 512) tensor."""
    try:
        img = nib.as_closest_canonical(nib.load(str(nii_path)))
        arr = img.get_fdata(dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]

        # CuriaImageProcessor casts to int16: rescale normalized images to ~[0,1000]
        # so they survive the int16 truncation (values in [0,1] would all become 0).
        nonzero = arr[arr > 0]
        if nonzero.size > 0:
            p99 = float(np.percentile(nonzero, 99))
            if p99 < 10.0:
                arr = (arr / p99 * 1000.0).astype(np.float32)

        n = arr.shape[2]
        slices = [arr[:, :, z] for z in range(n // 4, n - n // 4)]
        inputs = processor(slices, return_tensors="pt")
        return inputs["pixel_values"]   # (N_slices, 1, 512, 512)
    except Exception as exc:
        log.warning("  Preprocess failed for %s: %s", nii_path.name, exc)
        return None


# ─── Async prefetch pipeline ──────────────────────────────────────────────────

_SENTINEL = object()


def _io_worker(task_q: queue.Queue, result_q: queue.Queue, processor):
    """Pull (scan_idx, scan) tasks, preprocess, push (scan_idx, scan, pixel_values)."""
    while True:
        item = task_q.get()
        if item is _SENTINEL:
            result_q.put(_SENTINEL)
            return
        scan_idx, scan = item
        pv = volume_to_pixel_values(scan["path"], processor)
        result_q.put((scan_idx, scan, pv))


def embed_all(
    scans: list[dict],
    model,
    processor,
    device: torch.device,
    progress_cb=None,
) -> dict[int, np.ndarray]:
    """
    Return {scan_idx: embedding (768,)} for all scans using async CPU prefetch.
    Slices from multiple scans are batched together for GPU efficiency.
    """
    task_q: queue.Queue = queue.Queue(maxsize=PREFETCH_SCANS * 2)
    result_q: queue.Queue = queue.Queue(maxsize=PREFETCH_SCANS * 2)

    # Start IO workers
    workers = []
    for _ in range(N_IO_WORKERS):
        t = threading.Thread(target=_io_worker, args=(task_q, result_q, processor), daemon=True)
        t.start()
        workers.append(t)

    # Feed tasks
    def _feeder():
        for idx, scan in enumerate(scans):
            task_q.put((idx, scan))
        for _ in workers:
            task_q.put(_SENTINEL)

    threading.Thread(target=_feeder, daemon=True).start()

    # GPU consumer: accumulate slices across scans, fire when batch is full
    embeddings: dict[int, np.ndarray] = {}
    slice_buf: list[torch.Tensor] = []   # accumulated pixel_values chunks
    origin: list[tuple[int, int]] = []   # (scan_idx, n_slices) per chunk

    done_workers = 0
    completed = 0

    def _flush():
        if not slice_buf:
            return
        combined = torch.cat(slice_buf, dim=0).to(device)   # (total_slices, 1, 512, 512)
        with torch.no_grad():
            cls = model(combined).last_hidden_state[:, 0].cpu().float().numpy()
        offset = 0
        for scan_idx, n in origin:
            embeddings[scan_idx] = cls[offset : offset + n].mean(axis=0)
            offset += n
        slice_buf.clear()
        origin.clear()

    while done_workers < N_IO_WORKERS:
        item = result_q.get()
        if item is _SENTINEL:
            done_workers += 1
            continue
        scan_idx, scan, pv = item
        completed += 1
        if progress_cb:
            progress_cb(completed, scan["path"].name)
        if pv is None:
            continue
        n_slices = pv.shape[0]
        # Check if adding this scan would overflow GPU memory → flush first
        if slice_buf and sum(x.shape[0] for x in slice_buf) + n_slices > GPU_BATCH_SLICES:
            _flush()
        slice_buf.append(pv)
        origin.append((scan_idx, n_slices))

    _flush()   # process remaining
    return embeddings


# ─── Scan discovery ───────────────────────────────────────────────────────────

def discover_original(bids_root: Path, deriv_root: Path) -> list[dict]:
    return discover_scans(bids_root, deriv_root, no_mask=True)


def discover_synthetic(synth_root: Path) -> list[dict]:
    records = []
    for nii in sorted(synth_root.rglob("*_syn-*.nii.gz")):
        entities  = parse_bids(nii.name)
        sub       = nii.parts[nii.parts.index(synth_root.name) + 1]
        ses_parts = [p for p in nii.parts if p.startswith("ses-")]
        ses       = ses_parts[0] if ses_parts else "unknown"
        records.append({
            "path":       nii,
            "entities":   entities,
            "modality_id": (
                f"syn-{entities['syn']}_run-{entities['run']}"
                if "syn" in entities and "run" in entities
                else f"syn-{entities['syn']}" if "syn" in entities
                else make_modality_id(entities)
            ),
            "acq_tag":    make_acq_tag(entities),
            "sub":        sub,
            "ses":        ses,
        })
    return records


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode",       choices=["original", "synthetic"], required=True)
    p.add_argument("--bids-root",  type=Path, default=PROJECT_ROOT / "data" / "ON-Harmony")
    p.add_argument("--deriv-root", type=Path,
                   default=PROJECT_ROOT / "data" / "ON-Harmony" / "derivatives")
    p.add_argument("--synth-root", type=Path, default=None)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--rank",       type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--skip-csv",   type=Path, default=None,
                   help="CSV with already-done image_path values to skip.")
    p.add_argument("--gpu-id",     type=str, default=None,
                   help="Pin to single GPU (sets CUDA_VISIBLE_DEVICES before torch import).")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s  (%d GPUs visible)", device, torch.cuda.device_count())

    model, processor = load_model_and_processor(device)

    if args.mode == "original":
        scans = discover_original(args.bids_root, args.deriv_root)
    else:
        if args.synth_root is None:
            log.error("--synth-root required for synthetic mode"); sys.exit(1)
        scans = discover_synthetic(args.synth_root)

    # World-size split FIRST, then skip — ensures deterministic per-rank subsets
    if args.world_size > 1:
        scans = scans[args.rank :: args.world_size]

    if args.skip_csv and args.skip_csv.exists():
        done = set(pd.read_csv(args.skip_csv, usecols=["image_path"])["image_path"].astype(str))
        before = len(scans)
        scans = [s for s in scans if str(s["path"].resolve().relative_to(PROJECT_ROOT)) not in done]
        log.info("Skipping %d already-done → %d remaining", before - len(scans), len(scans))

    log.info("Processing %d scans (rank %d/%d) | IO workers=%d GPU_BATCH=%d slices",
             len(scans), args.rank, args.world_size, N_IO_WORKERS, GPU_BATCH_SLICES)

    out_csv = args.output_csv
    if args.world_size > 1:
        out_csv = args.output_csv.with_name(
            args.output_csv.stem + f"_rank{args.rank}" + args.output_csv.suffix
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    emb_cols = [f"curia_{i}" for i in range(N_EMB)]
    written_header = out_csv.exists()

    def _progress(done: int, name: str):
        if done % 10 == 0 or done == len(scans):
            log.info("[%d/%d] %s", done, len(scans), name)

    results = embed_all(scans, model, processor, device, progress_cb=_progress)

    with open(out_csv, "a", newline="") as fh:
        for scan_idx, scan in enumerate(scans):
            emb = results.get(scan_idx)
            if emb is None:
                log.warning("  Skipped (no embedding): %s", scan["path"].name)
                continue
            scanner_model, vendor = parse_scanner(scan["ses"])
            row = {
                "subject":        scan["sub"],
                "session":        scan["ses"],
                "modality_id":    scan["modality_id"],
                "scanner_model":  scanner_model,
                "scanner_vendor": vendor,
                "acq_tag":        scan.get("acq_tag", ""),
                "image_path":     str(scan["path"].resolve().relative_to(PROJECT_ROOT)),
            }
            for i, v in enumerate(emb):
                row[emb_cols[i]] = round(float(v), 6)
            df_row = pd.DataFrame([row])
            df_row.to_csv(fh, header=not written_header, index=False)
            written_header = True
            fh.flush()

    log.info("Done → %s  (%d embeddings)", out_csv, len(results))


if __name__ == "__main__":
    main()
