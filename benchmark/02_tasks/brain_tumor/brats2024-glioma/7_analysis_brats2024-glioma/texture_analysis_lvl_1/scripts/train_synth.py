#!/usr/bin/env python
"""
Trains ONE cross-contrast synthesis model: input = a single source contrast, output = the other
3 contrasts (fixed channel order from synth_common.targets_for). MONAI 3D UNet, random 96^3
patches (50% centred on tumor voxels, 50% on brain voxels), L1 loss masked to the brain, AMP.

Reads the cache built once by build_cache.py -- never builds it itself (parallel per-source jobs
would race). Logs train loss / val L1 every EVAL_EVERY iterations to
outputs/logs/train_curve_<source>.csv and saves checkpoints to $SCRATCH/brats_synth_checkpoints/
<source>/{latest.pt,final.pt}. --resume continues from latest.pt's stored iteration count.

Usage (inside a Slurm GPU job):
  .venv/bin/python train_synth.py --source t1n --iters 4000 [--resume] [--smoke]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import UNet

import synth_common as sc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PATCH = 96
BATCH = 4
LR = 2e-4
EVAL_EVERY = 250
N_VAL_PATCHES = 24
TUMOR_FRAC = 0.5


class PatientPool:
    """Holds every patient's cropped cache in host RAM and samples 96^3 patches on CPU (cheap
    array slicing) -- avoids re-reading .npz files every iteration and keeps the GPU fed."""

    def __init__(self, pids: list[str]):
        self.pids = pids
        self.cache = {}
        self.tumor_coords = {}
        for pid in pids:
            d = sc.load_cache(pid)
            self.cache[pid] = d
            ys, xs, zs = np.where(d["label"] > 0)
            self.tumor_coords[pid] = np.stack([ys, xs, zs], axis=1) if len(ys) else None

    def sample_patch(self, pid: str, rng: np.random.Generator):
        d = self.cache[pid]
        brain = d["brain"]
        shape = np.array(brain.shape)
        half = PATCH // 2
        c = None
        if rng.random() < TUMOR_FRAC and self.tumor_coords[pid] is not None:
            tc = self.tumor_coords[pid]
            c = tc[rng.integers(len(tc))].astype(np.int64)
        if c is None:
            for _ in range(10):
                cand = np.array([rng.integers(0, shape[0]), rng.integers(0, shape[1]),
                                  rng.integers(0, shape[2])])
                if brain[tuple(cand)]:
                    c = cand
                    break
            if c is None:
                c = shape // 2
        lo = np.clip(c - half, 0, shape - PATCH)
        sl = tuple(slice(int(a), int(a + PATCH)) for a in lo)
        return sl

    def batch(self, source: str, targets: tuple[str, ...], rng: np.random.Generator, batch=BATCH):
        xs, ys, ms = [], [], []
        for _ in range(batch):
            pid = self.pids[rng.integers(len(self.pids))]
            sl = self.sample_patch(pid, rng)
            d = self.cache[pid]
            xs.append(d[source][sl].astype(np.float32))
            ys.append(np.stack([d[t][sl].astype(np.float32) for t in targets], axis=0))
            ms.append(d["brain"][sl])
        x = torch.from_numpy(np.stack(xs))[:, None]           # (B,1,P,P,P)
        y = torch.from_numpy(np.stack(ys))                    # (B,3,P,P,P)
        m = torch.from_numpy(np.stack(ms))[:, None].float()   # (B,1,P,P,P)
        return x, y, m


def masked_l1(pred, target, mask):
    denom = mask.sum().clamp_min(1.0) * pred.shape[1]
    return (F.l1_loss(pred, target, reduction="none") * mask).sum() / denom


def build_model():
    return UNet(spatial_dims=3, in_channels=1, out_channels=3,
                channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sc.CONTRASTS)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s cuda_available=%s", device, torch.cuda.is_available())

    split = json.loads((sc.OUT_DIR / "data" / "synth_train_val_split.json").read_text())
    train_ids, val_ids = split["train"], split["val"]
    if args.smoke:
        train_ids, val_ids = train_ids[:4], val_ids[:2]
    targets = sc.targets_for(args.source)
    log.info("source=%s targets=%s train_n=%d val_n=%d", args.source, targets, len(train_ids), len(val_ids))

    log.info("loading train pool into RAM...")
    t0 = time.time()
    train_pool = PatientPool(train_ids)
    val_pool = PatientPool(val_ids)
    log.info("pools loaded in %.0fs", time.time() - t0)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    ckpt_dir = sc.CKPT_DIR / args.source
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = sc.OUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    curve_path = log_dir / f"train_curve_{args.source}.csv"

    start_iter = 0
    if args.resume and (ckpt_dir / "latest.pt").exists():
        ck = torch.load(ckpt_dir / "latest.pt", map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_iter = ck["iter"]
        log.info("resumed from iter %d", start_iter)

    write_header = not curve_path.exists()
    csv_f = open(curve_path, "a", newline="")
    writer = csv.writer(csv_f)
    if write_header:
        writer.writerow(["iter", "train_l1", "val_l1", "elapsed_s"])

    # Fixed seeded val batches so the val curve is comparable across evals/resumes.
    val_rng = np.random.default_rng(12345)
    val_batches = [val_pool.batch(args.source, targets, val_rng, batch=BATCH)
                   for _ in range(N_VAL_PATCHES // BATCH)]

    def run_val():
        model.eval()
        losses = []
        with torch.no_grad():
            for x, y, m in val_batches:
                x, y, m = x.to(device), y.to(device), m.to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    pred = model(x)
                    losses.append(masked_l1(pred, y, m).item())
        model.train()
        return float(np.mean(losses))

    rng = np.random.default_rng(sc.SEED + hash(args.source) % 1000)
    running = []
    t0 = time.time()
    model.train()
    for it in range(start_iter, args.iters):
        x, y, m = train_pool.batch(args.source, targets, rng)
        x, y, m = x.to(device), y.to(device), m.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred = model(x)
            loss = masked_l1(pred, y, m)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        running.append(loss.item())

        if (it + 1) % EVAL_EVERY == 0 or (it + 1) == args.iters:
            val_l1 = run_val()
            train_l1 = float(np.mean(running))
            running = []
            elapsed = time.time() - t0
            log.info("iter %d/%d train_l1=%.4f val_l1=%.4f (%.0fs)",
                      it + 1, args.iters, train_l1, val_l1, elapsed)
            writer.writerow([it + 1, train_l1, val_l1, f"{elapsed:.0f}"])
            csv_f.flush()
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it + 1,
                        "source": args.source, "targets": targets}, ckpt_dir / "latest.pt")

    torch.save({"model": model.state_dict(), "iter": args.iters, "source": args.source,
                "targets": targets}, ckpt_dir / "final.pt")
    csv_f.close()
    log.info("done: checkpoint at %s", ckpt_dir / "final.pt")


if __name__ == "__main__":
    main()
