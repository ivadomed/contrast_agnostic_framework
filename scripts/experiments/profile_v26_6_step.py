"""
Profile the V26_6 training step to locate the epoch-time bottleneck.

Builds the REAL network from the Dataset030 plans, loads a REAL preprocessed
full volume, and times each component of the train_step with proper CUDA
synchronisation.  Reports per-component ms and the fwd/bwd vs. data split so we
know where to spend optimisation effort.

Run:  set_slot 0 .venv/bin/python scripts/experiments/profile_v26_6_step.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PREP = ROOT / "datasets/on-harmony/2_nnUNet_on-harmony/preprocessed/Dataset030_OnHarmonyT1w"
N_WARMUP = 5
N_ITERS = 20


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __init__(self):
        self.acc = {}

    def __call__(self, name):
        return _Block(self, name)

    def add(self, name, dt):
        self.acc.setdefault(name, []).append(dt)

    def report(self, n_iter):
        print(f"\n{'component':32s} {'ms/iter':>10s} {'% of step':>10s}")
        print("-" * 56)
        total = sum(np.mean(v) for v in self.acc.values())
        for name, vals in self.acc.items():
            m = np.mean(vals)
            print(f"{name:32s} {m*1000:10.2f} {100*m/total:9.1f}%")
        print("-" * 56)
        print(f"{'TOTAL / iter':32s} {total*1000:10.2f}")
        print(f"{'=> epoch (150 iters)':32s} {total*150:10.2f} s")


class _Block:
    def __init__(self, timer, name):
        self.timer, self.name = timer, name

    def __enter__(self):
        _sync(); self.t0 = time.perf_counter(); return self

    def __exit__(self, *a):
        _sync(); self.timer.add(self.name, time.perf_counter() - self.t0)


def load_real_volume():
    import blosc2
    from batchgenerators.utilities.file_and_folder_operations import load_json
    cands = sorted((PREP / "nnUNetPlans_3d_fullres").glob("*_T1w.b2nd"))
    f = cands[0]
    data = np.asarray(blosc2.open(urlpath=str(f), mode="r"))
    seg = np.asarray(blosc2.open(urlpath=str(f).replace(".b2nd", "_seg.b2nd"), mode="r"))
    print(f"real volume: data {data.shape} seg {seg.shape} dtype {data.dtype}")
    # data is (C, D, H, W); take channel 0 -> (1,1,D,H,W)
    d = torch.from_numpy(np.ascontiguousarray(data[0:1][None])).float()
    s = torch.from_numpy(np.ascontiguousarray(seg[0:1][None])).float()
    return d, s


def build_trainer(compile_net: bool):
    os.environ["nnUNet_compile"] = "1" if compile_net else "0"
    os.environ["NNUNET_PLOT_EVERY"] = "100000"
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from on_harmony.trainers.v26_6 import nnUNetTrainerOnHarmonyV26_6

    plans = load_json(str(PREP / "nnUNetPlans.json"))
    plans.setdefault("continue_training", False)
    dataset_json = load_json(str(PREP / "dataset.json"))
    tr = nnUNetTrainerOnHarmonyV26_6(plans, "3d_fullres", 0, dataset_json,
                                     device=torch.device("cuda"))
    tr.initialize()
    tr._setup_v26_training()   # builds transforms, patch cfg, thread pool
    return tr


def profile(compile_net: bool):
    from src.synthesis.v26_6_synthesis import synthesize_volume_fast, gpu_spatial_augment
    from src.nnunet.transforms.synth_aug import random_crop_pair

    tr = build_trainer(compile_net)
    dev = tr.device
    chlast = os.environ.get("PROFILE_CHANNELS_LAST", "0") == "1"
    if chlast:
        tr.network = tr.network.to(memory_format=torch.channels_last_3d)
    data_v, seg_v = load_real_volume()
    data_v, seg_v = data_v.to(dev), seg_v.to(dev)
    ps = tr._patch_size_cfg
    nb = tr.batch_size
    print(f"compile={compile_net}  channels_last={chlast}  patch={ps}  batch={nb}")

    T = Timer()

    def one_step(measure: bool):
        tm = T if measure else None
        ctxn = (lambda n: tm(n)) if measure else (lambda n: _Null())

        with ctxn("01_norm"):
            vmin, vmax = data_v.min(), data_v.max()
            image_01 = ((data_v - vmin) / (vmax - vmin + 1e-7)).clamp(0, 1)
        with ctxn("02_gpu_spatial_aug"):
            image_01b, target = gpu_spatial_augment(image_01, seg_v)
        with ctxn("03_synthesize_volume"):
            synth_z, _ = synthesize_volume_fast(image_01b)
        with ctxn("04_random_crop"):
            patches, patches_seg = random_crop_pair(synth_z, target, ps, n_crops=nb)
        with ctxn("05_to_cpu"):
            patches_cpu = patches.cpu().float()
            patches_seg_cpu = patches_seg.cpu().to(torch.int16)
        with ctxn("06_cpu_transforms"):
            dl, sl = [], []
            for b in range(nb):
                t = tr._train_transforms(image=patches_cpu[b], segmentation=patches_seg_cpu[b])
                dl.append(t["image"]); sl.append(t["segmentation"])
        with ctxn("07_to_gpu"):
            data_aug = torch.stack(dl).to(dev, non_blocking=True)
            if chlast:
                data_aug = data_aug.to(memory_format=torch.channels_last_3d)
            if isinstance(sl[0], (list, tuple)):
                ns = len(sl[0])
                seg_aug = [torch.stack([sl[b][s] for b in range(nb)]).to(dev) for s in range(ns)]
            else:
                seg_aug = torch.stack(sl).to(dev)
        with ctxn("08_forward"):
            from torch.amp import autocast
            tr.optimizer.zero_grad(set_to_none=True)
            with autocast(dev.type, enabled=True):
                output = tr.network(data_aug)
                l = tr.loss(output, seg_aug)
        with ctxn("09_backward"):
            tr.grad_scaler.scale(l).backward()
        with ctxn("10_optim_step"):
            tr.grad_scaler.unscale_(tr.optimizer)
            torch.nn.utils.clip_grad_norm_(tr.network.parameters(), 12)
            tr.grad_scaler.step(tr.optimizer)
            tr.grad_scaler.update()
        with ctxn("11_loss_sync"):
            _ = l.detach().cpu().numpy()

    print(f"warmup ({N_WARMUP} iters, triggers compile)...")
    t0 = time.perf_counter()
    for _ in range(N_WARMUP):
        one_step(measure=False)
    _sync()
    print(f"warmup done in {time.perf_counter()-t0:.1f}s")

    for _ in range(N_ITERS):
        one_step(measure=True)
    T.report(N_ITERS)


class _Null:
    def __enter__(self): return self
    def __exit__(self, *a): pass


if __name__ == "__main__":
    compile_net = os.environ.get("PROFILE_COMPILE", "1") == "1"
    profile(compile_net)
