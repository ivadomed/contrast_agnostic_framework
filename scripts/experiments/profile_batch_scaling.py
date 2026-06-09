"""
Measure network fwd/bwd time vs batch size + peak GPU memory, to see whether
batch=2 underutilises the A6000 (=> headroom to trade iters for batch).

Run: set_slot 0 .venv/bin/python scripts/experiments/profile_batch_scaling.py
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PREP = ROOT / "datasets/on-harmony/2_nnUNet_on-harmony/preprocessed/Dataset030_OnHarmonyT1w"
PATCH = (128, 160, 112)
BATCHES = [1, 2, 4, 6, 8]
N_WARM, N_IT = 4, 12


def build_net():
    os.environ["nnUNet_compile"] = "1"
    os.environ["NNUNET_PLOT_EVERY"] = "100000"
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from on_harmony.trainers.v26_6 import nnUNetTrainerOnHarmonyV26_6
    plans = load_json(str(PREP / "nnUNetPlans.json")); plans.setdefault("continue_training", False)
    dj = load_json(str(PREP / "dataset.json"))
    tr = nnUNetTrainerOnHarmonyV26_6(plans, "3d_fullres", 0, dj, device=torch.device("cuda"))
    tr.initialize(); tr._setup_v26_training()
    return tr


def main():
    from torch.amp import autocast
    tr = build_net(); dev = tr.device
    nseg = tr.label_manager.num_segmentation_heads
    print(f"{'batch':>6} {'ms/iter':>9} {'ms/sample':>11} {'peak_GB':>9} {'samp/s':>8}")
    for nb in BATCHES:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.randn(nb, 1, *PATCH, device=dev)
            # deep-supervision targets at network output scales
            with autocast(dev.type, enabled=True):
                outs = tr.network(x)
            tgts = [torch.randint(0, nseg, (nb, 1, *o.shape[2:]), device=dev) for o in (outs if isinstance(outs,(list,tuple)) else [outs])]
            def step():
                tr.optimizer.zero_grad(set_to_none=True)
                with autocast(dev.type, enabled=True):
                    o = tr.network(x); l = tr.loss(o, tgts)
                tr.grad_scaler.scale(l).backward()
                tr.grad_scaler.step(tr.optimizer); tr.grad_scaler.update()
            for _ in range(N_WARM): step()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(N_IT): step()
            torch.cuda.synchronize(); dt = (time.perf_counter()-t0)/N_IT
            peak = torch.cuda.max_memory_allocated()/1e9
            print(f"{nb:>6} {dt*1000:>9.1f} {dt*1000/nb:>11.1f} {peak:>9.1f} {nb/dt:>8.1f}")
        except RuntimeError as e:
            print(f"{nb:>6}  OOM / error: {str(e)[:60]}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
