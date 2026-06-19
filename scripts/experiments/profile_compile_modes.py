"""
Find the pure-code (no logic change) ceiling for the network step:
  - eager vs torch.compile {default, reduce-overhead, max-autotune}
  - compiling the deep-supervision loss too
  - separate forward / loss / backward timing
  - graph-break check via torch._dynamo

Run: run_job --gpus 1 --slot 0 --wait -- .venv/bin/python scripts/experiments/profile_compile_modes.py
"""
from __future__ import annotations
import os, sys, time, copy
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PREP = ROOT / "datasets/on-harmony/2_nnUNet_on-harmony/preprocessed/Dataset030_OnHarmonyT1w"
PATCH = (128, 160, 112)
NB = 2
N_WARM, N_IT = 8, 20


def build_raw_trainer():
    os.environ["nnUNet_compile"] = "0"   # we compile manually below
    os.environ["NNUNET_PLOT_EVERY"] = "100000"
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from on_harmony.trainers.v26_6 import nnUNetTrainerOnHarmonyV26_6
    plans = load_json(str(PREP / "nnUNetPlans.json")); plans.setdefault("continue_training", False)
    dj = load_json(str(PREP / "dataset.json"))
    tr = nnUNetTrainerOnHarmonyV26_6(plans, "3d_fullres", 0, dj, device=torch.device("cuda"))
    tr.initialize(); tr._setup_v26_training()
    return tr


def sync(): torch.cuda.synchronize()


def time_step(net, loss_fn, x, tgts, opt, scaler):
    from torch.amp import autocast
    f = l_ = b = 0.0
    opt.zero_grad(set_to_none=True)
    sync(); t0 = time.perf_counter()
    with autocast("cuda", enabled=True):
        out = net(x)
    sync(); t1 = time.perf_counter()
    with autocast("cuda", enabled=True):
        loss = loss_fn(out, tgts)
    sync(); t2 = time.perf_counter()
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()
    sync(); t3 = time.perf_counter()
    return (t1-t0), (t2-t1), (t3-t2)


def bench(name, net, loss_fn, tr):
    dev = tr.device
    nseg = tr.label_manager.num_segmentation_heads
    x = torch.randn(NB, 1, *PATCH, device=dev)
    from torch.amp import autocast
    with torch.no_grad(), autocast("cuda", enabled=True):
        outs = net(x)
    outs = outs if isinstance(outs, (list, tuple)) else [outs]
    tgts = [torch.randint(0, nseg, (NB, 1, *o.shape[2:]), device=dev) for o in outs]
    opt = tr.optimizer; scaler = tr.grad_scaler
    try:
        for _ in range(N_WARM): time_step(net, loss_fn, x, tgts, opt, scaler)
        F = LO = B = 0.0
        for _ in range(N_IT):
            f, l_, b = time_step(net, loss_fn, x, tgts, opt, scaler)
            F += f; LO += l_; B += b
        F, LO, B = F/N_IT*1000, LO/N_IT*1000, B/N_IT*1000
        print(f"{name:34s} fwd {F:6.1f}  loss {LO:6.1f}  bwd {B:6.1f}  total {F+LO+B:6.1f} ms")
        return F+LO+B
    except Exception as e:
        print(f"{name:34s} FAILED: {str(e)[:70]}")
        return None


def main():
    tr = build_raw_trainer()
    base_net = tr.network
    base_loss = tr.loss

    print(f"patch={PATCH} batch={NB}\n")
    results = {}
    results["eager"] = bench("eager", base_net, base_loss, tr)

    for mode in ["default", "reduce-overhead", "max-autotune"]:
        try:
            net_c = torch.compile(base_net, mode=mode)
            results[f"compile[{mode}]"] = bench(f"compile[{mode}]", net_c, base_loss, tr)
            torch._dynamo.reset()
        except Exception as e:
            print(f"compile[{mode}] build FAILED: {str(e)[:70]}")

    # also: compile net (default) + compile loss
    try:
        net_c = torch.compile(base_net)
        loss_c = torch.compile(base_loss)
        results["compile[default]+loss"] = bench("compile[default]+loss", net_c, loss_c, tr)
        torch._dynamo.reset()
    except Exception as e:
        print(f"compile+loss FAILED: {str(e)[:70]}")

    print("\n=== summary (vs eager) ===")
    eager = results.get("eager")
    for k, v in results.items():
        if v and eager:
            print(f"{k:34s} {v:6.1f} ms   {eager/v:4.2f}x")


if __name__ == "__main__":
    main()
