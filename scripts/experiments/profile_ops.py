"""
Operator-level profile of the network fwd/bwd (the 185 ms/step wall).
Finds which pytorch/cuDNN ops dominate — conv fwd, conv bwd (dgrad/wgrad),
InstanceNorm, elementwise, etc. — to locate a pytorch-level lever.

Run: run_job --gpus 1 --slot 0 --wait -- .venv/bin/python scripts/experiments/profile_ops.py [compile]
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PREP = ROOT / "datasets/on-harmony/2_nnUNet_on-harmony/preprocessed/Dataset030_OnHarmonyT1w"
PATCH = (128, 160, 112)
NB = 2


def build(compile_net):
    os.environ["nnUNet_compile"] = "1" if compile_net else "0"
    os.environ["NNUNET_PLOT_EVERY"] = "100000"
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from on_harmony.trainers.v26_6 import nnUNetTrainerOnHarmonyV26_6
    plans = load_json(str(PREP / "nnUNetPlans.json")); plans.setdefault("continue_training", False)
    dj = load_json(str(PREP / "dataset.json"))
    tr = nnUNetTrainerOnHarmonyV26_6(plans, "3d_fullres", 0, dj, device=torch.device("cuda"))
    tr.initialize(); tr._setup_v26_training()
    return tr


class FastInstanceNorm3d(torch.nn.Module):
    """channels_last-native InstanceNorm3d (identical math to torch's, but does
    NOT reshape → preserves memory format → no copy explosion)."""
    def __init__(self, src):
        super().__init__()
        self.eps = src.eps
        self.affine = src.affine
        if src.affine:
            self.weight = src.weight
            self.bias = src.bias
    def forward(self, x):
        var, mean = torch.var_mean(x, dim=(2, 3, 4), keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        if self.affine:
            x = x * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
        return x


def swap_instancenorm(module):
    import torch.nn as nn
    for name, child in module.named_children():
        if isinstance(child, nn.InstanceNorm3d):
            setattr(module, name, FastInstanceNorm3d(child))
        else:
            swap_instancenorm(child)


def main():
    compile_net = "compile" in sys.argv
    chlast = "channels_last" in sys.argv
    fastin = "fastin" in sys.argv
    from torch.amp import autocast
    tr = build(compile_net); dev = tr.device
    if fastin:
        swap_instancenorm(tr.network)
    if chlast:
        tr.network = tr.network.to(memory_format=torch.channels_last_3d)
    nseg = tr.label_manager.num_segmentation_heads
    x = torch.randn(NB, 1, *PATCH, device=dev)
    if chlast:
        x = x.to(memory_format=torch.channels_last_3d)
    with torch.no_grad(), autocast("cuda", enabled=True):
        outs = tr.network(x)
    outs = outs if isinstance(outs, (list, tuple)) else [outs]
    tgts = [torch.randint(0, nseg, (NB, 1, *o.shape[2:]), device=dev) for o in outs]

    def step():
        tr.optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=True):
            o = tr.network(x); l = tr.loss(o, tgts)
        tr.grad_scaler.scale(l).backward()
        tr.grad_scaler.step(tr.optimizer); tr.grad_scaler.update()

    for _ in range(8):  # warmup + compile
        step()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(10):
            step()
        torch.cuda.synchronize()

    print(f"\n=== compile={compile_net}  top ops by CUDA time (10 steps) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30,
                                    max_name_column_width=55))


if __name__ == "__main__":
    main()
