"""
Proof-of-concept: 3-D Winograd F(2,2,2 ; 3,3,3) convolution.

cuDNN implements Winograd only for 2-D convs; the network's 3x3x3 stride-1 convs
fall back to implicit-GEMM. 3-D Winograd needs 4^3=64 elementwise multiplies per
2^3=8 outputs vs direct 27*8=216 → 3.375x fewer multiplies, and it is EXACT.

This script: (1) verifies numerical equivalence to F.conv3d, (2) benchmarks
Winograd vs cuDNN for representative layers (fp16, tensor cores).

Run: run_job --gpus 1 --slot 0 --wait -- .venv/bin/python scripts/experiments/winograd3d_poc.py
"""
from __future__ import annotations
import time
import torch
import torch.nn.functional as F

dev = torch.device("cuda")

# F(2,3) Winograd transform matrices (Lavin & Gray)
G  = torch.tensor([[1.0, 0.0, 0.0],
                   [0.5, 0.5, 0.5],
                   [0.5, -0.5, 0.5],
                   [0.0, 0.0, 1.0]], device=dev)
BT = torch.tensor([[1.0, 0.0, -1.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, -1.0, 1.0, 0.0],
                   [0.0, 1.0, 0.0, -1.0]], device=dev)
AT = torch.tensor([[1.0, 1.0, 1.0, 0.0],
                   [0.0, 1.0, -1.0, -1.0]], device=dev)


def _apply_last3(t, M):
    """Apply matrix M (K,I) to each of the last 3 axes of t (each currently size I)."""
    t = torch.matmul(t, M.t())                       # last axis  I->K
    t = torch.matmul(t.transpose(-1, -2), M.t()).transpose(-1, -2)   # mid axis
    t = torch.matmul(t.transpose(-1, -3), M.t()).transpose(-1, -3)   # first of the 3
    return t


def winograd_conv3d(x, w):
    """x:(N,Cin,D,H,W) w:(Cout,Cin,3,3,3); stride1 'same' (pad 1). Returns (N,Cout,D,H,W)."""
    N, Cin, D, H, W = x.shape
    Cout = w.shape[0]
    Dt, Ht, Wt = (D + 1) // 2, (H + 1) // 2, (W + 1) // 2
    Dp, Hp, Wp = Dt * 2, Ht * 2, Wt * 2
    xp = F.pad(x, (1, Wp - W + 1, 1, Hp - H + 1, 1, Dp - D + 1))
    tiles = xp.unfold(2, 4, 2).unfold(3, 4, 2).unfold(4, 4, 2)      # (N,Cin,Dt,Ht,Wt,4,4,4)
    dt = x.dtype
    g, bt, at = G.to(dt), BT.to(dt), AT.to(dt)

    # transforms applied axis-by-axis (small matmuls via cuBLAS), not one giant einsum
    U = _apply_last3(w.to(dt), g)                                  # (Cout,Cin,4,4,4)
    V = _apply_last3(tiles, bt)                                    # (N,Cin,Dt,Ht,Wt,4,4,4)

    # batched GEMM over the 64 Winograd coords: M[64,T,Cout] = V[64,T,Cin] @ U[64,Cin,Cout]
    T = N * Dt * Ht * Wt
    Vp = V.permute(0, 2, 3, 4, 5, 6, 7, 1).reshape(T, 64, Cin).permute(1, 0, 2)  # (64,T,Cin)
    Up = U.permute(2, 3, 4, 1, 0).reshape(64, Cin, Cout)                          # (64,Cin,Cout)
    M = torch.bmm(Vp, Up)                                                         # (64,T,Cout)
    M = M.permute(1, 2, 0).reshape(N, Dt, Ht, Wt, Cout, 4, 4, 4)

    Y = _apply_last3(M, at)                                        # (...,2,2,2)
    Y = Y.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(N, Cout, Dp, Hp, Wp)
    return Y[:, :, :D, :H, :W].contiguous()


def check_equiv():
    torch.manual_seed(0)
    x = torch.randn(1, 16, 24, 28, 20, device=dev)
    w = torch.randn(20, 16, 3, 3, 3, device=dev)
    ref = F.conv3d(x, w, padding=1)
    out = winograd_conv3d(x, w)
    md = (ref - out).abs().max().item()
    rel = ((ref - out).abs().mean() / ref.abs().mean()).item()
    print(f"equivalence (fp32): max|Δ|={md:.2e}  mean rel={rel:.2e}  "
          f"{'PASS' if rel < 1e-5 else 'FAIL'}")


def bench(Cin, Cout, D, H, W, dtype, n=30):
    from torch.amp import autocast
    x = torch.randn(1, Cin, D, H, W, device=dev)
    w = torch.randn(Cout, Cin, 3, 3, 3, device=dev)

    def run_cudnn():
        with autocast("cuda", dtype=dtype):
            return F.conv3d(x, w, padding=1)

    def run_wino():
        with autocast("cuda", dtype=dtype):
            return winograd_conv3d(x, w)

    for fn in (run_cudnn, run_wino):
        for _ in range(5): fn()
    torch.cuda.synchronize()
    res = {}
    for name, fn in (("cudnn", run_cudnn), ("winograd", run_wino)):
        t0 = time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize()
        res[name] = (time.perf_counter() - t0) / n * 1000
    print(f"  C{Cin}->{Cout} {D}x{H}x{W} {str(dtype).split('.')[-1]:>7}: "
          f"cudnn {res['cudnn']:6.2f}ms  winograd {res['winograd']:6.2f}ms  "
          f"speedup {res['cudnn']/res['winograd']:.2f}x")


if __name__ == "__main__":
    check_equiv()
    print("\nfp16 (tensor cores) — representative stride-1 3x3x3 layers:")
    bench(32, 32, 128, 160, 112, torch.float16)
    bench(64, 64, 64, 80, 56, torch.float16)
    bench(128, 128, 32, 40, 28, torch.float16)
    bench(256, 256, 16, 20, 14, torch.float16)
    bench(320, 320, 8, 10, 7, torch.float16)
