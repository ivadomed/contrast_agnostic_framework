# Task-level (panel-pooled) fill-swap, exactly as fig:ladder computes it; OLD = .bak (val100 rung 5) copies.
import sys, os
import numpy as np
sys.path.insert(0, "paper/scripts")
import make_per_contrast_curves as M
BAK = ".bak_20261006_pre_val000_rung5"
def run(use_bak, metric="dice"):
    panels = [(t, g, [(l, (r + BAK) if use_bak and os.path.exists(M.REPO / (r + BAK)) else r) for l, r in it]) for t, g, it in M.PANELS]
    loaded = [(t, g, [(l, M.load(r)) for l, r in it]) for t, g, it in panels]
    loaded = [(t, g, [(l, d) for l, d in it if d is not None]) for t, g, it in loaded]
    pooled = [M.panel_pooled([d for _, d in it], metric, True) for _, _, it in loaded]
    padj = M.holm([p for p, _ in pooled])
    out = {}
    for (t, g, it), pa, (_, dl) in zip(loaded, padj, pooled):
        avg = np.nanmean([np.asarray(d[metric][:M.N_RUNGS], float) for _, d in it], axis=0)
        out[t] = (avg[M.FILL - 1], dl, dl / avg[M.FILL - 1] * 100, pa)
    return out
new, old = run(False), run(True)
print(f"{'task':10s} {'noise-fill':>10s} | {'OLD d':>7s} {'rel%':>7s} {'p_holm':>9s} | {'NEW d':>7s} {'rel%':>7s} {'p_holm':>9s}")
for t in new:
    b, d, r, p = new[t]; _, d0, r0, p0 = old[t]
    print(f"{t:10s} {b:10.1f} | {d0:+7.2f} {r0:+7.1f} {p0:9.2g} | {d:+7.2f} {r:+7.1f} {p:9.2g}")
