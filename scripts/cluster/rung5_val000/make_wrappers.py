#!/usr/bin/env python3
"""
Generate the val000 ladder "real fill" wrappers (rung 5 = PALETTE alone, and the PV variant)
by COPYING each setting's noise-fill wrapper (rung 4) and changing ONLY the fill:

  METHOD                 -> v26_6_2_train050_val000          (pv: v26_6_2_pv_train050_val000)
  AUGLAB_PARAMS_GPU_JSON -> transform_params_gpu_v26_6_2_synth_spatialDA_train050.json
                            (pv: transform_params_gpu_v26_6_2_pv_synth_spatialDA_train050.json)
  LOG_DIR                -> same with the method swapped
  AUGLAB_VAL_PARAMS_GPU_JSON (only DualVal trainers use it, for their val100 MIRROR)
                         -> transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json

So trainer class, epochs, data, env and checkpoint selection are those of rung 4 by
construction: the real-fill vs noise-fill comparison differs in the fill only.

WHY (2026-10-05): the old real-fill wrappers (*_v26_6_2_train050_val100.sh) used ValSynth
trainers, which run every validation batch through VALsynthonly (PALETTE at p=1) and so pick
checkpoint_best on SYNTHETIC validation images, while rung 4 picks it on real ones. Those
wrappers are now guarded (they refuse to run).

Usage:  .venv/bin/python scripts/cluster/rung5_val000/make_wrappers.py [--write]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
T = REPO / "benchmark/02_tasks"
CFG = "transform_params_gpu_baseline_kmeans_label_remap_voronoi_spatialDA_train050.json"
VAL_OLD = "transform_params_gpu_VALsynthonly_kmeans_label_remap_voronoi.json"
VAL_NEW = "transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"

# (task dir, dataset, contrast tag used in the new file name, rung-4 wrapper basename)
SETTINGS = [
    ("brain_tumor", "brats2024-glioma", "t1n", "04_42_train_t1n_baseline_kmeans_label_remap_voronoi.sh"),
    ("brain_tumor", "brats2024-glioma", "t2w", "04_46_train_t2w_baseline_kmeans_label_remap_voronoi.sh"),
    ("brain_tumor", "brats2024-glioma", "t2f", "04_59_train_t2f_baseline_kmeans_label_remap_voronoi.sh"),
    ("brain_tumor", "brats2024-glioma", "t1c", "04_79_train_t1c_baseline_kmeans_label_remap_voronoi.sh"),
    ("brain_healthy", "on-harmony", "t1w", "04_30_train_t1w_baseline_kmeans_label_remap_voronoi.sh"),
    ("brain_healthy", "on-harmony", "t2w", "04_34_train_t2w_baseline_kmeans_label_remap_voronoi.sh"),
    ("brain_healthy", "on-harmony", "dwi_ap", "04_45_train_dwi_ap_baseline_kmeans_label_remap_voronoi.sh"),
    ("brain_ms", "open-ms", "t1w", "04_37_train_t1w_baseline_kmeans_label_remap_voronoi_train050_val000.sh"),
    ("abdomen_healthy", "chaos", "t2spir", "04_57_train_t2spir_baseline_kmeans_label_remap_voronoi.sh"),
    ("mandible_healthy", "toothfairy2", "cbct", "04_10_train_baseline_kmeans_label_remap_voronoi.sh"),
    ("breast_cancer", "ispy2", "t1wce", "04_18_train_t1wce_baseline_kmeans_label_remap_voronoi.sh"),
    ("breast_cancer", "ispy2", "t2w", "04_23_train_t2w_baseline_kmeans_label_remap_voronoi.sh"),
    ("pelvis_healthy", "totalseg-pelvic", "ct", "04_17_train_ct_baseline_kmeans_label_remap_voronoi.sh"),
    ("pelvis_healthy", "totalseg-pelvic", "mri", "04_21_train_mri_baseline_kmeans_label_remap_voronoi.sh"),
    ("brain_stroke", "isles2022", "dwi", "04_17_train_dwi_baseline_kmeans_label_remap_voronoi.sh"),
    ("brain_stroke", "isles2022", "flair", "04_21_train_flair_baseline_kmeans_label_remap_voronoi.sh"),
    ("pancreas_disease", "pansegdata", "t1wce", "04_17_train_t1wce_baseline_kmeans_label_remap_voronoi.sh"),
    ("pancreas_disease", "pansegdata", "t2w", "04_21_train_t2w_baseline_kmeans_label_remap_voronoi.sh"),
]
VARIANTS = {  # name -> (method, train config)
    "real": ("v26_6_2_train050_val000", "transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"),
    "pv": ("v26_6_2_pv_train050_val000", "transform_params_gpu_v26_6_2_pv_synth_spatialDA_train050.json"),
}

# no ladder rung 6 (PV branch) for these; their pinned AugLab (pansegdata: 7b761b5) predates PV
NO_PV = {"isles2022", "pansegdata"}


def code_lines(text):
    return [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]


def derive(src_text: str, method: str, cfg: str) -> str:
    m = re.search(r'^METHOD="([^"]+)"', src_text, re.M)
    assert m, "no METHOD line"
    old_method = m.group(1)
    out = []
    for line in src_text.splitlines():
        if line.lstrip().startswith("#"):
            out.append(line)
            continue
        if line.startswith("METHOD="):
            line = f'METHOD="{method}"'
        elif "LOG_DIR=" in line:
            line = line.replace(old_method, method)
        elif "AUGLAB_PARAMS_GPU_JSON=" in line:
            assert CFG in line, f"unexpected rung-4 config: {line}"
            line = line.replace(CFG, cfg)
        elif "AUGLAB_VAL_PARAMS_GPU_JSON=" in line and VAL_OLD in line:
            line = line.replace(VAL_OLD, VAL_NEW)
        out.append(line)
    text = "\n".join(out) + "\n"
    bad = [l for l in code_lines(text) if "kmeans_label_remap_voronoi" in l]
    assert not bad, f"noise-fill reference left in code: {bad}"
    assert cfg in text and f'METHOD="{method}"' in text
    return text


HEADER = """#!/usr/bin/env bash
# GENERATED by scripts/cluster/rung5_val000/make_wrappers.py from {src} (do not hand-edit):
# ladder "real fill" rung{pv} -- an exact copy of the noise-fill rung (same trainer class, epochs,
# data, env and validation = real images / val000) with only the fill changed to PALETTE's
# real-intensity remap ({cfg}).
# Replaces the *_v26_6_2{pvtag}_train050_val100.sh wrapper, whose ValSynth trainer selected
# checkpoint_best on synthetic validation images (2026-10-05).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    made = []
    for task, ds, tag, w4 in SETTINGS:
        d = T / task / ds / f"5_scripts_{ds}" / "04_train"
        src = d / w4
        text = src.read_text()
        body = text.split("\n", 1)[1] if text.startswith("#!") else text
        nums = sorted(int(p.name[3:5]) for p in d.glob("04_[0-9][0-9]_*.sh"))
        for vname, (method, cfg) in VARIANTS.items():
            if vname == "pv" and ds in NO_PV:
                continue
            existing = list(d.glob(f"04_*_train_{tag}_{method}.sh"))
            if existing:
                print(f"exists   {existing[0].relative_to(REPO)}")
                made.append(existing[0]); continue
            n = (nums[-1] + 1) if nums else 1
            nums.append(n)
            dst = d / f"04_{n:02d}_train_{tag}_{method}.sh"
            out = HEADER.format(src=w4, cfg=cfg, pv=" + boundary PV" if vname == "pv" else "",
                                pvtag="_pv" if vname == "pv" else "") + derive(body, method, cfg)
            print(f"{'write' if a.write else 'plan '}    {dst.relative_to(REPO)}")
            if a.write:
                dst.write_text(out)
            made.append(dst)
    return made


if __name__ == "__main__":
    main()
