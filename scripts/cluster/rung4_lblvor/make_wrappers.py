#!/usr/bin/env python3
"""
Generate the ladder's NEW noise-fill rung 4 ("+Voronoi", noise fill) wrappers, 2026-10-07, by COPYING each
setting's current rung-4 wrapper and changing ONLY the transform config:

  AUGLAB_PARAMS_GPU_JSON  transform_params_gpu_baseline_kmeans_label_remap_voronoi_spatialDA_train050.json
                       -> transform_params_gpu_baseline_kmeans_label_remap_voronoi_lblvor_spatialDA_train050.json
  METHOD / LOG_DIR        <rung-4 method>  ->  <rung-4 method>_lblvor

WHY: in the rung-4 noise fill (label_fill_noise=true) a label whose label step fires is refilled as ONE noise level,
so the Voronoi sub-regions vanish inside remapped labels, while the real-fill rung 5 keeps them (its label step is an
affine re-remap). The fill swap 4->5 therefore also added Voronoi inside labels. The lblvor config keeps the
SynthSeg-style label refill (k-means structure inside the label still replaced) but splits the refilled label into
spatial Voronoi cells (same s_choices / skip_sub_parc_prob), each with its own noise level (Paul, 2026-10-07).

No DualVal (Paul): a DualVal rung-4 trainer is swapped for the dataset's plain AugLabDefault trainer (checkpoint_best
on real validation, as DualVal's val000 mirror) and AUGLAB_VAL_PARAMS_GPU_JSON is emptied.

Usage:  .venv/bin/python scripts/cluster/rung4_lblvor/make_wrappers.py [--write]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
T = REPO / "benchmark/02_tasks"
OLD_CFG = "transform_params_gpu_baseline_kmeans_label_remap_voronoi_spatialDA_train050.json"
NEW_CFG = "transform_params_gpu_baseline_kmeans_label_remap_voronoi_lblvor_spatialDA_train050.json"
SUFFIX = "_lblvor"

# (task dir, dataset, contrast tag, current rung-4 wrapper) -- the wrapper behind each ladder's run_keys[3]
SETTINGS = [
    ("brain_tumor", "brats2024-glioma", "t1n", "04_42_train_t1n_baseline_kmeans_label_remap_voronoi.sh"),
    ("brain_tumor", "brats2024-glioma", "t2w", "04_46_train_t2w_baseline_kmeans_label_remap_voronoi.sh"),
    ("brain_tumor", "brats2024-glioma", "t2f", "04_59_train_t2f_baseline_kmeans_label_remap_voronoi.sh"),
    ("brain_tumor", "brats2024-glioma", "t1c", "04_79_train_t1c_baseline_kmeans_label_remap_voronoi.sh"),
    ("brain_healthy", "on-harmony", "t1w", "04_30_train_t1w_baseline_kmeans_label_remap_voronoi.sh"),
    ("brain_healthy", "on-harmony", "t2w", "04_34_train_t2w_baseline_kmeans_label_remap_voronoi.sh"),
    ("brain_healthy", "on-harmony", "dwi_ap", "04_45_train_dwi_ap_baseline_kmeans_label_remap_voronoi.sh"),
    ("brain_ms", "open-ms", "flair", "04_25_train_baseline_kmeans_label_remap_voronoi_train050_val000.sh"),
    ("brain_ms", "open-ms", "t1w", "04_37_train_t1w_baseline_kmeans_label_remap_voronoi_train050_val000.sh"),
    ("abdomen_healthy", "chaos", "t1in", "04_47_train_t1in_baseline_kmeans_label_remap_voronoi.sh"),
    ("abdomen_healthy", "chaos", "t2spir", "04_57_train_t2spir_baseline_kmeans_label_remap_voronoi.sh"),
    ("mandible_healthy", "toothfairy2", "cbct", "04_10_train_baseline_kmeans_label_remap_voronoi.sh"),
    ("breast_cancer", "ispy2", "t1wce", "04_18_train_t1wce_baseline_kmeans_label_remap_voronoi.sh"),
    ("breast_cancer", "ispy2", "t2w", "04_23_train_t2w_baseline_kmeans_label_remap_voronoi.sh"),
    ("pelvis_healthy", "totalseg-pelvic", "ct", "04_17_train_ct_baseline_kmeans_label_remap_voronoi.sh"),
    ("pelvis_healthy", "totalseg-pelvic", "mri", "04_21_train_mri_baseline_kmeans_label_remap_voronoi.sh"),
    ("pancreas_disease", "pansegdata", "t1wce", "04_17_train_t1wce_baseline_kmeans_label_remap_voronoi.sh"),
    ("pancreas_disease", "pansegdata", "t2w", "04_21_train_t2w_baseline_kmeans_label_remap_voronoi.sh"),
]


def derive(src: str) -> tuple[str, str, str]:
    m = re.search(r'^METHOD="([^"]+)"', src, re.M); assert m, "no METHOD line"
    old_method = m.group(1); method = old_method + SUFFIX
    tm = re.search(r'^TRAINER="([^"]+)"', src, re.M); assert tm, "no TRAINER line"
    trainer = tm.group(1)
    new_trainer = trainer.replace("AugLabDualVal", "AugLabDefault")
    out = []
    for line in src.splitlines():
        if not line.lstrip().startswith("#"):
            if line.startswith("METHOD="):
                line = f'METHOD="{method}"'
            elif line.startswith("TRAINER="):
                line = f'TRAINER="{new_trainer}"'
            elif "LOG_DIR=" in line:
                line = line.replace(old_method, method)
            elif "AUGLAB_PARAMS_GPU_JSON=" in line:
                assert OLD_CFG in line, f"unexpected rung-4 config: {line}"
                line = line.replace(OLD_CFG, NEW_CFG)
            elif "AUGLAB_VAL_PARAMS_GPU_JSON=" in line and new_trainer != trainer:
                line = 'export AUGLAB_VAL_PARAMS_GPU_JSON=""   # plain trainer: no synthetic-validation mirror'
        out.append(line)
    text = "\n".join(out) + "\n"
    assert NEW_CFG in text and f'METHOD="{method}"' in text and OLD_CFG not in text
    return text, method, new_trainer


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    for task, ds, tag, w4 in SETTINGS:
        d = T / task / ds / f"5_scripts_{ds}" / "04_train"
        src = (d / w4).read_text()
        body = src.split("\n", 1)[1] if src.startswith("#!") else src
        text, method, trainer = derive(body)
        existing = [p for p in d.glob("04_*.sh") if f"rung4_lblvor/make_wrappers.py from {w4} " in p.read_text()]
        if existing:
            print(f"exists   {existing[0].relative_to(REPO)}"); continue
        nums = sorted(int(p.name[3:5]) for p in d.glob("04_[0-9][0-9]_*.sh"))
        n = next((i for i in range(max(nums) + 1, 100)), None)
        if n is None:                                   # numbering full (brats): highest unused slot
            n = max(i for i in range(100) if i not in nums)
        name = f"04_{n:02d}_train_{tag}_{method}.sh" if tag not in method else f"04_{n:02d}_train_{method}.sh"
        header = (f"#!/usr/bin/env bash\n# GENERATED by scripts/cluster/rung4_lblvor/make_wrappers.py from {w4} (do not hand-edit):\n"
                  f"# ladder rung 4 (+Voronoi, NOISE fill) with Voronoi cells kept inside noise-refilled labels ({NEW_CFG});\n"
                  f"# otherwise identical to {w4} (trainer {trainer}: checkpoint_best on real validation, no DualVal).\n")
        out = d / name
        print(f"{'write' if a.write else 'would '} {out.relative_to(REPO)}  METHOD={method}  TRAINER={trainer}")
        if a.write:
            out.write_text(header + text)


if __name__ == "__main__":
    main()
