#!/usr/bin/env python3
"""Build the 4 derived, synth-only + spatial-OFF AugLab GPU configs for the
texture_analysis_lvl_1 generation.

Each derived config is written to ``../configs/`` so the originals under
``AugLab/auglab/configs/`` stay untouched and our exact settings are reproducible.

Design rules (see GENERATION_PROMPT.md):
  * palette         -> ONLY ImageContrastV26_6_2GPUTransform @ 1.0 (already clean, copied verbatim)
  * synthseg_em     -> SynthSeg @ 1.0 (EM completion ON), every OTHER transform zeroed
  * synthseg_noem   -> SynthSeg @ 1.0 (EM completion OFF), every OTHER transform zeroed
  * auglab_default  -> keep intensity augs, zero ONLY spatial ones (Flip, Affine)

Spatial notes:
  * The RandomSynthSegGPU wrapper already forces apply_affine=False,
    apply_nonlinear=False, flipping=False, output_shape=None (see
    auglab/transforms/synthseg/transforms.py:85). So the SynthSeg block's internal
    scaling/rotation/shearing/nonlin/flipping params are inert w.r.t. voxel geometry
    and the output stays grid-aligned. We therefore preserve the SynthSeg block
    VERBATIM (it also encodes the EM-vs-anatomical + GMM priors) and only zero the
    sibling transforms.
  * nnUNetSpatialTransform is NOT consumed by AugTransformsGPU._build_transforms at
    all (no builder branch), so it is inert regardless.
  * SimulateLowResTransform / AcqTransform are blur-then-resample-back to the SAME
    grid -> grid-preserving, kept.
  * FlipTransform (mirror) and AffineTransform (rotate/scale/shear/translate) DO move
    voxels -> forced to probability 0 everywhere.
"""
import json
from pathlib import Path

SRC_CFG = Path(__file__).resolve().parents[5] \
    / "sub-workspaces/auglab_workspace/AugLab/auglab/configs"
OUT_CFG = Path(__file__).resolve().parents[1] / "configs"
OUT_CFG.mkdir(parents=True, exist_ok=True)

# Transforms that move voxels -> must always be off for alignment.
SPATIAL_KEYS = {"FlipTransform", "AffineTransform"}

# Generative / image-replacing transforms (used to assert "exactly one active").
GENERATIVE_KEYS = {
    "SynthSeg",
    "ImageContrastV26_6_2GPUTransform",
    "ImageContrastGPUTransform",
    "RandomDomainTransferGPU",
    "DomainTransferTransform",
}


def load(name):
    with open(SRC_CFG / name) as f:
        return json.load(f)


def prob_of(block):
    return block.get("probability") if isinstance(block, dict) else None


def zero_prob(block):
    if isinstance(block, dict) and "probability" in block:
        block["probability"] = 0.0


def assert_spatial_off(cfg, tag):
    for k in SPATIAL_KEYS:
        p = prob_of(cfg.get(k, {}))
        assert not p, f"[{tag}] spatial transform {k} active (prob={p})"


def assert_one_generative(cfg, tag):
    active = [k for k in GENERATIVE_KEYS if prob_of(cfg.get(k, {})) not in (None, 0, 0.0)]
    assert len(active) == 1, f"[{tag}] expected exactly 1 generative transform, got {active}"
    return active[0]


def write(cfg, name):
    out = OUT_CFG / name
    with open(out, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"  wrote {out.relative_to(OUT_CFG.parents[1])}")


def build_palette():
    # Already clean: only ImageContrastV26_6_2GPUTransform @ 1.0, rest @ 0.
    cfg = load("transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json")
    assert_spatial_off(cfg, "palette")
    assert_one_generative(cfg, "palette") == "ImageContrastV26_6_2GPUTransform"
    write(cfg, "palette.json")


def build_synthseg(src_name, out_name, tag):
    cfg = load(src_name)
    assert prob_of(cfg["SynthSeg"]) == 1.0, f"{tag}: SynthSeg not @1.0"
    # Zero EVERY transform except SynthSeg (incl. FlipTransform, SimulateLowRes,
    # BiasField, Noise, ... — this must be a pure GMM-fill of the label map).
    for k, v in cfg.items():
        if k == "SynthSeg":
            continue
        zero_prob(v)
    assert_spatial_off(cfg, tag)
    assert_one_generative(cfg, tag) == "SynthSeg"
    write(cfg, out_name)


def build_auglab_default():
    # Keep the intensity/filter augs; zero ONLY the voxel-moving spatial ones.
    cfg = load("transform_params_gpu_default01-23.json")
    for k in SPATIAL_KEYS:
        if k in cfg:
            zero_prob(cfg[k])
    assert_spatial_off(cfg, "auglab_default")
    # No image-from-GMM/contrast generator here by design (traditional aug pipeline);
    # RedistributeSeg is a seg-conditioned intensity redistribution, kept as intensity.
    gen = [k for k in GENERATIVE_KEYS if prob_of(cfg.get(k, {})) not in (None, 0, 0.0)]
    assert gen == [], f"auglab_default should have no image-replacing generator, got {gen}"
    write(cfg, "auglab_default.json")


def main():
    print(f"Source configs: {SRC_CFG}")
    print(f"Output configs: {OUT_CFG}")
    build_palette()
    build_synthseg("transform_params_gpu_default01-23_Synthseg_EM.json",
                   "synthseg_em.json", "synthseg_em")
    build_synthseg("transform_params_gpu_default01-23_Synthseg.json",
                   "synthseg_noem.json", "synthseg_noem")
    build_auglab_default()
    print("All derived configs built and validated.")


if __name__ == "__main__":
    main()
