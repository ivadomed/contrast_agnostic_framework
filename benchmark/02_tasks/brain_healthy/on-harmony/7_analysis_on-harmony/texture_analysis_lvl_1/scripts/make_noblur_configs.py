#!/usr/bin/env python3
"""Build the NO-BLUR ablation configs for texture_analysis_lvl_1.

Mechanism-isolation ablation: identical to the main derived configs in
`../configs/` EXCEPT blur and resolution degradation are disabled. Everything
else (K-means/Voronoi parcellation, signed-alpha, GMM priors, EM settings,
anatomical-label fill, all other intensity augs, spatial-OFF, normalization) is
untouched, so the only difference vs `data/generated/` is "no blur / no res".

Reads  : ../configs/<method>.json          (the validated with-blur configs)
Writes : ../configs_noblur/<method>.json   (blur/res disabled)

Per-method delta (the ONLY changes applied):
  palette         : ImageContrastV26_6_2GPUTransform.blur_sigmas -> [0.0]
                    (the GPU transform picks sigma via random.choice(blur_sigmas);
                    [0.0] => never blurs. The GPU op reads blur_sigmas straight
                    from this JSON and is self-contained — it does NOT import
                    src/synthesis/v26_6_synthesis.py:BLUR_SIGMAS — so no training
                    source is edited and there is nothing to revert.)
  synthseg_em     : SynthSeg.randomise_res -> False, blur_range -> 1.0,
  synthseg_noem     data_res -> 1.0, atlas_res -> 1.0, thickness -> None
                    (fixed 1mm acquisition == atlas res => no blur/downsample;
                    GMM/EM/label/bias/gamma settings preserved verbatim)
  auglab_default  : GaussianBlurTransform.probability -> 0.0,
                    SimulateLowResTransform.probability -> 0.0
                    (all other intensity augs unchanged)
"""
import json
from pathlib import Path

CFG_IN  = Path(__file__).resolve().parents[1] / "configs"
CFG_OUT = Path(__file__).resolve().parents[1] / "configs_noblur"
CFG_OUT.mkdir(parents=True, exist_ok=True)


def load(name):
    with open(CFG_IN / name) as f:
        return json.load(f)


def write(cfg, name):
    with open(CFG_OUT / name, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"  wrote configs_noblur/{name}")


def build_palette():
    cfg = load("palette.json")
    cfg["ImageContrastV26_6_2GPUTransform"]["blur_sigmas"] = [0.0]
    assert cfg["ImageContrastV26_6_2GPUTransform"]["probability"] == 1.0
    write(cfg, "palette.json")


def build_synthseg(src, tag):
    cfg = load(src)
    s = cfg["SynthSeg"]
    assert s["probability"] == 1.0
    em_before = s["em_label_completion"]                 # preserve EM identity
    s["randomise_res"] = False
    s["blur_range"] = 1.0
    s["data_res"] = 1.0
    s["atlas_res"] = 1.0
    s["thickness"] = None
    assert s["em_label_completion"] == em_before, "EM setting must be preserved"
    write(cfg, src)


def build_auglab_default():
    cfg = load("auglab_default.json")
    cfg["GaussianBlurTransform"]["probability"] = 0.0
    cfg["SimulateLowResTransform"]["probability"] = 0.0
    # sanity: spatial still off (inherited from main config)
    assert cfg["FlipTransform"]["probability"] == 0.0
    assert cfg["AffineTransform"]["probability"] == 0.0
    write(cfg, "auglab_default.json")


def main():
    print(f"in : {CFG_IN}\nout: {CFG_OUT}")
    build_palette()
    build_synthseg("synthseg_em.json", "synthseg_em")
    build_synthseg("synthseg_noem.json", "synthseg_noem")
    build_auglab_default()

    # Report the exact blur/res knobs now in effect.
    print("\nNo-blur config verification:")
    p = load("palette.json"); _ = p  # (in-dir originals unchanged; re-read outputs)
    import json as _j
    for n in ["palette", "synthseg_em", "synthseg_noem", "auglab_default"]:
        d = _j.load(open(CFG_OUT / f"{n}.json"))
        if n == "palette":
            print(f"  palette blur_sigmas={d['ImageContrastV26_6_2GPUTransform']['blur_sigmas']}")
        elif n == "auglab_default":
            print(f"  auglab GaussianBlur.p={d['GaussianBlurTransform']['probability']} "
                  f"SimulateLowRes.p={d['SimulateLowResTransform']['probability']}")
        else:
            s = d["SynthSeg"]
            print(f"  {n} randomise_res={s['randomise_res']} blur_range={s['blur_range']} "
                  f"data_res={s['data_res']} atlas_res={s['atlas_res']} thickness={s['thickness']} "
                  f"(em_label_completion={s['em_label_completion']})")
    print("Done.")


if __name__ == "__main__":
    main()
