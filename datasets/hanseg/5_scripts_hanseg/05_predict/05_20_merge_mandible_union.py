#!/usr/bin/env python3
"""
Collapse a toothfairy2 3-class prediction into HaN-Seg's single mandible class.

WHY THIS EXISTS
---------------
The two label spaces do not line up one-to-one, and the mismatch is real anatomy,
not a numbering accident. HaN-Seg delineates ONE structure, `Bone_Mandible`, which
INCLUDES the lower dentition. toothfairy2 splits that same anatomy into two classes,
`mandible` (id 1, the jawbone) and `lower_teeth` (id 2). So the correct comparison is

    HaN-Seg Bone_Mandible   vs   (predicted mandible ∪ predicted lower_teeth)

and the union is EXACT — the two toothfairy2 classes tile HaN-Seg's one structure
with nothing left over. (This exactness is why toothfairy2's task was reduced to the
mandibular block in the first place; see toothfairy2_labels.MANDIBLE_UNION_*.)

Evaluating `mandible` alone against HaN-Seg's GT would score every lower tooth as a
false negative and report a large, entirely artefactual deficit — the annotation-
style mismatch this project has repeatedly been bitten by. The shared evaluator's
--label_map handles a one-to-one id remap but not a UNION, hence this pass.

`pharynx` (id 3) has no counterpart in HaN-Seg's mandible mask and becomes background.

Reads a prediction dir, writes a sibling dir with binary {0,1} masks.
Idempotent: an existing up-to-date output file is left alone.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DATASET_ROOT.parent / "toothfairy2" / "5_scripts_toothfairy2" / "00_utils"))
from toothfairy2_labels import MANDIBLE_UNION_TARGET_IDS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(a.pred_dir.glob("*.nii.gz"))
    if not files:
        raise SystemExit(f"no predictions in {a.pred_dir}")
    n = 0
    for f in files:
        out = a.out_dir / f.name
        if out.exists() and out.stat().st_mtime >= f.stat().st_mtime:
            continue
        img = nib.load(str(f))
        arr = np.asanyarray(img.dataobj)
        union = np.isin(arr, MANDIBLE_UNION_TARGET_IDS).astype(np.uint8)
        nib.save(nib.Nifti1Image(union, img.affine, img.header), str(out))
        n += 1
    print(f"merged {n}/{len(files)} -> {a.out_dir} "
          f"(union of target ids {MANDIBLE_UNION_TARGET_IDS})")


if __name__ == "__main__":
    main()
