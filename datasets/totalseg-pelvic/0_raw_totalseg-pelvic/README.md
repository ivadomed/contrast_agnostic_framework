# totalseg-pelvic — onboarding notes

Pelvic/hip musculoskeletal segmentation built from TotalSegmentator's public CT and MRI
releases (University Hospital Basel). Onboarded 2026-09-15 as this project's first
musculoskeletal (non-organ) training task, and first task built from two **unpaired,
already-fully-labeled public releases** (no DICOM/BIDS conversion, no DUA/registration
gate) rather than a single challenge archive.

## Source data

| Release | Zenodo record | DOI | License |
|---|---|---|---|
| TotalSegmentator CT  | 10047292 | 10.5281/zenodo.10047292 | **CC BY 4.0** |
| TotalSegmentator MRI | 11367005 | 10.5281/zenodo.11367005 | **CC BY-NC-SA 2.0** (Generic — NOT 4.0) |

Both verified two independent ways: the record's own page prose, AND the Zenodo API's
`metadata.license.id` field (`"cc-by-4.0"` / `"cc-by-nc-sa-2.0"` respectively). Neither
archive bundles a separate LICENSE file inside the zip itself, so the pre-flight
checklist's "3rd way" doesn't apply here. The MRI release's NC restriction is the same
situation as duke-breast-mri elsewhere in this project's roster: fine for this project's
research/paper use, **not** eligible for the CC-BY-only git-annex public upload list if
that's ever revisited — tag it separately, don't blend it into one combined license
statement for this dataset.

## Claimed-N vs counted-N

Both counted directly from each archive's own zip central directory (not trusted from
page prose):

- CT: 1228 claimed, **1228 counted** (subject ids s0000–s1429, sparse — the organizers'
  own internal training pool is larger; 1228 is the public release).
- MRI: 298 claimed, **298 counted** (subject ids s0001–s0298, contiguous). Cross-checked
  against the bundled `meta.csv`: 251 University-Hospital-Basel + 47 IDC-origin = 298,
  matching the source paper's reported 251/47 split exactly.

## Label subset — pelvic/hip musculoskeletal (10 classes)

Chosen from the 42 class names the two releases share, restricted to the ones verified at
**100% coverage in both releases** (298/298 MRI, 1228/1228 CT) — every one of those 42
shared classes happens to be at 100% coverage, so this was a scope choice (genuinely new
tissue class, no overlap with chaos/amos's abdominal organs, no overlap with anything
else in the roster), not a coverage-driven necessity:

```
hip_left, hip_right, sacrum,
gluteus_maximus_left, gluteus_maximus_right,
gluteus_medius_left, gluteus_medius_right,
gluteus_minimus_left, gluteus_minimus_right,
iliopsoas_left, iliopsoas_right
```

Integer label ids fixed in `02_nnunet/02_00_convert.py`'s `LABEL_MAP` (1–11, background=0)
— do not reorder once training has started.

Abdominal organs and vascular structures (also present at 100% coverage in both releases)
were deliberately ruled out — the former for redundancy with chaos/amos, the latter in
favor of a genuinely new tissue class for this project.

## Label semantics (from source, not inferred)

Both releases' ground truth was built via the same disclosed, semi-automated iterative
process: preliminary model → auto-segment → manual correction → retrain, reviewed by a
board-certified radiologist (12 yrs experience for the MRI release). Sources:
- CT: Wasserthal et al., *Radiology: Artificial Intelligence* 2023 (v1, 104 classes); the
  public v2 release (117 classes, what this project uses) is documented in the project's
  own GitHub changelog (`improvements_in_v2.md`), which also discloses real known
  limitations (rib-near-spine and colon/small-bowel segmentation are acknowledged as
  imperfect — neither affects the pelvic/hip subset used here).
- MRI: D'Antonoli et al., *Radiology* 2025.

No "malignant-only"-style mislabeling risk found on inspection — this is disclosed
methodology, not an inherited docstring claim.

## MRI is sequence-heterogeneous, not one named contrast

The bundled `meta.csv` shows a real mix of SE/GR/IR sequences across scanners and
institutes (251 University Hospital Basel + 47 IDC-origin cases). The source paper's own
framing is **"sequence-independent segmentation"** by design — this is a deliberate
property of the release, not a data-quality problem, and arguably a nice conceptual fit
with this project's own domain-randomization thesis. Downstream docs/configs should
describe MRI this way, not imply it's a single T1w/T2w-style contrast the way most of this
project's other datasets are.

MRI also carries real anisotropic thick-slice spacing in some cases (e.g. 4.4mm slice
thickness observed in spot checks) — a genuine clinical-routine characteristic, handled by
normal nnU-Net resampling during preprocessing, not a special case.

## Orientation check

`nib.aff2axcodes` verified on 5 samples spread across both modalities and both MRI source
sites (University Hospital Basel PACS + IDC) during pre-flight — all RAS, every
image/label pair matched in shape and spacing. Re-verify on the real converted dataset
once `02_00_convert.py` has run (spot-check, not exhaustive — see project feedback memory
on this point).

## CT subsampling (scope decision, not silent)

The public CT release (1228 cases) is far larger than any other dataset in this project's
roster. `01_create_splits/01_01_create_splits.py` subsamples CT down to `CT_SAMPLE_N`
(currently 200, random seed 42) before building folds, to keep preprocessing/training time
comparable to the rest of the roster. MRI (298 cases) is used in full. **Flagged for the
coordinating session/user to override** (raise or remove `CT_SAMPLE_N`) if the full 1228
should be used instead — this was not a unilateral silent default.

## Epoch policy

200 epochs — no dataset-specific precedent exists yet for TotalSegmentator; chosen as the
closest structural analog to chaos (unpaired CT+MRI, shared organ/tissue label set), not
autopet's 2000 (very different scale problem) or toothfairy2's 1000. Revisit if a sizing
probe on TamIA suggests otherwise.

## Pipeline shape

CT and MRI are **unpaired** (entirely different patient cohorts) — unlike this project's
other two-training-contrast datasets (chaos t1in/t2spir, autopet ct/pet, ispy2
t1wce/t2w), which are the same patients' different sequences/channels. This means:
- Each modality gets its own independent split under
  `4_splits_totalseg-pelvic/{ct,mri}/` (see `00_utils/env.sh`'s `SPLITS_DIR` override).
- `02_nnunet/02_00_convert.py` reads directly from the two Zenodo zips (no full unzip to
  disk — avoids the "many small files hammer networked filesystems" trap for the
  un-subsampled classes) and merges each case's 10 binary structure masks into one
  multi-class label map.
- Cross-contrast evaluation means predicting a CT-trained model on the MRI test set (and
  vice versa) — there is no "same case, other channel" relationship the way autopet has.

Standard 6-method suite (baseline, auglab_default, synthseg_noEM, synthseg_EM, srcsm,
auglabAug_v26_6_2/OURS via DualVal) × 3 folds (0/1/2, permanent project policy) × 2
training modalities, **plus** the 4 causal-ablation-ladder training configs
(baseline_kmeans, +label_remap, +label_remap_voronoi, v26_6_2_train050_val100 standalone)
× 3 folds × 2 modalities — trained in parallel with the headline suite, following
autopet's precedent (`04_16`–`04_23` there) rather than as a later follow-up phase.
**Total: 10 distinct training configs × 3 folds × 2 modalities = 60 training jobs.**
