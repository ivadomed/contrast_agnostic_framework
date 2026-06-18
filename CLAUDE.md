# Project: MRI Synthesis — Claude Context

## Cluster resource management (`set_slot`)

**Every GPU- or RAM-heavy command MUST use `set_slot`.**
The login node has ~5% of total RAM and no GPU. Running compute directly on the login node will OOM or be killed.

```bash
# Single job
set_slot 0 .venv/bin/python script.py ...

# Parallel jobs
set_slot 0 cmd_a > /tmp/a.log 2>&1 & P0=$!
set_slot 1 cmd_b > /tmp/b.log 2>&1 & P1=$!
set_slot 2 cmd_c > /tmp/c.log 2>&1 & P2=$!
wait $P0 $P1 $P2
```

Hardware: 4 × NVIDIA RTX A6000 (48 GB each), 64 CPUs.
Slots: `set_slot 0-3` → 256 CPU workers + 1 GPU each.
Note: you can use `set_slot 0-3` to leverage all these hardware ressources for heavy tasks and tasks we want to process quickly.
Note: there is no set_slot 4...
---

## Project scope

This is an **MRI synthesis project** using domain randomization. The synthesizer is trained to generate synthetic data from T1w data, the goal is **domain randomization across all contrasts** (T1w, T2w, FLAIR, GRE, bold, dwi, epi, …) — there is **no single "target contrast"**. Synthetic images are intentional augmentations that explore far beyond any specific real scanner/contrast cluster.

The analysis pipeline (`contrast_manifold`) measures how synthetic images relate to real multi-scanner, multi-contrast MRI datasets via feature extraction → normalization → manifold analysis (PCA, UMAP, PRDC, Vendi).

---

## Python environment

```bash
.venv/bin/python   # always use this, not system python
```

---

## Dataset structure

All datasets live under `datasets/`. We work in a dataset-centric manner. Every dataset follows the same 9-subdir standard:
```
datasets/
  validate_standard_dataset_structure.py   # run to check compliance
  <dataset>/
    0_raw_<dataset>/     # raw data as downloaded (DICOM, non-BIDS NIfTI, etc.)
    1_BIDS_<dataset>/    # BIDSified data (usually derived from 0_raw — both can coexist)
    2_nnUNet_<dataset>/raw/ + preprocessed/     # nnUNet converted data
    3_conf_<dataset>/data.yaml                  # Hydra data config
    4_splits_<dataset>/                         # train/val/test splits
    5_scripts_<dataset>/                        # pipeline scripts (see below)
    6_checkpoints_<dataset>/                    # model weights
    7_analysis_<dataset>/                       # manifold analysis (on-harmony only)
    8_results_<dataset>/                        # evaluation outputs
      01_results/   02_nnUNet_results/   03_aggregated_results/
    9_tests_<dataset>/                          # dataset-specific tests
```

Scripts inside `5_scripts_*/` follow a strict numbered convention:
```
00_utils/           # shared helpers; env.sh sets all paths (source this first)
01_create_splits/   01_NN_name.py/.sh
02_nnunet/
03_preprocess/
04_train/           04_00_common.sh = shared bash functions
05_predict/
06_evaluate/
```

### Manifold analysis paths (on-harmony)
```
datasets/on-harmony/7_analysis_on-harmony/contrast_manifold/
  config/
  scripts/
    run_all_analysis.py       # orchestrator
    plot_umap_joint.py        # PCA + UMAP
    plot_prdc.py              # PRDC + Vendi
    extract_features_*.py
    normalize_combined.py
  outputs/
    data/
      original/<mask_type>/
      synthetic_<version>/<mask_type>/
    plots/v<major>/<version>_r<run>/<mask_type>/pca/ prdc/ umap/
```
---

#### Feature types (mask types)

| Name | Dims | Description |
|---|---|---|
| `regional_hist_64` | 448 | 7 brain regions × 64-bin intensity histogram |
| `regional_hist_13_64` | 832 | 13 brain regions × 64-bin intensity histogram |
| `histogram_256` | 256 | global intensity histogram |
| `hog_972` | 972 | HOG features |
| `hog3d_512` | 512 | 3D HOG |
| `curia_embeddings` | varies | CURIA neural embeddings |

`regional_hist_64` is the primary feature space for cross-version comparison.
`regional_hist_13_64` adds resolution for T1w groups but makes GRE/other groups even harder to penetrate.

---

#### PRDC + Vendi (plot_prdc.py)

Key constants (do not change without good reason):
```python
THRESHOLD_PCT = 95    # P95 of real NN dists → IND/OOD boundary
MAX_K = 3             # max nearest_k for compute_prdc
MIN_REAL = 4          # min real samples to attempt PRDC
MIN_IND  = 5          # min IND synth (prdc uses kth=k+1 internally → needs k+2)
MAX_OOD_VENDI = 150   # eigvalsh is O(n³); spikes at n≈250 on this machine's OpenBLAS
```

**Known finding**: 99%+ of synthetic samples are OOD for all modality × scanner groups.
Only `GRE × Siemens Trio` (~25–33% IND) and a handful of T1w groups have enough IND samples for PRDC. This is by design — domain randomization intentionally explores far beyond real cluster boundaries (~5× median NN distance).

**Vendi API**: `from vendi_score import vendi; vendi.score_X(X)` — no `model` argument.

---

#### HTML interactive plots (plot_umap_joint.py)

- Every 3D PCA/UMAP plot (`*_3d.html`) has a companion 2D lasso plot (`*_2d.html`).
- The 2D plot supports: lasso selection → remove points, click-to-copy metadata, reset.
- Safari clipboard fix: textarea must be `position:absolute;left:-9999px` (off-screen), NOT `opacity:0` (Safari blocks `execCommand('copy')` on invisible elements).
- `go.Scatter3d` does NOT support lasso2d — lasso only works on the 2D companion.

---

## Common gotchas

- `set_slot` is a real binary (`/usr/local/bin/set_slot` → `sudo ml_job` → `systemd-run --slice=ml-1slot-N.slice`). It is available in all subshells and works from non-interactive contexts including Claude Code's Bash tool. `sudo` is passwordless for `ml_job`.
- Whenever we run anything, we want to run a script with a simple bash command. The likely already exists in `5_scripts_*/` and if not, it should be added there, the structure should naturally guide you. We want to avoid running Python scripts directly from the command line without a proper script wrapper. Running existing scripts is good for consistency, and creating new scripts in the right place is good for organization and future reproducibility, it also ensure we don't try to debug the same thing many times. 
