# MRI Synthesis Project — Agent Handover

This document is the single source of truth for a coding agent picking up this project.
It covers cluster usage, the synthesis pipeline, the manifold analysis pipeline, naming conventions, and common gotchas.

---

## 1. Cluster resource management (`set_slot`)

**Every GPU- or RAM-heavy command MUST use `set_slot`.**
The login node has ~5% of total RAM and no GPU. Running compute directly on the login node will OOM or be killed.

```bash
# Single job
set_slot 0 .venv/bin/python script.py ...

# Parallel jobs on all 4 GPUs
set_slot 0 cmd_a > /tmp/a.log 2>&1 & P0=$!
set_slot 1 cmd_b > /tmp/b.log 2>&1 & P1=$!
set_slot 2 cmd_c > /tmp/c.log 2>&1 & P2=$!
set_slot 3 cmd_d > /tmp/d.log 2>&1 & P3=$!
wait $P0 $P1 $P2 $P3
```

- Hardware: 4 × NVIDIA RTX A6000 (48 GB each), 64 CPUs
- Slots: `set_slot 0–3` → 256 CPU workers + 4 GPU (~250 GB)
- **`set_slot` is a real binary** (`/usr/local/bin/set_slot` → `sudo ml_job` → `systemd-run --slice=ml-1slot-N.slice`) — it is available in all subshells and works from non-interactive contexts. `sudo` is passwordless for `ml_job`.
- **Claude Code (VS Code extension) runs inside slot 0 automatically** via a PreToolUse hook (`.claude/hooks/slot_wrapper.py` + `.claude/settings.json`). Every Bash command Claude issues is transparently routed through `~/.local/bin/claude_slot`, which wraps it in `sudo ml_job 0` and strips the cowsay header. Commands that already contain `set_slot`/`ml_job`/`claude_slot` are passed through unchanged. Use `set_slot 1`/`2`/`3` explicitly when you need a different slot.
- Multiple jobs can share one slot (they will run in parallel)
- There is no `set_slot 4` on romane
- Several processes can run at the same time on a slot
- Running a command outside of the set_slot function could create many problems
- Note that using `set_slot 0` (for example), will open a shell inside the slot 0.

### Monitoring slot usage

```bash
systemctl status "ml-1slot-0.slice" "ml-1slot-1.slice" "ml-1slot-2.slice" "ml-1slot-3.slice" 2>/dev/null | grep -E "Tasks:|Memory:|CGroup:|└─[0-9]"
```

Stale processes from old sessions can hold slots indefinitely at 0% CPU. If slots appear full but nothing is running, kill the PIDs listed under the CGroup.

---

## 2. Python environment

```bash
.venv/bin/python   # always use this, never system python
```

---

## 3. Project scope

### Downstream goal
The ultimate goal is a **contrast-robust segmentation model** that works across any MRI scanner, field strength, and acquisition protocol without retraining. The approach is domain randomization: train the segmenter on synthetic images that span the full space of plausible MRI contrasts, so the model never overfits to a specific acquisition.

### What this codebase focuses on
This project focuses on **the synthesis side and its manifold analysis** — not segmentation training. The question we are answering here is: *do our synthetic images actually cover the contrast space we care about?*

We have two complementary coverage goals:
1. **Cover all real contrast types** — synthetic images should land near (or within) the real scanner/modality clusters so a model trained on them will generalize to real data.
2. **Cover the "empty space" with sufficient density** — the space between and beyond real clusters should also be explored so the model is robust to unseen acquisition protocols, artifacts, or pathology-induced contrast shifts. 99%+ of synthetic samples are intentionally OOD (outside real clusters); this is a feature, not a bug.

### Two synthesis approaches
There are two complementary ways to create synthetic data in this project:

1. **Guidance map generation** (current focus): the target generator runs directly on T1w input to produce a guidance map — a 3D intensity remapping that encodes the desired contrast. These maps can be saved as NIfTI files and visualized independently. This is what `scripts/generate_synthetic_guidance.py` does, and what all the `v23_*` / `v25_*` versions produce.

2. **Full MRI synthesis via U-Net**: a U-Net is trained to take a T1w scan + guidance map as input and output a photorealistic synthetic MRI. The guidance map is used as a conditioning signal. The trained model can then synthesize new images for downstream segmentation training.

Right now all analysis is done on the guidance maps themselves — they capture the contrast structure (intensity distribution per tissue class) without the added complexity of the U-Net. This is sufficient for manifold analysis because the guidance map directly determines what contrast the synthesizer will produce.

**Critical constraint**: the synthesis framework must NEVER use labels, atlases, or anatomical templates. It must work for pathological brains where such priors fail. Any approach requiring SynthSeg output, label maps, or registered templates is rejected.

The analysis pipeline (`contrast_manifold`) measures how synthetic guidance maps relate to real multi-scanner, multi-contrast MRI datasets via: feature extraction → normalization → manifold analysis (PCA, UMAP, PRDC, Vendi).

---

## 4. Synthesis pipeline

### 4.1 Generating guidance maps

```bash
set_slot 0 .venv/bin/python scripts/generate_synthetic_guidance.py \
    --generator v23_4 \
    --lhc \
    --n-variants 10 \
    --world-size 2 --rank 0 \
    > /tmp/gen_rank0.log 2>&1 &

set_slot 1 .venv/bin/python scripts/generate_synthetic_guidance.py \
    --generator v23_4 \
    --lhc \
    --n-variants 10 \
    --world-size 2 --rank 1 \
    > /tmp/gen_rank1.log 2>&1 &
```

Key arguments:
- `--generator`: version name (see §5 for list). Determines which target generator class and checkpoint to load.
- `--lhc`: use Sobol quasi-random (mu, alpha) parameter sampling. Always use this for production runs.
- `--n-variants`: number of synthetic variants per subject (default 10 → 165 subjects × 10 = 1650 files)
- `--world-size` / `--rank`: split subjects across N processes for parallelism
- `--output-dir`: default is `data/ON-Harmony/derivatives/synthetic_<generator>_guidance[_lhc]`
- `--limit N`: generate only N subjects (useful for quick visual checks before full run)

Output lands at:
```
data/ON-Harmony/derivatives/synthetic_<version>_guidance_lhc/
  sub-<id>/ses-<id>/
    sub-<id>_ses-<id>_run-00_syn-T1w.nii.gz
    sub-<id>_ses-<id>_run-01_syn-T1w.nii.gz
    ...
```

The "guidance" and "lhc" suffixes describe **how** the generator is used, not what the version is — keep them.

### 4.2 Checkpoints

Checkpoints live at:
```
checkpoints/on_harmony/generator/<version>/t1w/
```

Most recent versions borrow the `v23_3` checkpoint. The mapping is in `_ckpt_base()` in [scripts/generate_synthetic_guidance.py](scripts/generate_synthetic_guidance.py).

### 4.3 Generator configs

Each version has a Hydra config in `conf/generator/<version>.yaml`. When adding a new version, create the yaml and wire it into `_BLUR_SIGMA_RANGES`, `_ZOOM_RANGES`, `_NATIVE_BLUR`, and `_ckpt_base()` in `generate_synthetic_guidance.py`, plus add to the `--generator` choices list and the analysis REGISTRY.

---

## 5. Synthesis versions

Naming: `v<major>_<minor>`. Major increments on architectural changes, minor on hyperparameter/sampling variants. Full details in [docs/synthesis_versions.md](docs/synthesis_versions.md).

### v19_c — Baseline
**What it does**: splits the brain foreground into K=8 equal quantile chunks; each chunk gets an independent affine remap (mu, alpha). The remapped intensity map (guidance map) drives the U-Net synthesizer.  
**What's special**: establishes the core mechanism. Fixed K=8 means contrast diversity is limited — the generator always produces 8-region remappings, which are fairly T1w-like in structure.

### v22_1, v22_2 — LHC sampling
**What it does**: same architecture as v19. Replaces grid-sampled (mu, alpha) parameters with Sobol quasi-random (Latin Hypercube) sampling for better coverage of the parameter space.  
**What's special**: `v22_1` achieves the best recall and manifold alignment with real data seen in this project. `v22_2` maximizes Vendi diversity (more spread-out outputs) at the cost of lower recall. These two represent the recall/diversity Pareto frontier for the K=8 architecture.

### v23_1 — Random K
**What it does**: same as v22 but K is drawn randomly from {2, 3, 4, 6, 8, 12, 16} on every forward pass.  
**What's special**: with K=2, the probability of T2w-like contrast (all WM chunks low, all GM chunks high) rises to ~16% — the first version capable of producing biologically plausible contrast inversions without any explicit T2w target. LHC sampling still used for (mu, alpha).

### v23_2 — Foreground-quantile chunk boundaries
**What it does**: chunk edges are placed at foreground-intensity quantile percentiles instead of uniform linspace.  
**What's special**: ensures each chunk covers equal brain-tissue mass rather than equal intensity range. With small K this better separates CSF/GM/WM since their populations are unequal in intensity space. Less impactful at large K.

### v23_3 — Random blur + random resolution
**What it does**: V23RandomChunk base (same as v23_1) plus per-variant random Gaussian blur σ ~ U(0.3, 3.0) and random resolution downsampling zoom ~ U(0.4, 1.0) applied to the guidance map before synthesis.  
**What's special**: adds diversity orthogonal to the chunk parameterization — the same (K, mu, alpha) tuple produces different guidance maps depending on the blur/zoom draw. Blur ~ U(0.3, 3.0) uniform means heavily-blurred samples are as common as sharp ones, which was found to push the distribution away from tight T1w clusters (T1w IND counts drop vs v23_1) while improving GRE × Siemens Trio recall (0.22 → 0.41). Vendi also improves. The over-blurring is the main weakness addressed by v23_4.

### v23_4 — Log-skewed blur
**What it does**: identical to v23_3 except the blur sigma distribution is changed to: 30% probability of no blur (sigma=0), 70% log-uniform over [0.3, 3.0].  
**What's special**: the log-uniform distribution concentrates mass at low sigma (~50% of blurred samples below σ=0.95) while still allowing rare heavy blur. Motivation: v23_3 produced too many over-blurred guidance maps that were visually unnatural and pushed too far from all real clusters. v23_4 preserves the diversity benefit of blur while keeping most outputs sharp. Uses the v23_3 checkpoint (no retraining needed).

### v25_1 — Ellipsoidal blob spatial modulator
**What it does**: generates a base V23RandomChunk guidance map for the whole volume, then stamps 0–3 random ellipsoidal blobs on top. Each blob has a random center sampled from the brain foreground, anisotropic radii drawn from U(15, 80) voxels per axis, a hard binary boundary, and its own independent V23RandomChunk guidance map call (different K, mu, alpha from the base). Inherits blur and zoom from v23_3 settings.  
**What's special**: introduces localized spatial contrast variation without any labels or atlas. The blobs create hard-boundary regions with distinct tissue remapping — simulating the kind of focal contrast differences seen in pathology or partial-volume effects. Key PRDC finding vs v23_3: GRE × Siemens Prisma 64ch IND jumps from 5 to 24 (precision 0 → 0.83), suggesting the blobs help cover part of GRE feature space. However, Vendi drops from 8.35 to 6.85 — the base+blob structure makes variants more correlated with each other than pure v23_3.

---

### Design constraints that apply to all versions
- **No labels, no atlas, no anatomical templates.** Must work on pathological brains where such priors fail. Any approach that requires SynthSeg output, label maps, or a registered template is rejected.
- **No targeted contrast generation.** We do not explicitly optimize for T2w similarity or any specific modality. All contrast diversity emerges from randomization of intensity remapping parameters.

Legacy classes (V18*, V20) are in `src/target_generators_legacy.py`, re-exported from `src/target_generators.py` for backward compatibility with old configs.

---

## 6. Manifold analysis pipeline

### 6.1 Full pipeline (step by step)

For a new version `vXX_Y` with LHC sampling, run these steps in order:

#### Step 1 — Extract features

```bash
set_slot 0 .venv/bin/python datasets/on-harmony/7_analysis_on-harmony/contrast_manifold/scripts/extract_features_regional_hist.py \
    --mode synthetic \
    --synth-root data/ON-Harmony/derivatives/synthetic_vXX_Y_guidance_lhc \
    --output-csv analysis/contrast_manifold/outputs/data/synthetic_vXX_Y_guidance_lhc/regional_hist_64/synthetic_vXX_Y_guidance_lhc_features.csv \
    --n-workers 56
```

#### Step 2 — Normalize + feature selection

```bash
set_slot 0 .venv/bin/python datasets/on-harmony/7_analysis_on-harmony/contrast_manifold/scripts/normalize_combined.py \
    --original_csv  analysis/contrast_manifold/outputs/data/original/regional_hist_64/on_harmony_features.csv \
    --synthetic_csv analysis/contrast_manifold/outputs/data/synthetic_vXX_Y_guidance_lhc/regional_hist_64/synthetic_vXX_Y_guidance_lhc_features.csv \
    --output_original  analysis/contrast_manifold/outputs/data/synthetic_vXX_Y_guidance_lhc/regional_hist_64/on_harmony_features_normalized_combined_downsampled100.csv \
    --output_synthetic analysis/contrast_manifold/outputs/data/synthetic_vXX_Y_guidance_lhc/regional_hist_64/synthetic_vXX_Y_guidance_lhc_features_normalized_combined.csv \
    --feature_config   analysis/contrast_manifold/config/feature_selection.yaml
```

**Critical**: pass paths WITHOUT `_feat_selected` suffix. When `--feature_config` is provided, the script automatically appends `_feat_selected` to the output filenames. Passing paths that already contain `_feat_selected` will produce `_feat_selected_feat_selected` double-suffixed files that won't be found by the analysis registry.

#### Step 3 — Run analysis

```bash
set_slot 0 .venv/bin/python datasets/on-harmony/7_analysis_on-harmony/contrast_manifold/scripts/run_all_analysis.py \
    --mask-type regional_hist_64 \
    --only vXX_Y_guidance_lhc_r1
```

The `--only` value must match a `run` name in the REGISTRY inside `run_all_analysis.py`. The `_r1` suffix is the run index (increment if re-running with different settings).

### 6.2 Adding a new version to the REGISTRY

In `analysis/contrast_manifold/scripts/run_all_analysis.py`, add a tuple to the `REGISTRY` list:

```python
(
    "vXX",                          # major version (used as plot subdirectory)
    "vXX_Y_guidance_lhc_r1",        # run name (must be unique)
    {                               # normalized synthetic CSV paths, by mask_type
        "regional_hist_64": DATA_ROOT / "synthetic_vXX_Y_guidance_lhc" / "regional_hist_64"
                    / "synthetic_vXX_Y_guidance_lhc_features_normalized_combined_feat_selected.csv",
    },
    {                               # raw synthetic feature CSV paths
        "regional_hist_64": DATA_ROOT / "synthetic_vXX_Y_guidance_lhc" / "regional_hist_64"
                    / "synthetic_vXX_Y_guidance_lhc_features.csv",
    },
    {                               # normalized real CSV paths (per version output dir)
        "regional_hist_64": DATA_ROOT / "synthetic_vXX_Y_guidance_lhc" / "regional_hist_64"
                    / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
    },
),
```

### 6.3 Running only specific analysis steps

`run_all_analysis.py` has no `--steps` argument. To re-run only specific plots:

```bash
ANALYSIS=datasets/on-harmony/7_analysis_on-harmony/contrast_manifold

# PCA only (skip UMAP)
.venv/bin/python $ANALYSIS/scripts/plot_umap_joint.py \
    --synthetic_csv <synth_norm_csv> \
    --original_csv  <orig_norm_csv> \
    --output_dir    $ANALYSIS/outputs/plots/vXX/vXX_Y_r1/regional_hist_64/pca \
    --plot_pca --plot_loadings --skip_umap

# PRDC only, with custom PCA variance
.venv/bin/python $ANALYSIS/scripts/plot_prdc.py \
    --original_csv  <orig_norm_csv> \
    --synthetic_csv <synth_norm_csv> \
    --output_dir    $ANALYSIS/outputs/plots/vXX/vXX_Y_r1/regional_hist_64/prdc \
    --pca-variance  0.90
```

### 6.4 Parallel analysis across all versions

```bash
for rank in 0 1 2 3; do
    set_slot $rank .venv/bin/python analysis/contrast_manifold/scripts/run_all_analysis.py \
        --mask-type regional_hist_64 \
        --rank $rank --world-size 4 \
        > /tmp/analysis_rank${rank}.log 2>&1 < /dev/null &
done
```

---

## 7. Path conventions

### Dataset root structure

All datasets live under `datasets/`. Every dataset uses the same 9-subdir standard:
```
datasets/
  validate_standard_dataset_structure.py  # run to verify compliance
  on-harmony/
    1_BIDS_on-harmony/                    # BIDS data (real scans + derivatives)
      derivatives/synthetic_<v>_guidance_lhc/sub-*/ses-*/*.nii.gz
    2_nnUNet_on-harmony/raw/ + preprocessed/
    3_conf_on-harmony/data.yaml
    4_splits_on-harmony/on_harmony_split.json
    5_scripts_on-harmony/                 # numbered pipeline scripts
      00_utils/env.sh                     # source this — sets nnUNet_raw, nnUNet_preprocessed, etc.
      01_create_splits/ … 06_evaluate/
    6_checkpoints_on-harmony/
    7_analysis_on-harmony/contrast_manifold/
    8_results_on-harmony/01_results/ + 02_nnUNet_results/ + 03_aggregated_results/
    9_tests_on-harmony/
  brats/         # same structure, 0_raw_brats/ (not BIDS → slot 0)
  spider-spine/  # same structure, 0_raw_spider-spine/ (not BIDS → slot 0)
  ms-multi-spine/# same structure, 1_BIDS_ms-multi-spine/ → symlink to data/ms_multi_spine
```

### Feature data (manifold analysis)
```
datasets/on-harmony/7_analysis_on-harmony/contrast_manifold/outputs/data/
  original/<mask_type>/
    on_harmony_features.csv
    on_harmony_features_normalized_combined_downsampled100_feat_selected.csv
  synthetic_<version>/<mask_type>/
    synthetic_<version>_features.csv
    synthetic_<version>_features_normalized_combined_feat_selected.csv
    on_harmony_features_normalized_combined_downsampled100_feat_selected.csv
```

### Plots / results (manifold analysis)
```
datasets/on-harmony/7_analysis_on-harmony/contrast_manifold/outputs/plots/
  v<major>/<version>_r<run>/<mask_type>/
    pca/   umap/   prdc/   prdc_pca60/   coverage/
```

### Configs
```
conf/
  config.yaml                  # top-level Hydra (searchpath includes all 3_conf_* dirs)
  generator/<version>.yaml     # per-version synthesis config
  training/defaults.yaml
  logging/wandb.yaml
  model/   segmenter/

datasets/<ds>/3_conf_<ds>/data.yaml   # dataset-specific Hydra config (auto-discovered)
```

---

## 8. Feature types (mask types)

Each feature type gives a different view of the intensity distribution of an MRI volume. They are extracted by the `extract_features_*.py` scripts and used as input to all manifold analysis.

| Name | Dims | What it captures | Strengths / Weaknesses |
|---|---|---|---|
| `regional_hist_64` | 448 | 7 brain regions × 64-bin intensity histogram | **Primary space.** Directly measures the per-tissue-class intensity distribution — exactly what the guidance map controls. Compact, fast, interpretable. Insensitive to spatial texture or fine structure. |
| `regional_hist_13_64` | 832 | 13 brain regions × 64-bin intensity histogram | More anatomical detail than regional_hist_64, helpful for sub-group T1w comparisons. But the extra dims add noise for GRE/bold/dwi groups whose regional distributions are less distinct — makes those groups harder to penetrate and less comparable across versions. |
| `histogram_256` | 256 | Whole-brain global intensity histogram | Captures gross dark/bright balance and number of modes. Loses all regional tissue-contrast information. Useful for sanity checks but insufficient for modality discrimination. |
| `hog_972` | 972 | 2D Histogram of Oriented Gradients | Captures local texture, edge density, and boundary sharpness in 2D slices. Sensitive to blurriness vs sharpness — orthogonal to the contrast remapping that synthesis controls. |
| `hog3d_512` | 512 | 3D HOG | Same as hog_972, computed volumetrically. Heavier to compute, captures 3D texture structure. |
| `curia_embeddings` | varies | Deep neural embeddings (CURIA encoder) | High-dimensional semantic features from a pre-trained MRI encoder. Can capture multi-scale structure but is harder to interpret and depends on the encoder's own training distribution. |

### Why `regional_hist_64` is the primary space

The 7 regions (white matter, gray matter, CSF, brainstem, cerebellum WM, cerebellum GM, subcortical GM) directly correspond to the tissue classes whose contrast the guidance map remaps. A T1w→T2w contrast shift shows up as a clear inversion of the WM and GM histogram peaks — exactly what the synthesis targets. The 7-region breakdown is coarse enough to be robust across scanners and resolutions while fine enough to distinguish modality-specific patterns.

Global histograms conflate tissue contributions. Texture features measure frequency content rather than contrast, making them orthogonal to the synthesis goal. Neural embeddings are harder to interpret and slower. `regional_hist_13_64` adds useful T1w sub-group resolution but at the cost of noisier non-T1w groups.

The feature extractor normalizes each volume by [p1, p99] whole-brain intensity before computing histograms, removing absolute scanner scaling differences.

---

## 9. PRDC + Vendi metrics

### 9.1 Column reference

**`prdc_metrics.csv` columns**: `group, n_real, n_synth_total, real_median_nn_dist, real_p95_nn_dist, n_ind, n_ood, ood_pct, prdc_k, precision, recall, density, coverage, ood_mean_norm_dist, ood_var_norm_dist, vendi_diversity_score`

| Column | What it measures |
|---|---|
| `group` | `<modality> × <scanner>` label, e.g. `GRE × Siemens Trio` |
| `n_real` | Number of real scans in this group |
| `n_synth_total` | Total synthetic samples (same for all groups = n_subjects × n_variants) |
| `real_median_nn_dist` | Median NN distance within the real group — proxy for how tight the real cluster is |
| `real_p95_nn_dist` | P95 NN distance within real group — the IND/OOD boundary threshold |
| `n_ind` | Synthetic samples inside the real cluster (dist to nearest real ≤ P95 threshold) |
| `ood_pct` | % of synthetic that is OOD — nearly always 99–100% by design |
| `prdc_k` | k used for PRDC (auto-selected, `—` if insufficient IND) |
| `precision` | Of IND synth: fraction whose k-NN are all real (synth quality within cluster) |
| `recall` | Of real: fraction covered by at least one IND synth neighbor (real manifold coverage) |
| `density` | Mean number of IND synth in each real's neighborhood / k (local density ratio) |
| `coverage` | Fraction of real samples with ≥1 IND synth within their P95 radius |
| `ood_mean_norm_dist` | Mean distance of OOD synth to nearest real, normalized by the group's P95 threshold — how far outside the cluster the OOD samples sit (higher = more exploratory) |
| `ood_var_norm_dist` | Variance of the above — spread of OOD exploration |
| `vendi_diversity_score` | Diversity of the **full** synthetic distribution (real + synth combined kernel) — higher = more diverse |

### 9.2 How to interpret the results

**The IND/OOD split is the first thing to check.** A group with `n_ind = 0` means not a single synthetic sample landed inside that real cluster — PRDC metrics are meaningless and shown as `—`. This is normal for most groups; the synthesis is doing domain randomization, not targeted generation.

**Groups with enough IND (n_ind ≥ 5) are where PRDC tells you something useful:**
- High **recall** → the synthetic distribution covers the real cluster well (good for training diversity)
- High **precision** → the IND samples that exist are truly close to real (not just barely inside the boundary)
- High **coverage** → the IND samples are spread across the cluster, not concentrated in one corner
- Low **density** → sparse coverage (few synth per real neighborhood)

**`GRE × Siemens Trio`** is the reference group — it has n_real=32 and consistently gets 25–33% IND. Use it for version-to-version comparisons.

**`ood_mean_norm_dist`** is useful even for 100% OOD groups. A value of ~8 means synthetic samples sit ~8× the real cluster radius away. Higher is more exploratory; lower means the generator is drifting toward some real cluster even if not penetrating it. Sudden drops in OOD distance for a specific group can indicate the generator is accidentally biased toward it.

**Vendi diversity score** summarizes the whole synthetic distribution. Higher = more spread-out, more varied outputs. A drop in Vendi (like v25_1 vs v23_3: 6.85 vs 8.35) signals that the new generator introduces correlation or repetition — e.g., the base + blob maps in v25_1 are more similar to each other than pure v23_3 variants.

**Two PCA variance thresholds are always run:**
- `prdc/` at 90% — keeps ~51 PCs for `regional_hist_64`, retains most signal, standard comparison
- `prdc_pca60/` at 60% — keeps fewer PCs, distances are dominated by the strongest real-data axes only; IND counts often increase because the projection is coarser

PCA is **always fit on real data only**, then both real and synthetic are projected into that space.

### 9.3 Constants (do not change without good reason)

```python
THRESHOLD_PCT = 95    # P95 of real NN dists → IND/OOD boundary
MAX_K = 3             # max nearest_k for compute_prdc
MIN_REAL = 4          # min real samples to attempt PRDC
MIN_IND  = 5          # min IND synth (prdc uses kth=k+1 internally → needs k+2)
MAX_OOD_VENDI = 150   # eigvalsh is O(n³); cap OOD sample for Vendi
```

**Vendi API**: `from vendi_score import vendi; vendi.score_X(X)` — no `model` argument.

---

## 10. HTML interactive plots

- Every 3D PCA/UMAP plot (`*_3d.html`) has a companion 2D lasso plot (`*_2d.html`).
- The 2D plot supports: lasso selection → remove points, click-to-copy metadata, reset.
- Safari clipboard fix: textarea must be `position:absolute;left:-9999px` (off-screen), NOT `opacity:0`.
- `go.Scatter3d` does NOT support lasso2d — lasso only works on the 2D companion.

---

## 11. Common gotchas

- **`set_slot` not inherited**: never call `set_slot` inside `(subshell) &` or a script invoked with `bash script.sh &` from another shell. The `wait <pid>` trick also fails across shell boundaries — use polling (`until [ $(find ... | wc -l) -ge N ]`) instead.
- **`normalize_combined.py` appends `_feat_selected`**: always pass output paths WITHOUT the suffix when using `--feature_config`. The script adds it automatically.
- **No `--steps` in `run_all_analysis.py`**: call individual plot scripts directly to re-run a single step.
- **kd-trees and ball-trees are useless in 448+ dims**: use `sklearn.metrics.euclidean_distances` for batch distance computation, then slice per group.
- **`compute_prdc`** (from `prdc` package) internally calls `np.partition(arr, kth=nearest_k+1)`, so `n_fake` must be ≥ `nearest_k + 2`.
- **Stale slot processes**: if `set_slot` jobs hang waiting for a slot, check `systemctl status ml-1slot-N.slice` for zombie PIDs from old sessions and kill them.
