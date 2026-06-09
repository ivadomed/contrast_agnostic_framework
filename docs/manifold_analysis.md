# Manifold Analysis — Technical Reference

The contrast manifold analysis pipeline measures how well synthetic guidance maps cover the real MRI contrast space. It answers: *do our synthetic images actually look like the multi-scanner, multi-contrast data a segmentor will encounter at test time?*

All scripts live under `analysis/contrast_manifold/scripts/`. Outputs land under `analysis/contrast_manifold/outputs/`.

---

## Pipeline overview

```
1. Feature extraction      extract_features_*.py       → raw CSV (one row per scan)
2. Normalization            normalize_combined.py        → z-scored + feature-selected CSVs
3. Analysis                 run_all_analysis.py          → PCA plots, UMAP, PRDC, coverage, ...
```

Each step is idempotent: re-running it overwrites the previous output with updated results. Always run steps in order — a normalized CSV is required before analysis; a raw feature CSV is required before normalization.

---

## Feature space: `regional_hist_64`

### What it is

Each MRI volume is represented as a **448-dimensional vector** built from intensity histograms computed independently in 7 anatomical brain regions:

| Index | Region name | FreeSurfer labels |
|---|---|---|
| 0 | `white_matter` | 2, 41 |
| 1 | `cortical_gm` | 3, 42 |
| 2 | `csf_ventricles` | 4, 5, 14, 15, 43, 44 |
| 3 | `subcortical_gm` | 10–13, 17–18, 26, 28, 49–54, 58, 60 |
| 4 | `cerebellum` | 7, 8, 46, 47 |
| 5 | `brainstem` | 16 |
| 6 | `whole_brain` | union of all above |

Each region contributes **64 histogram bins** → 7 × 64 = **448 features** total.

Column names follow the pattern `{region_name}_hist_{bin}`, e.g., `white_matter_hist_0` … `white_matter_hist_63`.

### Why this feature

`regional_hist_64` directly captures what the synthesis pipeline controls: the **per-tissue-class intensity distribution**. A T1w→T2w contrast change shows as a clear inversion of the white matter and grey matter histogram peaks. The 7-region decomposition is coarse enough to be stable across subjects and scanners while fine enough to discriminate modality-specific patterns.

Compared to alternatives:
- **Global histogram**: loses all regional tissue-contrast information.
- **HOG**: captures spatial texture and gradient orientation, which is mostly determined by anatomy and acquisition physics — not directly controlled by the intensity remap. Also requires reliable P95 estimates, which is problematic for small real groups.
- **CURIA embeddings**: harder to interpret, slower to compute, depends on encoder training distribution.

### How it is computed

Script: `extract_features_regional_hist.py`

1. Load the scan + its SynthSeg label map (looked up from `data/ON-Harmony/derivatives/synthseg_masks/`).
2. Resample label map to scan space if shapes differ.
3. Normalize scan intensities to [0, 1] using the **p1–p99** percentiles of the whole-brain foreground voxels.
4. For each region: collect voxel intensities, compute a 64-bin histogram on [0, 1], L1-normalize the histogram (so it sums to 1). Regions with fewer than 50 voxels after resampling are skipped (row filled with NaN).
5. Concatenate the 7 region histograms into a single 448-dim row.

```bash
# Example — synthetic mode, set_slot 0-3 for all CPU workers
set_slot 0-3 .venv/bin/python analysis/contrast_manifold/scripts/extract_features_regional_hist.py \
    --mode synthetic \
    --synth-root data/ON-Harmony/derivatives/synthetic_v26_6_guidance_lhc \
    --output-csv analysis/contrast_manifold/outputs/data/synthetic_v26_6_guidance_lhc/regional_hist_64/synthetic_v26_6_guidance_lhc_features.csv \
    --n-workers 224
```

---

## Normalization

Script: `normalize_combined.py`

Takes one original CSV and one synthetic CSV and:
1. **Fits a `StandardScaler`** (zero mean, unit variance) jointly on the combined real + synthetic data.  Using a joint scaler means both datasets are in the same normalized space.
2. **Downsamples the original** to at most `--n_per_contrast 100` rows per contrast type (stratified by scanner model). This prevents dominant contrasts (GRE: 3270 rows) from controlling the PCA.
3. **Applies feature selection** when `--feature_config` is provided: drops all-NaN columns, zero-variance columns, and any additional columns specified in the YAML. The suffix `_feat_selected` is appended to the output filename automatically — do **not** include it in the `--output_*` paths.

Output files (written to the **version-specific synthetic directory**, not the global original dir):
```
synthetic_v26_6_guidance_lhc/regional_hist_64/
  on_harmony_features_normalized_combined_downsampled100_feat_selected.csv    # real, re-fitted scaler
  synthetic_v26_6_guidance_lhc_features_normalized_combined_feat_selected.csv # synthetic
```

> **Why per-version normalized originals?** The scaler is fitted jointly on real + synthetic. A different synthetic version shifts the joint distribution slightly, so each version gets its own normalized copy of the originals. This ensures the analysis for version A is never accidentally invalidated by re-normalizing with version B's data.

```bash
set_slot 0 .venv/bin/python analysis/contrast_manifold/scripts/normalize_combined.py \
    --original_csv  analysis/contrast_manifold/outputs/data/original/regional_hist_64/on_harmony_features.csv \
    --synthetic_csv analysis/contrast_manifold/outputs/data/synthetic_v26_6_guidance_lhc/regional_hist_64/synthetic_v26_6_guidance_lhc_features.csv \
    --output_original  analysis/contrast_manifold/outputs/data/synthetic_v26_6_guidance_lhc/regional_hist_64/on_harmony_features_normalized_combined_downsampled100.csv \
    --output_synthetic analysis/contrast_manifold/outputs/data/synthetic_v26_6_guidance_lhc/regional_hist_64/synthetic_v26_6_guidance_lhc_features_normalized_combined.csv \
    --feature_config   analysis/contrast_manifold/config/feature_selection.yaml
```

---

## Analysis

Script: `run_all_analysis.py`

Runs all downstream analysis steps for a registered version. Each step is a separate subscript call. Steps and their outputs:

| Step key | Script | What it produces |
|---|---|---|
| `feature_analysis` | `analyze_features.py` | feature divergence, PCA loadings, UMAP-axis correlations |
| `contrast_clustering` | `analyze_contrast_clustering.py` | LDA 3D (original-only and with synthetic), feature F-scores |
| `pca` | `plot_umap_joint.py --plot_pca --skip_umap` | PCA scatter plots (several variants), loading heatmap |
| `umap` | `plot_umap_joint.py` | UMAP 3D interactive + 2D lasso |
| `coverage` | `plot_coverage.py` | recall curves, coverage heatmap |
| `pca/prdc_pca90` | `plot_prdc.py --pca-variance 0.90` | PRDC + Vendi at 90% PCA variance |
| `pca/prdc_pca60` | `plot_prdc.py --pca-variance 0.60` | PRDC + Vendi at 60% PCA variance |

```bash
# Run a single version
set_slot 0 .venv/bin/python analysis/contrast_manifold/scripts/run_all_analysis.py \
    --mask-type regional_hist_64 \
    --only v26_6_guidance_lhc_r1

# Run all versions in parallel (4 slots)
for rank in 0 1 2 3; do
    set_slot $rank .venv/bin/python analysis/contrast_manifold/scripts/run_all_analysis.py \
        --mask-type regional_hist_64 --rank $rank --world-size 4 \
        > /tmp/analysis_r${rank}.log 2>&1 < /dev/null &
done
```

---

## PCA plots

Location: `plots/v26/v26_6_guidance_lhc_r1/regional_hist_64/pca/`

### Variants

| Filename | What it shows |
|---|---|
| `pca_original_axes_with_synth_3d.html` | PCA fitted on **real only**, synthetic projected. ⭐ Main diagnostic. |
| `pca_original_axes_with_synth_2d.html` | Same, 2D with lasso selection (remove points, click to copy metadata). |
| `pca_original_axes_only_{2d,3d}.html` | Real data only in real-data PCA axes. Reference for group positions. |
| `pca_joint_with_synth_{2d,3d}.html` | PCA fitted on **real + synthetic jointly**. Emphasizes where synthetic adds new modes. |
| `pca_joint_original_only_{2d,3d}.html` | Joint-PCA space, real only. |
| `pca_loadings.pdf` | Feature contributions to PC1–PC10 (which brain regions drive each axis). |
| `pca_loading_heatmap.pdf` | Heatmap of all 448 feature loadings across top PCs. |

### How to read `pca_original_axes_with_synth_3d.html`

- **Coloured clusters**: real ON-Harmony scans, coloured by modality (T1w=red, T2w=blue, FLAIR=green, dwi=orange, bold=purple, epi=cyan, GRE=brown). Marker shape encodes scanner model.
- **Grey semi-transparent cloud**: all 1650 synthetic samples.
- **Good coverage**: grey cloud wraps around or interpenetrates the coloured clusters.
- **Coverage gap**: a coloured cluster sits in empty grey space (no synthetic nearby). Those groups will have low PRDC recall.
- **PCA axes are the real-data axes**: synthetic data projected into the same space. Tight clusters that stay T1w-like occupy the same region of PC1–PC2 as real T1w.
- The axes capture 95% of real-data variance, but this typically needs ~40–60 PCs. The 3D view shows only PC1–PC3 (15–25% of variance) — use it for orientation, not completeness.

### Gotchas

- `go.Scatter3d` does **not** support lasso selection — use the `_2d.html` companion for point selection.
- Safari clipboard: the 2D lasso plot uses a textarea at `position:absolute; left:-9999px` (not `opacity:0`) so `execCommand('copy')` works in Safari.

---

## UMAP

Location: `plots/.../umap/`

| Filename | Description |
|---|---|
| `umap_joint_3d.html` | 3D UMAP of real + synthetic jointly embedded. |
| `umap_joint_2d.html` | 2D lasso companion. |
| `umap_joint_coords.csv` | UMAP coordinates + metadata per point. |
| `connectivity/` | Minimum spanning tree and bottleneck analyses. |

UMAP is fitted on **real + synthetic jointly** (co-embedding), so the embedding reflects how the two distributions co-exist in the learned latent space. Real clusters that synthetic doesn't reach appear isolated; well-covered groups merge into connected regions.

UMAP is slower than PCA (a few minutes for 448-dim, 2800-point data). The `--skip_umap` flag on the PCA step lets you re-run PCA without waiting for UMAP.

---

## PRDC metrics

Location: `plots/.../pca/prdc_pca60/prdc_metrics.csv` and `prdc_pca90/prdc_metrics.csv`

### Two PCA variance thresholds

The PRDC is computed in PCA-reduced space (not in the full 448-D space) to avoid the curse of dimensionality:
- **`prdc_pca60/`**: PCA captures 60% of variance — fewer, dominant axes. Distances are coarser; IND counts tend to be higher.
- **`prdc_pca90/`**: PCA captures 90% of variance — more axes, finer resolution. More informative but noisier for small groups.

`prdc_pca60` is used for cross-version comparisons (reported in all experiment tables). `prdc_pca90` is the default shown in the HANDOVER.

### Column reference

| Column | Meaning |
|---|---|
| `group` | `{modality} × {scanner}` label, e.g. `T2w × Siemens Trio` |
| `n_real` | Number of real scans in this group |
| `n_synth_total` | Total synthetic samples (constant across rows = 1650) |
| `real_median_nn_dist` | Median nearest-neighbour distance among real samples — proxy for cluster tightness |
| `real_p95_nn_dist` | **IND/OOD boundary**: P95 of within-group real NN distances |
| `n_ind` | Synthetic samples with distance-to-nearest-real ≤ `real_p95_nn_dist` |
| `n_ood` | Complement: `n_synth_total - n_ind` |
| `ood_pct` | `100 × n_ood / n_synth_total` — almost always 97–100% by design |
| `prdc_k` | k used for PRDC computation (`—` if insufficient IND samples) |
| `precision` | Fraction of IND synthetic whose k-NN are all real (quality of IND coverage) |
| `recall` | Fraction of real samples covered by ≥1 IND synthetic neighbour |
| `density` | Mean IND synthetic per real neighbourhood / k (local density ratio) |
| `coverage` | Fraction of real samples with ≥1 IND synthetic within `real_p95_nn_dist` |
| `ood_mean_norm_dist` | Mean OOD distance / `real_p95_nn_dist` — how far outside the cluster the OOD samples sit |
| `ood_var_norm_dist` | Variance of the above — spread of OOD exploration |
| `vendi_diversity_score` | Effective number of distinct modes in the **full** synthetic distribution (cosine kernel) |

### Constants

```python
THRESHOLD_PCT  = 95   # real_p95_nn_dist uses the 95th percentile of within-group NN dists
MAX_K          = 3    # maximum k for PRDC (k chosen as min(3, floor(n_ind/2 - 1)))
MIN_REAL       = 4    # skip group if fewer than 4 real samples
MIN_IND        = 5    # skip PRDC if fewer than 5 IND synthetic (prdc internally uses kth=k+1)
MAX_OOD_VENDI  = 150  # Vendi is O(n³); cap OOD sample count for tractability
```

### How to interpret

**Step 1 — Check IND coverage.** A group with `n_ind = 0` means no synthetic sample landed inside that real cluster. PRDC metrics will be `—`. This is normal for unusual acquisition protocols; it signals a gap in domain randomization.

**Step 2 — For IND-eligible groups (`n_ind ≥ 5`), read recall and density.**
- **High recall** (→ 1.0): the synthetic distribution covers the real cluster well. Good for training generalization.
- **Low recall** (< 0.5): the synthetic cloud misses parts of the real cluster. Some real acquisition settings won't have training analogues.
- **High density**: many synthetic samples per real neighbourhood — the cluster is densely covered.
- **Low density**: sparse coverage — the synthetic distribution skims the cluster boundary.

**Step 3 — Check OOD exploration (`ood_mean_norm_dist`).**  
A value of 10 means OOD synthetic samples sit 10× the real cluster radius away from the nearest real scan. Higher values = more aggressive domain randomization. For v26_6: ~17, for SynthSeg-modeB: ~18.

**Step 4 — Vendi as a global diversity signal.**  
Vendi measures the effective number of distinct contrast archetypes in the full synthetic distribution. Values around 2.7–2.9 are typical for affine-remap methods. Values above 3.5 usually indicate degenerate (flat-constant) synthetic images, not meaningful diversity — do not interpret Vendi > 3 as better generalization without inspecting the actual images.

### Reference group for cross-version comparison

`GRE × Siemens Trio` (n_real = 32) is the most stable reference:
- Consistently has 25–33% IND across versions (enough for PRDC, not saturated).
- Large enough cluster (32 scans) that the P95 threshold is stable.
- Scanner/contrast combination with distinctive HOG and intensity patterns.

### Known limitation: small-group P95 inflation

Groups with `n_real < 20` have unreliable `real_p95_nn_dist`. The P95 of 6–10 NN distances can be very large (essentially the max), making the IND boundary enormous and artificially inflating `n_ind`. This is most visible in regional_hist_64 for rare scanner/modality combinations. Always cross-check with the PCA visual when `n_ind` is suspiciously large relative to `n_real`.

---

## Coverage plots

Location: `plots/.../coverage/`

These complement PRDC with a **scale-independent** view of recall:

| File | Description |
|---|---|
| `coverage_recall_curve.pdf` | Recall vs ε (distance threshold as multiples of the real cluster scale) for each modality |
| `coverage_heatmap.pdf` | Recall at 1× scale per modality × scanner cell (heatmap) |
| `coverage_scanner_modality.pdf` | Same per-cell recall as a bar chart |
| `coverage_metrics.csv` | Tabular: `eps_50pct_recall`, `eps_90pct_recall`, recall at 0.25×, 0.5×, 1×, 2×, 4× scale, synth diversity, hull coverage |

**Key columns in `coverage_metrics.csv`:**
- `ref_scale_median_orig_nn`: median NN distance among real scans for this modality — the "1× scale" reference.
- `recall_1.00x_scale`: fraction of real scans with ≥1 synthetic within 1× the reference scale.
- `eps_50pct_recall`: how many reference-scale radii you need to expand to cover 50% of real scans.
- `synth_diversity_mean_self_nn`: mean NN distance among synthetic samples — diversity proxy.

The **recall curve** is the most informative: it shows what fraction of real scans are covered as a function of the search radius. A curve that rises quickly (coverage near 1× scale) means the synthetic cloud is dense and well-aligned. A flat curve near 0 until very large ε means the synthetic cloud is far from all real scans of that modality.

---

## Contrast clustering

Location: `plots/.../contrast_clustering/`

| File | Description |
|---|---|
| `lda_3d_original.html` | 3D LDA of real data only, coloured by modality. Shows how well the feature space separates real acquisitions. |
| `lda_3d_with_synthetic.html` | Same axes, synthetic overlaid. Tells you which modality LDA zone each synthetic sample "looks like". |
| `feature_fscores.csv` | F-score of each feature for separating modalities (which bins discriminate T1w vs T2w vs FLAIR etc.) |
| `feature_scatter_top5_with_synthetic.pdf` | Scatter plots of the 5 highest-F-score features: real clusters + synthetic overlay. |

LDA projects the 448-D feature space onto the axes that best separate the known modality groups. Synthetic data projected onto LDA axes tells you which contrast archetype each synthetic image most resembles. A good generator produces synthetic samples spread across all LDA zones; a T1w-biased generator concentrates them in the T1w zone.

---

## Feature analysis

Location: `plots/.../feature_analysis/`

| File | Description |
|---|---|
| `feature_divergence.pdf` | Per-feature JS divergence between real and synthetic marginal distributions. High divergence = the feature is distributed very differently in synthetic vs real. |
| `feature_pca_loadings.pdf` | Contribution of each of the 448 features to the top 5 PCA components (which bins and which regions drive the main axes). |
| `feature_umap_corr_axis{1,2,3}.pdf` | Spearman correlation of each feature with UMAP axis 1/2/3 — which features explain the manifold topology. |

The **divergence plot** is useful for diagnosing systematic biases. If `white_matter_hist_0` (very dark WM bin) has high JS divergence, synthetic images don't generate WM-dark contrasts as frequently as real data.

---

## Path conventions

```
analysis/contrast_manifold/outputs/
  data/
    original/
      regional_hist_64/
        on_harmony_features.csv                                               # raw real features
        on_harmony_features_normalized_combined_downsampled100_feat_selected.csv  # global normalized (legacy only)
    synthetic_<version>_guidance_lhc/
      regional_hist_64/
        synthetic_<version>_guidance_lhc_features.csv                         # raw synth features
        on_harmony_features_normalized_combined_downsampled100_feat_selected.csv  # real, re-fitted for this version
        synthetic_<version>_guidance_lhc_features_normalized_combined_feat_selected.csv  # synth, same scaler
  plots/
    v<major>/
      <version>_guidance_lhc_r1/
        regional_hist_64/
          pca/                   # PCA plots + PRDC CSVs
            prdc_pca60/
              prdc_metrics.csv
              prdc_metrics.json
            prdc_pca90/
              prdc_metrics.csv
              prdc_metrics.json
            pca_original_axes_with_synth_3d.html
            pca_original_axes_with_synth_2d.html   # lasso interactive
            pca_joint_with_synth_3d.html
            pca_loadings.pdf
          umap/
            umap_joint_3d.html
            umap_joint_2d.html
          coverage/
            coverage_recall_curve.pdf
            coverage_heatmap.pdf
            coverage_metrics.csv
          contrast_clustering/
            lda_3d_original.html
            lda_3d_with_synthetic.html
            feature_fscores.csv
          feature_analysis/
            feature_divergence.pdf
            feature_pca_loadings.pdf
```

---

## Registering a new version

Add a tuple to `VERSIONS` in `run_all_analysis.py`:

```python
(
    "v26",                          # major version → subdirectory name
    "v26_6_guidance_lhc_r1",        # run name (must be unique; _r1 = run index)
    {                               # normalized synth CSV paths, per feature type
        "regional_hist_64": DATA_ROOT / "synthetic_v26_6_guidance_lhc" / "regional_hist_64"
                    / "synthetic_v26_6_guidance_lhc_features_normalized_combined_feat_selected.csv",
    },
    {                               # raw synth CSV paths
        "regional_hist_64": DATA_ROOT / "synthetic_v26_6_guidance_lhc" / "regional_hist_64"
                    / "synthetic_v26_6_guidance_lhc_features.csv",
    },
    {                               # normalized real CSV path (version-specific, re-fitted scaler)
        "regional_hist_64": DATA_ROOT / "synthetic_v26_6_guidance_lhc" / "regional_hist_64"
                    / "on_harmony_features_normalized_combined_downsampled100_feat_selected.csv",
    },
),
```

The 5th element (override per-version real CSV) is optional. When omitted, the global `ORIG_CSVS[mask_type]` path is used instead — but this is the legacy path that may not have the right scaler for the current synthetic version.

---

---

## Feature space: `hog3d_512`

### What it is

Each MRI volume is represented as a **512-dimensional vector** of 3D Histogram of Oriented Gradients computed on the brain crop at **native voxel pitch** (no resize to a fixed cube). A 4×4×4 spatial cell grid is defined proportionally over the crop; within each cell, voxel gradients are projected onto 8 reference directions uniformly distributed on the upper hemisphere (antipodal/unsigned) using a Fibonacci lattice. Each cell histogram is magnitude-weighted and L2-normalised independently.

Feature column naming: `hog3d_c{i}{j}{k}_o{orientation}`, e.g. `hog3d_c023_o7` = cell (0,2,3), orientation 7.

Total: 4³ × 8 = **512 features**.

### Why native resolution matters

The extractor uses an **adaptive stride** so that the subsampled voxel count is ≈ 250 k:
```
stride = max(1, round((crop_voxels / 250_000)^(1/3)))
```

Key consequence: a T1w at 1 mm iso (crop ≈ 7 M voxels, stride = 3, effective 3 mm) and a bold at 3 mm iso (crop ≈ 135 k voxels, stride = 1, native 3 mm) end up at comparable effective spatial scales. **Do not resize to a fixed cube** — it would upsample coarse-resolution images (adding fake fine-scale gradients) and downsample fine-resolution images, making them artificially similar.

The old extractor used `skimage.resize(..., anti_aliasing=True)` to 64³, which caused this problem. The current extractor (`extract_features_hog3d.py`, v2) avoids it.

### Extraction command

```bash
# Use set_slot 0-3 for all 256 CPU workers in one command (recommended)
set_slot 0-3 .venv/bin/python analysis/contrast_manifold/scripts/extract_features_hog3d.py \
    --mode synthetic \
    --synth-root data/ON-Harmony/derivatives/synthetic_v28_1_guidance_lhc \
    --output-csv analysis/contrast_manifold/outputs/data/synthetic_v28_1_guidance_lhc/hog3d_512/synthetic_v28_1_guidance_lhc_features.csv \
    --n-workers 224
```

Speed: ≈ 10 s for 1650 synthetic files (224 workers), ≈ 30 s for 7803 originals (224 workers). The dominant cost is NIfTI I/O, not computation.

> **Important**: when the extractor changes (different algorithm or parameters), **always re-extract the originals** using the new extractor before comparing against new synthetic versions. Mixing old-extractor originals with new-extractor synthetics makes metrics meaningless.

---

## PRDC reliability for hog3d_512

### The P95 inflation problem

PRDC computes `real_p95_nn_dist` (the IND/OOD boundary) from within-group real NN distances. For groups with small `n_real` (< 20), the P95 is essentially the maximum NN distance in the cluster. In HOG3D space — where within-group variance is high (many subjects, many scan sessions) — this produces enormous thresholds that engulf most or all synthetic samples, inflating `n_ind` artificially.

**Observed example:** `T1w × GE MR750` (n_real=6): real_p95_nn_dist = **6.26** while the feature space range after normalisation is ≈ [−5, +9]. A threshold of 6.26 covers essentially the entire feature space, so n_ind = 1618/1650 is meaningless.

**Reliable groups:** only those with `n_real ≥ 30` yield stable P95 thresholds. In the ON-Harmony dataset this is primarily the GRE groups (n_real = 32–182).

### What to use instead

For HOG3D analysis, prefer **`coverage_metrics.csv`** over the PRDC table:

| Metric | File | What it measures | Reliable for HOG3D? |
|---|---|---|---|
| `n_ind`, `recall`, `density` | `prdc_pca60/prdc_metrics.csv` | Cluster penetration | **Only for n_real ≥ 30** |
| `recall_1.00x_scale` | `coverage/coverage_metrics.csv` | Coverage at 1× real cluster radius | ✓ Modality-level, robust |
| `mean_nn_dist_orig_to_synth` | `coverage/coverage_metrics.csv` | Mean distance real→nearest synthetic | ✓ Best single number |
| `ood_mean_norm_dist` | `prdc_pca60/prdc_metrics.csv` | OOD exploration (all groups) | ✓ Doesn't depend on P95 |
| PCA / UMAP visual | `pca/*.html`, `umap/*.html` | Distributional overlap | ✓ Always informative |

### HOG3D baseline (v26_6 guidance LHC)

Measured with the native-resolution extractor, 1650 synthetic samples:

| Modality | `mean_nn_dist / ref_scale` | `recall_1×` | `recall_4×` | GRE PRDC |
|---|---|---|---|---|
| T1w | 1.03 | 0.48 | 1.00 | — |
| T2w | 1.60 | 0.00 | 1.00 | — |
| FLAIR | 1.54 | 0.00 | 1.00 | — |
| bold | 2.69 | 0.00 | 1.00 | — |
| dwi | 2.53 | 0.00 | 0.99 | — |
| epi | 2.79 | 0.00 | 1.00 | — |
| **GRE** | **4.92** | **0.00** | **0.12** | 0/7 IND |

`ref_scale = 7.14` (median within-group NN distance across all real scans).

The `recall_1×` = 0 for every non-T1w modality means no synthetic sample lands within 1× the real cluster radius of any real bold/GRE/dwi/epi/T2w scan. Only at 4× the radius (a very forgiving criterion) do bold/dwi/epi/T2w/FLAIR become ≥ 99% covered — but GRE remains at only 12% even then.

### Root causes of the gap

**1. Resolution mismatch.** All v26_6 guidance maps are 1 mm isotropic (T1w-derived). Real acquisitions span:
- T2w/FLAIR: 1–2 mm iso
- DWI: 2 mm iso
- bold/EPI: 3–4 mm iso

With the native-res extractor (stride = 3 for all 1 mm iso synthetics), the synthetic images are sampled at 3 mm effective pitch. But bold at native 3 mm has coarser, blurrier tissue boundaries and completely different noise characteristics. The HOG gradient patterns differ even at the same effective pitch because the *source* PSF differs.

**2. GRE susceptibility dropout.** GRE sequences are highly sensitive to B₀ field inhomogeneity (χ effects). This creates characteristic signal voids in inferior brain regions (orbitofrontal cortex, inferior temporal lobe, near the petrous ridge). These voids generate very strong and spatially specific gradient discontinuities that T1w-derived intensity remapping cannot produce. The most discriminative HOG3D features (from F-score analysis) are exactly in inferior-brain cells (`c{i}{j}3`, k=3 = most inferior layer) with near-vertical orientation (o7, the highest Fibonacci direction ≈ superior-inferior axis) — the precise signatures of susceptibility dropout.

**3. Acquisition noise texture.** Bold/EPI have low SNR and Rician noise that creates a high-frequency texture pattern in the gradient orientation histograms. Synthetic images (pure intensity remaps, no noise) have unnaturally smooth gradients.

### Most discriminative HOG3D features (F-score ranking)

Top features from `feature_fscores.csv` (F-score for separating modality × scanner groups):

| Rank | Feature | F-score | Interpretation |
|---|---|---|---|
| 1 | `hog3d_c023_o7` | 5184 | Anterior-inferior brain, vertical gradient |
| 2 | `hog3d_c013_o7` | 4009 | Anterior-inferior, vertical |
| 3 | `hog3d_c313_o7` | 3916 | Posterior-inferior, vertical |
| 4 | `hog3d_c323_o7` | 3248 | Posterior-inferior, vertical |
| 5 | `hog3d_c303_o7` | 2729 | Posterior-inferior, vertical |

Cell indices `c{i}{j}{k}`: i = anterior-posterior (0=anterior, 3=posterior), j = lateral, k = superior-inferior (3 = most inferior). Orientation 7 ≈ z-axis (superior-inferior direction).

All top features share: **inferior brain** (k=3) and **superior-inferior gradient orientation** (o7). This is precisely where GRE susceptibility dropout dominates.

---

## v28 family — improving HOG3D coverage

### Strategy

Three independent interventions, in priority order:

1. **Resolution simulation** — target bold/DWI/EPI (2.5–5× too far). Downsample guidance maps to 3–4 mm effective resolution, matching what the native-res extractor sees in real bold/EPI at stride=1.

2. **Rician noise injection** — target all non-T1w modalities. Adds acquisition-specific noise texture that changes gradient magnitude distributions.

3. **GRE susceptibility simulation** — target GRE only (4.9× too far). Add synthetic signal voids in inferior brain regions to mimic χ dropout.

### v28_1: aggressive resolution diversity + Rician noise

Inherits V26_6 intensity remap. After the remap, adds:
- Rician noise: `y_noisy = √((y + N(0,σ))² + N(0,σ)²)`, σ ~ U(0, 0.12)
- The guidance pipeline applies zoom drawn from U(0.20, 1.0) (vs v26_6's U(0.40, 1.0))

At zoom=0.20, effective resolution = 1 mm / 0.20 = 5 mm — covering the EPI regime. Combined with Rician noise, this should close most of the bold/DWI/EPI HOG gap while keeping the regional_hist_64 results stable (intensity distribution is mostly preserved by partial-volume blurring at tissue boundaries).

## Common gotchas

- **`normalize_combined.py` appends `_feat_selected`**: always pass output paths WITHOUT the suffix when using `--feature_config`. The script adds it automatically; passing it explicitly produces double-suffixed files that won't be found by the analysis registry.
- **Concurrent normalization + extraction**: if normalization runs while the raw feature CSV is still being written, it reads a partial CSV. Always confirm that the raw CSV row count matches `n_subjects × n_variants` before running normalization.
- **kd-trees are useless in 448-D**: `plot_prdc.py` uses `sklearn.metrics.euclidean_distances` for batch distance computation, then slices per group. Do not try to accelerate it with kd-trees or ball-trees.
- **`compute_prdc` internal k+1**: the `prdc` package internally calls `np.partition(arr, kth=k+1)`, so `n_ind` must be ≥ k+2 = 5. This is enforced by `MIN_IND = 5`.
- **PRDC unreliable for `n_real < 20`**: the P95 threshold is estimated from too few samples and can be enormous, inflating `n_ind`. For small groups, use the PCA visual and `ood_mean_norm_dist` instead of recall.
- **Vendi cap**: Vendi is O(n³) in the number of OOD samples. `MAX_OOD_VENDI = 150` caps the OOD sample count to keep it tractable. The Vendi score is therefore based on a random subsample, not all OOD samples.
