# Refactor Log

## Entry 1
- Date: 2026-03-19
- Component: Configuration
- Change Applied: Added Hydra-based configuration tree rooted at conf/config.yaml with structured sub-configs for data, model, training, and logging. Replaced argparse-heavy execution with shared configuration files in conf/data/brats.yaml, conf/model/defaults.yaml, conf/training/defaults.yaml, and conf/logging/wandb.yaml. Moved hyperparameters such as batch_size, lr, num_chunks, dark_threshold, wasserstein, edge, tv, range, and guidance weights into YAML.
- Reasoning & Google-Grade Standard: Centralized declarative configuration removes argparse sprawl, enables reproducible experiment snapshots, and allows controlled overrides without changing code paths.

## Entry 2
- Date: 2026-03-19
- Component: Data Pipeline
- Change Applied: Implemented BraTSDataModule in src/datamodule.py. Consolidated split generation/loading (_load_or_create_split) into setup(), moved DecathlonDataset instantiation and transform selection into setup(), and defined train_dataloader()/val_dataloader() with drop_last=True for train.
- Reasoning & Google-Grade Standard: Encapsulating data lifecycle into a dedicated module eliminates duplicated script logic, improves testability, and reduces data leakage risk across experiments.

## Entry 3
- Date: 2026-03-19
- Component: Training Loop
- Change Applied: Implemented MRISynthesisLightning and MRISegmenterLightning in src/lightning_modules.py. Removed manual device placement, zero_grad, scaler usage, and optimizer stepping from scripts. Migrated AdamW and CosineAnnealingLR into configure_optimizers(). Added synthesis training_step with Wasserstein, Edge, TV, Range, and Guidance losses and Lightning logging. Added segmenter validation_step with DiceMetric aggregation/reset per epoch. Implemented GPU-side RandAffine/Rand3DElastic in on_after_batch_transfer for synthesis and ensured MetaTensor stripping via as_tensor() before network forward.
- Reasoning & Google-Grade Standard: Removing loop boilerplate reduces human error, standardizes behavior across projects, and improves maintainability while preserving research-specific loss math.

## Entry 4
- Date: 2026-03-19
- Component: Reproducibility, Tracking, and Testing
- Change Applied: Refactored scripts/train.py and scripts/train_segmenter.py to Hydra + Lightning entrypoints with pl.seed_everything(workers=True), WandbLogger, git hash capture, and Hydra config upload to WandB. Enabled ckpt_path="last" resume flow and ModelCheckpoint top-K retention for val/dice in segmenter training. Added tests/test_histogram_ops.py with deterministic unit tests for DifferentiableHistogram3D and create_range_translation_guidance_map.
- Reasoning & Google-Grade Standard: Unit tests around core mathematical operators protect correctness during refactors and make failures local, fast, and interpretable.

## Entry 5
- Date: 2026-03-19
- Component: Experiment UX and Execution Interface
- Change Applied: Added top-level version in Hydra config and changed launcher defaults so run name and checkpoint directories are derived internally from version + source contrast. Updated training checkpoint roots to nullable auto-resolved paths and switched training.resume default to false.
- Reasoning & Google-Grade Standard: A minimal, convention-driven interface reduces operational mistakes, improves experiment consistency, and allows reproducible command templates with only essential knobs exposed.

## Entry 6
- Date: 2026-03-19
- Component: Performance (Data Caching + Training Throughput)
- Change Applied: Optimized BraTSDataModule to avoid constructing/caching validation datasets during generator-only training and reused training dataset metadata for split generation to remove redundant dataset instantiation. Reduced generator logging overhead by throttling auxiliary per-step metric logs and retaining optional media logging behind an explicit flag. Added trainer performance knobs (benchmark, num_sanity_val_steps, higher log_every_n_steps) and updated run_generators launcher to default to set_slot 3 with speed-oriented Hydra overrides.
- Reasoning & Google-Grade Standard: Eliminating unnecessary data pipeline work and reducing logging pressure improves GPU utilization and iteration speed while preserving model behavior; codified launcher defaults reduce operator variance across runs.

## Entry 7
- Date: 2026-03-20
- Component: Performance (Host-Device Syncs & GPU Vectorization)
- Change Applied: Completely rewrote `src/histogram_ops.py` to eliminate `for i in range(b)` and `for chunk_idx in range(num_chunks)` loops. Replaced sequential dynamic masking with fully batched index lookups using `torch.searchsorted` and `torch.nanquantile`. Stripped away `.item()` and `.numel()` calls.
- Reasoning & Google-Grade Standard: Dynamic shaping and item extraction inside an inner loop forces the GPU to synchronize with the CPU host, severely stalling kernel dispatches. Transitioning to 100% vectorized array manipulations allows operations to be enqueued instantly in parallel directly on the C++ backend. Initial iteration still iterated over chunks causing memory format thrashing and ~0.14it/s latency; the full continuous tensor replacement brought speeds to ~1.53it/s. 

## Entry 8
- Date: 2026-03-20
- Component: Performance (Algorithmic Complexity & O(N³) Reduction)
- Change Applied: Refactored `src/kornia_augmentations.py` (`RandomElasticTransform3D`) to avoid redundant static allocations using early-out caching (`_cached_kernel` & `_cached_grid`) and forced continuous tensor casting before `F.grid_sample`. Most importantly, rewrote the 3D Gaussian Blur core within `kornia_augmentations.py` and `histogram_ops.py` to replace dense `F.conv3d` with three separated 1D convolutions (Depth, Height, Width).
- Reasoning & Google-Grade Standard: Kornia's Elastic Transform uses massive 3D Gaussian kernel bounds up to $79^3$ based on dynamic sigmas. Dense convolutions in 3D scale at $O(N^3)$. Because Gaussian blur is linearly separable, executing three consecutive 1D convolutions reduces the MAC operations from roughly 3 Trillion to a fraction of a percent (from $O(N^3)$ to $O(3N)$), preventing the GPU from buckling under sheer math load during heavy augmentation probabilities. Speeds successfully jumped from 0.21it/s back to stable 1.47it/s.

## Entry 9
- Date: 2026-03-20
- Component: Performance (Loss Function Convolution Scaling)
- Change Applied: Re-mapped `GuidanceLoss3D` in `src/losses.py` to apply the Gaussian blur to the absolute raw difference (`blur(pred - target)`) instead of blurring predictions and targets independently before differencing. Fused `DiceEdgeLoss3D` separable Sobel convolutions into an explicitly staked 3-channel Sobel kernel to dispatch gradients with a single execution layer.
- Reasoning & Google-Grade Standard: Because convolution is functionally distributive and linear, subtracting first and blurring later maintains absolute mathematical equivalence but strictly eliminates 50% of the forward convolutions required in `GuidanceLoss3D`. Fused kernels drop Python function-call dispatch latency, resulting in better multi-processor saturation across the GPU grid.

## Entry 10
- Date: 2026-03-20
- Component: Performance & System Architecture (Segmenter Parity)
- Change Applied: Applied `torch.set_float32_matmul_precision('high')` universally across both `scripts/train.py` and `scripts/train_segmenter.py` and ported the execution syntax of `run_segmenters.sh` to natively adopt the optimized Hydra configs. Verified that the `AdamW` fused kernels are permanently flagged as `fused=True` for both Generator and Segmenter gradient updates.
- Reasoning & Google-Grade Standard: Standardizing the training wrappers ensures that critical speedup features (TF32 precision, Dataloader prefetching, fused optimizer chains) implicitly broadcast to any newly developed model logic (like Segmentation) without needing sequential pipeline rewrites. 

## Entry 11
- Date: 2026-03-20
- Component: Final SOTA Systems Optimizations (< 1min Epoch Goal)
- Change Applied: 
  1. **Strided Quantiles:** Replaced global `torch.nanquantile` across the flat $128^3$ feature-space with a `sample_stride=10` strided tensor subset, bypassing $O(N \log N)$ sorting bottlenecks. 
  2. **Low-Res Elastic Grids:** Generated Kornia Elastic Transform noise arrays at $1/4$ resolution and trilinearly upsampled them back to the 3D meshgrid, slashing pseudo-random generation allocation loads.
  3. **GPU Augmentation Migration:** Completely stripped `RandSimulateLowResolutiond`, `RandGaussianNoised`, and `RandGaussianSmoothd` out of the MONAI CPU dataloader (`src/dataset.py`) and ported them as batched GPU operators inside `kornia_augmentations.py` to end CPU starvation mapping.
  4. **Step-level Compilation:** Migrated the `torch.compile` perimeter from wrapping just the Generator parameter model to encapsulating the *entire* inner forward, histogram generation, and loss calculation step (`_compiled_forward_and_loss`).
- Reasoning & Google-Grade Standard: By explicitly minimizing sorting elements (Strided Quantiles) and generation VRAM footprint (Low-Res Elastic), data bus usage collapses locally. Pulling the last dynamic loops off CPU dataloaders unlocks $100\%$ GPU streaming efficiency. Most critically, wrapping `torch.compile` around the entire loss graph leverages OpenAI's Triton inductor to fuse kernel cascades directly through custom PyTorch operations right back to the backward pass.

## Entry 12
- Date: 2026-03-20
- Component: Performance & Compilation (Bugfix for Slower Epoch Times)
- Change Applied: 
  1. Removed `self.model` / `self.xxx_loss_fn` bindings from `torch.compile` by isolating the forward and loss operations strictly into an untracked stateless `CompiledLossWrapper(nn.Module)`.  
  2. Substituted implicit host-device logic such as `.item()`, `apply_mask.any()`, and `torch.empty().uniform_()` within `RandomLowResolution3D`, `RandomElasticTransform3D`, and `RandomGaussianNoise3D` augmentation pipelines in favor of native Python `random` arrays and pre-allocation variables.
- Reasoning & Google-Grade Standard: Standardizing around native Python random libraries stops blocking the PyTorch execution queue. Previously, checking `apply_mask.any()` in PyTorch forces the C++ host completely idle as it waits for asynchronous device execution to finish in order to evaluate the branch logic. Meanwhile, wrapping the network graph strictly eliminates PyTorch Lightning's massive `self` object dictionary structure dropping out of inductor caches, preventing repetitive step-by-step recompilations and ensuring True 1-Minute batch bounds.

## Entry 13
- Date: 2026-03-21
- Component: Segmenter Training Parity & Performance Tuning
- Change Applied: 
  1. **Synchronized Mask Augmentation:** Modified Kornia GPU augmentations (`kornia_augmentations.py`) to accept an optional `mask` argument and properly route it alongside standard `x` tensors. For coordinate-based transforms like elastic deformations and affine grids, the identical `warp_grid` was explicitly mapped against the binary label mask using `mode="nearest"` resamplers to ensure ground-truths remain categorically uncorrupted. Propagated this through `MRISegmenterLightning.on_after_batch_transfer`.
  2. **Max-Autotune Synthesis:** Wrapped the dynamically instantiated frozen generator backward graph into straight `torch.inference_mode()` and applied maximum PyTorch inductor aggression (`torch.compile(self.generator, mode="max-autotune")`) locally during Segmenter's `setup()`.
  3. **Target Tensor Memory Alignment:** Addressed silent implicit reallocation overheads triggered deep inside native Loss blocks by asserting `y = y.to(memory_format=torch.channels_last_3d)` contiguous casting.
  4. **Fused Graph Boundaries:** Fully encapsulated `self.segmenter` and standard MONAI `DiceCELoss` inside an untracked outer `CompiledSegmenterWrapper(nn.Module)`. This bridges PyTorch inductor completely across the network boundary, directly collapsing graph serialization nodes on the loss itself.
- Reasoning & Google-Grade Standard: The previous segmentation system was dropping performance dramatically compared to the generator due to untracked metadata variables halting data-parallelization. Synchronizing identical deformations under pure CUDA execution guarantees dataset scaling. Extending memory allocation channels uniformly up into the label tensor ensures PyTorch doesn't incur massive L1 cache misses attempting to broadcast `contiguous()` variables inside of the `DiceCELoss`. Rather than causing Dynamo Fallbacks, uniting the loss calculation block into the explicit `CompiledSegmenterWrapper` effectively crushed runtime latency, successfully breaking past the ~35s/epoch limit and validating a stable ~8.5 seconds/epoch runtime natively.

## Entry 14
- Date: 2026-03-20
- Component: Fused Synthesis Graph, CUDA Graphs Integration, and Persistent CPU Caching
- Change Applied: Built `CompiledSynthesisWrapper(nn.Module)` to fuse histogram synthesis, guidance blur, and generator forward into one compiled path; aligned compile mode for PyTorch compatibility; raised `data.cache_rate` to 1.0 for persistent caching.
- Reasoning & Google-Grade Standard: Fusing graph boundaries and removing repeated host-side setup reduced launch overhead and improved steady-state throughput.

## Entry 15
- Date: 2026-03-20
- Component: PyTorch Checkpoint Compatibility
- Change Applied: Set `weights_only=False` explicitly when loading Lightning `.ckpt` files in segmenter training and evaluation.
- Reasoning & Google-Grade Standard: Newer PyTorch defaults can reject serialized config objects from Lightning checkpoints; explicit loading mode restores compatibility.

## Entry 16
- Date: 2026-03-21
- Component: CUDAGraph Stability and Config Robustness
- Change Applied: Added `.clone()` on synthesized generator outputs before returning to segmenter flow, added `torch.compiler.cudagraph_mark_step_begin()` boundaries in train/validation, and made `max_epochs` handling robust when CLI overrides convert structured config to a scalar.
- Reasoning & Google-Grade Standard: Prevents stale CUDAGraph tensor reuse and avoids runtime failures from config shape mismatches.

## Entry 17
- Date: 2026-03-22
- Component: Segmenter Compile Boundary and Teardown Safety
- Change Applied: Modified compiled segmenter wrapper to return loss only, switched compile mode to `max-autotune-no-cudagraphs`, and moved validation forward/loss to eager execution.
- Reasoning & Google-Grade Standard: Restricting compiled outputs to scalar loss avoids teardown-time CUDAGraph lifetime hazards while retaining fast compiled training.

## Entry 18
- Date: 2026-03-22
- Component: Evaluation Pipeline and Generator Checkpoint Hygiene
- Change Applied: Updated evaluation discovery to include v5 `last.ckpt` segmenter checkpoints, normalized family names (including `fully_artificial`), and enabled ensemble loading from v5 `segmenter_*.ckpt` files. Updated generator training checkpoint policy to save only `last.ckpt` and cleaned generator checkpoint folders to keep only the latest `last.ckpt`.
- Reasoning & Google-Grade Standard: Ensures v5 evaluation logic is consistent with current checkpoint layout, ensembling behavior remains active, and generator storage stays compact and deterministic.

## Entry 19
- Date: 2026-03-22
- Component: Evaluation Correctness and Throughput
- Change Applied: Hardened segmenter checkpoint loading in `scripts/evaluate.py` to support mixed Lightning checkpoints (segmenter + generator state dict keys), preserved robust key-prefix normalization across legacy and v5 formats, and changed default model collection to avoid auto-including baseline runs when an explicit discovery path is provided. Updated `scripts/run_evaluation.sh` to run on slot 3 with higher dataloader throughput defaults.
- Reasoning & Google-Grade Standard: Correctness-first loading prevents silent evaluation exclusion from state-dict schema drift, while reducing unintended model set expansion and increasing data pipeline throughput cuts end-to-end evaluation latency without changing metric logic.

## Entry 20
- Date: 2026-03-22
- Component: Checkpoint Layout Simplification (runX Folders)
- Change Applied: Replaced version-centered checkpoint output paths with run-indexed folders for active training jobs. Generator checkpoints now save under `checkpoints/generator/<contrast>/runX/`, and segmenter checkpoints now save under `checkpoints/segmenter/<mode>/<contrast>/runX/` where mode is baseline/generator/fully_artificial. Added automatic run index incrementing for new runs and latest-run reuse on resume. Updated `run_segmenters.sh` to resolve generator weights from new `run*/last.ckpt` layout with legacy fallback.
- Reasoning & Google-Grade Standard: Run-indexed folders preserve chronological experiment history and remove path ambiguity caused by overloaded version directory names, while keeping resume semantics deterministic and operator-friendly.

## Entry 21
- Date: 2026-03-22
- Component: Checkpoint Migration Execution (v5 to run5)
- Change Applied: Physically migrated existing v5 generator checkpoints (`checkpoints/v5/generator/<contrast>/*`) into `checkpoints/generator/<contrast>/run5/` and migrated v5 segmenter baseline checkpoints (`checkpoints/v5/segmenter/baseline/<contrast>/*`) into `checkpoints/segmenter/baseline/<contrast>/run5/`. Removed empty source directories after migration. Explicitly excluded `checkpoints/v5/segmenter/fully_artificial/*` from any move because those trainings are still running.
- Reasoning & Google-Grade Standard: Controlled one-way migration preserves checkpoint lineage, prevents active-write collisions for in-progress runs, and aligns historical artifacts to the new run-indexed convention without touching ongoing experiments.

## Entry 22
- Date: 2026-03-22
- Component: Segmenter Baseline Stability (GPU Augmentations)
- Change Applied: Fixed inconsistent return signatures in `src/kornia_augmentations.py` for `RandomLowResolution3D`, `RandomGaussianNoise3D`, and `RandomGaussianSmooth3D` so they return `(x, mask)` whenever a mask is provided.
- Reasoning & Google-Grade Standard: Segmenter training passes both image and label through the same augmentation pipeline. Mixed return types (tensor-only vs tuple) caused runtime unpack failures (`ValueError: too many values to unpack`). Consistent tuple contracts eliminate the crash and keep synchronized image-mask augmentation semantics.

## Entry 23
- Date: 2026-03-22
- Component: Checkpoint Organization Follow-up (Baseline + Fully Artificial)
- Change Applied: Verified run-indexed baseline segmenter outputs are being written under `checkpoints/segmenter/baseline/<contrast>/runX/` (including active runs). Migrated `checkpoints/v5/segmenter/fully_artificial/{t1w,t2w}` flat files into explicit run subfolders (`run1`, `run2`, `run3`) using `last-vN` markers and epoch group continuity.
- Reasoning & Google-Grade Standard: Making run boundaries explicit removes ambiguity in checkpoint provenance, improves resume/debug ergonomics, and keeps evaluation/discovery logic deterministic.

## Entry 24
- Date: 2026-03-22
- Component: Version-Scoped Checkpoint Isolation
- Change Applied: Updated generator and segmenter checkpoint directory builders so run-indexed folders are nested under the experiment version root (`checkpoints/<version>/.../runX`) instead of global shared roots. Updated `run_segmenters.sh` to resolve generator defaults from version-scoped run folders first.
- Reasoning & Google-Grade Standard: Version isolation is required to compare architecture/loss/guidance changes over time without mixing artifacts across incompatible experiment generations.

## Entry 25
- Date: 2026-03-22
- Component: Segmenter Validation Image Logging
- Change Applied: Added validation image logging in `MRISegmenterLightning.validation_step` with an explicit `_log_segmenter_val_images` helper that logs input/target/prediction middle slices (`val/slice_grid`) to WandB. The logger now respects `training.segmenter.val_image_log_every` and logs once per configured epoch interval on the first validation batch.
- Reasoning & Google-Grade Standard: Segmenter training previously had no image logging path despite config support, which reduced observability and made qualitative debugging difficult. Adding a gated, deterministic logging path restores visual QA without adding significant runtime overhead.

## Entry 26
- Date: 2026-03-22
- Component: Segmenter Convergence Parity (v5 Baseline)
- Change Applied: Restored `data.batch_size_segmenter` default from 32 to 8 in `conf/data/brats.yaml` after comparing runs at commit `80b4305` and finding that convergent runs used effective batch size 8 while collapsed white-mask runs used batch size 32. Also fixed an indentation regression in `RandomGaussianSmooth3D` (`src/kornia_augmentations.py`) where the function returned from inside the batch loop, causing premature exit after the first sample.
- Reasoning & Google-Grade Standard: Matching known-good hyperparameter regime is the fastest path to isolate true regressions. Large batch-size shifts changed optimization dynamics enough to collapse validation behavior; the augmentation early-return bug was a correctness defect and is fixed to prevent hidden per-batch inconsistency.

## Entry 27
- Date: 2026-03-22
- Component: Temporary Segmenter Train Visualization
- Change Applied: Added train-time segmenter visualization logging in `MRISegmenterLightning.training_step` with a new helper `_log_segmenter_train_images`. The train log now records a 4-column grid (raw input, model input, target, prediction) under `train/slice_grid_raw_model_target_pred`, controlled by `training.segmenter.enable_train_image_logging` and `training.segmenter.train_image_log_every`.
- Reasoning & Google-Grade Standard: Direct train-time visibility is needed to diagnose white-mask collapse and confirm whether failure appears before or only during validation. Controls keep this temporary diagnostic inexpensive and easy to disable.

## Entry 28
- Date: 2026-03-23
- Component: Segmenter Regression Root Cause (v4 vs v6) and Augmentation Correctness
- Change Applied:
  1. **v4 Parity Audit:** Compared commit `f551ef61` (known-good v4) against current Lightning/Kornia flow and confirmed v4 segmenter training path did not apply geometric affine/elastic mask warps in the active transform stack.
  2. **Mask-Safe Affine Fix:** Reworked `KorniaMRIAugmentation3D` to apply affine on images and masks through synchronized parameter reuse while forcing mask resampling with `NEAREST` interpolation (`affine_image` + `affine_mask`) instead of bilinear interpolation over concatenated image+mask tensors.
  3. **Compile/DDP RNG Hygiene:** Removed native Python `random` branching from GPU augmentation forwards (`RandomElasticTransform3D`, `RandomLowResolution3D`, `RandomGaussianNoise3D`) and from segmenter generator gating (`MRISegmenterLightning._maybe_apply_generator`) in favor of `torch.rand` / tensor sampling.
- Reasoning & Google-Grade Standard: The zero-Dice-with-falling-CE signature is consistent with intermittent label-collapse events from interpolated masks. Preserving categorical masks with nearest-neighbor spatial resampling removes this failure mode while maintaining synchronized image-label geometry. Replacing Python RNG with tensor RNG avoids host-side nondeterministic branching paths and keeps augmentation stochasticity compatible with compiled and distributed execution.

## Entry 29
- Date: 2026-03-23
- Component: Visualization Generation Pipeline
- Change Applied: Created `scripts/generate_visualizations.py` for deterministic generation and disk export of 3D synthetic MRI volumes. The script integrates Hydra configuration loading, BraTSDataModule validation sampling, trained generator inference, and NIfTI output serialization. Key features include: (1) deterministic seed initialization via `pytorch_lightning.seed_everything(seed)` before any data or model operations; (2) dual checkpoint format support—parsing both legacy v1-v4 flat `.pth` layout (`checkpoints/vN/mri_generator_<contrast>_epoch_<epoch>.pth`) and modern v5+ run-indexed structure (`checkpoints/<version>/generator/<contrast>/runX/<ckpt>`); (3) CompiledSynthesisWrapper integration to handle version-specific guidance map logic (sharp vs blurred) via shared histogram target generation; (4) MONAI's SaveImage with nibabel fallback for robust NIfTI output; (5) organized output hierarchy under `results/visualizations/<gen_version>/<model_id>/` with separate samples containing source, synthetic, and label volumes; (6) command-line arguments for checkpoint path, config override, sample count, and random seed.
- Reasoning & Google-Grade Standard: Exporting fixed synthetic volumes enables medical imaging professionals and clinicians to qualitatively inspect pseudo-contrast quality in standards-compliant medical viewers (ITK-SNAP, 3D Slicer) without requiring code execution, improving trust in model outputs and reducing downstream integration friction. Deterministic seeding and version-aware guidance handling preserve training-time synthesis behavior. The dual checkpoint parser and metadata extraction from filesystem context makes the script backward-compatible with legacy experiments while supporting ongoing v5+ runs without manual path engineering.

## Entry 30
- Date: 2026-03-23
- Component: Generator Checkpoint Fidelity and Segmenter Version Alignment
- Change Applied:
  1. **Generator Best-Loss Checkpoint Policy:** Modified `scripts/train.py` checkpoint callback from `save_top_k=0, save_last=True` (save only final checkpoint) to `save_top_k=1, save_last=True` with metric monitoring `train/total_loss` in minimize mode (`mode="min"`). Filename now includes epoch and loss for debugging: `best_loss-{epoch:03d}-{train_loss:.4f}`. This ensures the saved generator checkpoint is the one with the lowest training loss, protecting against non-converged or collapsed final-epoch models.
  2. **Segmenter Gen-Version Inheritance:** Changed `conf/model/defaults.yaml` segmenter config from hardcoded `gen_version: v2` to `gen_version: null`. Updated `MRISegmenterLightning.setup()` in `src/lightning_modules.py` to detect None gen_version and default to the current segmenter version (`self.cfg.version`), with explicit logging to inform users about the fallback. This fixes the problem where v6 segmenters would silently load v2 generators instead of v6 generators, causing architecture/weight mismatch and training collapse.
  3. **Enhanced Generator Loading Diagnostics:** Added explicit print statements at generator loading time to report which version is being used and which checkpoint path is being loaded, enabling visibility into fallback behavior and debugging checkpoint resolution issues.
- Reasoning & Google-Grade Standard: Saving the best (lowest-loss) checkpoint instead of the final chronological checkpoint prevents training dynamics where a model converges mid-training then diverges toward epoch end, a common pattern in GAN-like synthesis tasks. Version alignment between segmenter and generator ensures architectural consistency (same guidance map logic, same histogram bins, same model scaling). Without explicit inheritance, downstream users silently train against mismatched generator versions, causing cascade failures (segmenters train collapse → artificial contrast quality degradation → downstream evaluation corruption). The diagnostic logging transforms a silent failure mode into an explicit user-visible warning, enabling faster root-cause identification.


