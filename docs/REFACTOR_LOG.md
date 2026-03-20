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
