#!/usr/bin/env python3
"""Quick 2-epoch benchmark to identify where time is spent."""
import time
import torch
import sys
from hydra import initialize_config_dir, compose
from pathlib import Path

def benchmark_training(max_epochs=2, config_overrides=None):
    if config_overrides is None:
        config_overrides = []
    
    config_dir = str(Path(__file__).parent / "conf")
    
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "training.max_epochs.generator=2",
                "training.limit_train_batches=1.0",
                "training.limit_val_batches=1.0",
                "training.log_every_n_steps=1",
                "training.num_sanity_val_steps=0",
                "data.cache_rate=0.0",
                "training.generator.enable_image_logging=false",
            ] + config_overrides
        )
    
    from pytorch_lightning import Trainer
    from src.lightning_modules import MRISynthesisLightning
    from src.datamodule import BraTSDataModule
    
    dm = BraTSDataModule(cfg)
    dm.setup("fit")
    
    model = MRISynthesisLightning(cfg)
    
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        deterministic=cfg.training.deterministic,
        benchmark=cfg.training.benchmark,
        enable_model_summary=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {config_overrides if config_overrides else 'BASELINE'}")
    print(f"{'='*60}")
    
    start = time.time()
    trainer.fit(model, dm)
    total = time.time() - start
    
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {total:.1f}s ({total/2:.1f}s per epoch)")
    print(f"{'='*60}\n")
    
    return total / 2

if __name__ == "__main__":
    benchmark_training(config_overrides=["training.generator.gpu_aug.enabled=true"])
