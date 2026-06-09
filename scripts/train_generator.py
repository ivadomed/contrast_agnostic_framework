from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import re

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import torch.multiprocessing as mp

mp.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision("high")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.datamodule import BraTSDataModule
from src.training.lightning_modules import MRISynthesisLightning


def _resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _get_git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _list_run_dirs(base_dir: Path) -> list[tuple[int, Path]]:
    if not base_dir.exists():
        return []
    run_pattern = re.compile(r"^run(\d+)$")
    run_dirs = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        match = run_pattern.match(child.name)
        if match:
            run_dirs.append((int(match.group(1)), child))
    return run_dirs


def _build_run_name(cfg: DictConfig) -> str:
    if cfg.logging.run_name is not None:
        return str(cfg.logging.run_name)
    dataset_name = str(getattr(cfg.data, "name", "dataset"))
    gen_version = cfg.model.generator.gen_version
    return f"{dataset_name}_generator_{gen_version}_{cfg.data.source_contrast}"


def _build_checkpoint_dir(cfg: DictConfig) -> Path:
    configured_dir = cfg.training.checkpoint.dirpath_generator
    if configured_dir is not None:
        return _resolve_path(str(configured_dir))

    dataset_name = str(getattr(cfg.data, "name", "unknown_dataset"))
    gen_version = cfg.model.generator.gen_version
    base_dir = (
        PROJECT_ROOT
        / "checkpoints"
        / dataset_name
        / "generator"
        / str(gen_version)
        / str(cfg.data.source_contrast)
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = _list_run_dirs(base_dir)

    if bool(cfg.training.resume) and run_dirs:
        return max(run_dirs, key=lambda item: item[0])[1]

    next_idx = max((idx for idx, _ in run_dirs), default=0) + 1
    return base_dir / f"run{next_idx}"


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(int(cfg.seed), workers=True)

    datamodule = BraTSDataModule(cfg)
    model = MRISynthesisLightning(cfg)

    wandb_logger = WandbLogger(
        project=str(cfg.logging.project_name_generator),
        name=_build_run_name(cfg),
        save_dir=str(_resolve_path(str(cfg.logging.save_dir))),
        log_model=bool(cfg.logging.log_model),
        offline=bool(cfg.logging.offline),
    )

    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    try:
        experiment = wandb_logger.experiment
        if hasattr(experiment, "config") and hasattr(experiment.config, "update"):
            experiment.config.update(cfg_container, allow_val_change=True)
            experiment.config.update(
                {"git_commit_hash": _get_git_commit_hash()},
                allow_val_change=True,
            )
        else:
            wandb_logger.log_hyperparams(
                {
                    "git_commit_hash": _get_git_commit_hash(),
                    "hydra_config": cfg_container,
                }
            )
    except Exception:
        wandb_logger.log_hyperparams(
            {
                "git_commit_hash": _get_git_commit_hash(),
                "hydra_config": cfg_container,
            }
        )

    ckpt_dir = _build_checkpoint_dir(cfg)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best_loss-{epoch:03d}-{train_loss:.4f}",
        monitor="train/total_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    lr_callback = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.max_epochs.generator),
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_callback],
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        num_nodes=int(cfg.training.num_nodes),
        strategy=cfg.training.strategy,
        precision=str(cfg.training.precision),
        deterministic=bool(cfg.training.deterministic),
        benchmark=bool(cfg.training.benchmark) and (not bool(cfg.training.deterministic)),
        num_sanity_val_steps=int(cfg.training.num_sanity_val_steps),
        log_every_n_steps=int(cfg.training.log_every_n_steps),
        enable_model_summary=bool(cfg.training.enable_model_summary),
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=0,
        gradient_clip_val=float(cfg.training.generator.gradient_clip_val),
    )

    resume_last = ckpt_dir / "last.ckpt"
    ckpt_path = "last" if bool(cfg.training.resume) and resume_last.exists() else None

    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    trainer.fit(model=model, train_dataloaders=train_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
