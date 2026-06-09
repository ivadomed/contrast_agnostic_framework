from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import re

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
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
from src.training.lightning_modules import MRISegmenterLightning, MRISynthesisLightning


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


def _get_model_id(cfg: DictConfig) -> str:
    """Return the run identifier in <dataset>_<segmenter>_<generator> form.

    Reads Hydra config-group choices at runtime so new-style invocations
    (data=brats2017 segmenter=seg_A generator=gen_19) produce
    'brats2017_seg_A_gen_19'. Falls back to cfg.version for legacy scripts.
    """

    dataset_name = str(getattr(cfg.data, "name", "unknown_dataset"))
    task = str(getattr(cfg, "task", "generator"))
    try:
        choices = HydraConfig.get().runtime.choices
        seg_id = getattr(getattr(cfg, "segmenter", None), "name", None) or choices.get("segmenter", None)
        gen_id = getattr(getattr(cfg, "generator", None), "name", None) or choices.get("generator", None)
        if task == "generator" and gen_id:
            return f"{dataset_name}_{gen_id}"
        if seg_id and gen_id:
            return f"{dataset_name}_{seg_id}_{gen_id}"
    except Exception:
        pass
    version = getattr(cfg, "version", None)
    if version is not None and str(version) not in ("None", "null"):
        return str(version)
    return f"{dataset_name}_unknown_segmenter_unknown_generator"


def _get_segmenter_choice_name(cfg: DictConfig) -> str:
    try:
        choices = HydraConfig.get().runtime.choices
        seg_id = choices.get("segmenter", None)
        if seg_id:
            return str(seg_id)
    except Exception:
        pass
    return str(getattr(getattr(cfg, "segmenter", None), "name", "segmenter"))


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


def _legacy_checkpoint_bases(cfg: DictConfig, model_id: str, task: str) -> list[Path]:
    bases: list[Path] = []
    source_contrast = str(cfg.data.source_contrast)
    if task == "generator":
        gen_version = cfg.model.generator.gen_version
        if gen_version is None:
            gen_version = model_id
        bases.append(PROJECT_ROOT / "checkpoints" / str(gen_version) / "generator" / source_contrast)
        bases.append(PROJECT_ROOT / "checkpoints" / "generator" / source_contrast)
        return bases

    if bool(cfg.model.segmenter.fully_artificial):
        mode = "fully_artificial"
    elif bool(cfg.model.segmenter.use_generator):
        mode = "generator"
    else:
        mode = "baseline"

    bases.append(PROJECT_ROOT / "checkpoints" / model_id / "segmenter" / mode / source_contrast)
    return bases


def _build_run_name(cfg: DictConfig) -> str:
    if cfg.logging.run_name is not None:
        return str(cfg.logging.run_name)

    model_id = _get_model_id(cfg)
    task = str(cfg.task)
    dataset_name = str(getattr(cfg.data, "name", "dataset"))

    if task == "generator":
        gen_version = cfg.model.generator.gen_version
        if gen_version is None:
            gen_version = model_id
        return f"generator-{dataset_name}-{gen_version}-{cfg.data.source_contrast}"

    if bool(cfg.model.segmenter.fully_artificial):
        return f"fully-artificial-{model_id}-{cfg.data.source_contrast}-segmenter"
    if bool(cfg.model.segmenter.use_generator):
        gen_version = cfg.model.segmenter.gen_version
        if gen_version is None:
            gen_version = "unknown"
        seg_name = _get_segmenter_choice_name(cfg)
        return f"segmenter-{dataset_name}-{seg_name}-generated-{gen_version}-{cfg.data.source_contrast}"
    seg_name = _get_segmenter_choice_name(cfg)
    return f"baseline-{dataset_name}-{seg_name}-{cfg.data.source_contrast}-segmenter"


def _build_checkpoint_dir(cfg: DictConfig) -> Path:
    model_id = _get_model_id(cfg)
    dataset_name = str(getattr(cfg.data, "name", "unknown_dataset"))
    task = str(cfg.task)

    if task == "generator":
        configured_dir = cfg.training.checkpoint.dirpath_generator
        if configured_dir is not None:
            return _resolve_path(str(configured_dir))

        gen_version = cfg.model.generator.gen_version
        if gen_version is None:
            gen_version = model_id
        base_dir = (
            PROJECT_ROOT
            / "checkpoints"
            / dataset_name
            / model_id
            / "generator"
            / str(cfg.data.source_contrast)
        )
    else:
        configured_dir = cfg.training.checkpoint.dirpath_segmenter
        if configured_dir is not None:
            return _resolve_path(str(configured_dir))

        if bool(cfg.model.segmenter.fully_artificial):
            mode = "fully_artificial"
        elif bool(cfg.model.segmenter.use_generator):
            mode = "generator"
        else:
            mode = "baseline"

        seg_name = _get_segmenter_choice_name(cfg)
        gen_version = cfg.model.segmenter.gen_version
        if gen_version is None:
            gen_version = "none"
        run_family = seg_name if mode == "baseline" else f"{seg_name}_gen_{gen_version}"

        base_dir = (
            PROJECT_ROOT
            / "checkpoints"
            / dataset_name
            / run_family
            / "segmenter"
            / mode
            / str(cfg.data.source_contrast)
        )

    base_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = _list_run_dirs(base_dir)

    if bool(cfg.training.resume) and not run_dirs:
        for legacy_base in _legacy_checkpoint_bases(cfg, model_id=model_id, task=task):
            legacy_runs = _list_run_dirs(legacy_base)
            if legacy_runs:
                return max(legacy_runs, key=lambda item: item[0])[1]
            legacy_last = legacy_base / "last.ckpt"
            if legacy_last.exists():
                return legacy_base

    # Resume into latest run directory; otherwise create next run directory.
    if bool(cfg.training.resume) and run_dirs:
        return max(run_dirs, key=lambda item: item[0])[1]

    next_idx = (max((idx for idx, _ in run_dirs), default=0) + 1)
    return base_dir / f"run{next_idx}"


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.task not in ("generator", "segmenter"):
        raise ValueError(f"Unsupported task '{cfg.task}'. Expected 'generator' or 'segmenter'.")

    if str(cfg.task) == "segmenter":
        if cfg.model.segmenter.gen_version is None and hasattr(cfg.model, "generator") and hasattr(cfg.model.generator, "gen_version"):
            if cfg.model.generator.gen_version is not None:
                cfg.model.segmenter.gen_version = cfg.model.generator.gen_version
        if cfg.model.segmenter.gen_version is not None and not bool(cfg.model.segmenter.use_generator):
            cfg.model.segmenter.use_generator = True

    pl.seed_everything(int(cfg.seed), workers=True)

    datamodule = BraTSDataModule(cfg)
    if str(cfg.task) == "generator":
        model = MRISynthesisLightning(cfg)
        project_name = str(cfg.logging.project_name_generator)
    else:
        model = MRISegmenterLightning(cfg)
        project_name = str(cfg.logging.project_name_segmenter)

    wandb_logger = WandbLogger(
        project=project_name,
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

    if str(cfg.task) == "generator":
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(_build_checkpoint_dir(cfg)),
            filename="best_loss-{epoch:03d}-{train_loss:.4f}",
            monitor="train/total_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        max_epochs = int(cfg.training.max_epochs.generator)
        limit_val_batches = 0
        gradient_clip_val = float(cfg.training.generator.gradient_clip_val)
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(_build_checkpoint_dir(cfg)),
            filename=str(cfg.training.checkpoint.filename_segmenter),
            monitor=str(cfg.training.checkpoint.monitor_segmenter),
            mode=str(cfg.training.checkpoint.mode_segmenter),
            save_top_k=int(cfg.training.checkpoint.save_top_k),
            save_last=True,
            auto_insert_metric_name=False,
        )
        max_epochs = int(cfg.training.max_epochs.segmenter)
        limit_val_batches = cfg.training.limit_val_batches
        gradient_clip_val = 0.0

    lr_callback = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
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
        limit_val_batches=limit_val_batches,
        gradient_clip_val=gradient_clip_val,
    )

    ckpt_dir = _build_checkpoint_dir(cfg)
    resume_last = ckpt_dir / "last.ckpt"
    ckpt_path = "last" if bool(cfg.training.resume) and resume_last.exists() else None

    if str(cfg.task) == "generator":
        datamodule.setup("fit")
        train_loader = datamodule.train_dataloader()
        trainer.fit(model=model, train_dataloaders=train_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
