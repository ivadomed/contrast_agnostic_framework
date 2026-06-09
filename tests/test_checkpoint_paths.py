"""
tests/test_checkpoint_paths.py
Verify checkpoint path-building and discovery logic without GPU or training.
Uses tmp_path + monkeypatching PROJECT_ROOT — no actual training occurs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock
import importlib
import importlib.util
import pytest
from omegaconf import OmegaConf

# ── Mock heavy deps before any script import ───────────────────────────────────
_HEAVY = [
    "pytorch_lightning",
    "pytorch_lightning.callbacks",
    "pytorch_lightning.loggers",
    "hydra",
    "hydra.core",
    "hydra.core.hydra_config",
    "monai",
    "monai.apps",
    "monai.data",
    "monai.inferers",
    "monai.metrics",
    "monai.networks",
    "monai.networks.nets",
    "monai.transforms",
    "src.training.datamodule",
    "src.training.lightning_modules",
]
for _m in _HEAVY:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_script(name: str) -> ModuleType:
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_train_gen = _load_script("train_generator")
_train_seg = _load_script("train_segmenter")
_evaluate = _load_script("evaluate")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _gen_cfg(
    dataset: str = "brats2017",
    contrast: str = "t1w",
    gen_version: str = "v19",
    resume: bool = False,
) -> OmegaConf:
    return OmegaConf.create({
        "data": {"name": dataset, "source_contrast": contrast},
        "model": {"generator": {"gen_version": gen_version}},
        "training": {
            "resume": resume,
            "checkpoint": {"dirpath_generator": None},
            "generator": {"gradient_clip_val": 1.0},
            "max_epochs": {"generator": 100},
        },
        "logging": {"run_name": None},
    })


def _seg_cfg(
    dataset: str = "brats2017",
    contrast: str = "t1w",
    out_channels: int = 4,
    use_generator: bool = False,
    gen_version: str | None = None,
    fully_artificial: bool = False,
    pretrained_ckpt_path: str | None = None,
    freeze_encoder: bool = False,
    resume: bool = False,
) -> OmegaConf:
    return OmegaConf.create({
        "data": {"name": dataset, "source_contrast": contrast},
        "model": {
            "segmenter": {
                "out_channels": out_channels,
                "use_generator": use_generator,
                "gen_version": gen_version,
                "fully_artificial": fully_artificial,
                "pretrained_ckpt_path": pretrained_ckpt_path,
                "freeze_encoder": freeze_encoder,
                "tf32": False,
            }
        },
        "training": {
            "resume": resume,
            "checkpoint": {
                "dirpath_segmenter": None,
                "filename_segmenter": "best",
                "monitor_segmenter": "val/dice",
                "mode_segmenter": "max",
                "save_top_k": 1,
            },
            "max_epochs": {"segmenter": 50},
        },
        "logging": {"run_name": None},
    })


# ── Generator path construction ────────────────────────────────────────────────

def test_generator_path_brats(tmp_path, monkeypatch):
    monkeypatch.setattr(_train_gen, "PROJECT_ROOT", tmp_path)
    cfg = _gen_cfg(dataset="brats2017", contrast="t1w", gen_version="v19")
    result = _train_gen._build_checkpoint_dir(cfg)
    expected = tmp_path / "checkpoints" / "brats2017" / "generator" / "v19" / "t1w" / "run1"
    assert result == expected


def test_generator_path_spider_spine(tmp_path, monkeypatch):
    monkeypatch.setattr(_train_gen, "PROJECT_ROOT", tmp_path)
    cfg = _gen_cfg(dataset="spider_spine", contrast="t1_sag", gen_version="v19")
    result = _train_gen._build_checkpoint_dir(cfg)
    expected = tmp_path / "checkpoints" / "spider_spine" / "generator" / "v19" / "t1_sag" / "run1"
    assert result == expected


# ── Segmenter path construction ────────────────────────────────────────────────

def test_segmenter_baseline_multiclass(tmp_path, monkeypatch):
    monkeypatch.setattr(_train_seg, "PROJECT_ROOT", tmp_path)
    cfg = _seg_cfg(out_channels=4)
    result = _train_seg._build_checkpoint_dir(cfg)
    parts = result.parts
    assert "seg_A" in parts
    assert "baseline" in parts
    assert "multiclass" in parts
    assert "t1w" in parts
    assert result.name == "run1"


def test_segmenter_baseline_merged_classes(tmp_path, monkeypatch):
    monkeypatch.setattr(_train_seg, "PROJECT_ROOT", tmp_path)
    cfg = _seg_cfg(out_channels=1)
    result = _train_seg._build_checkpoint_dir(cfg)
    parts = result.parts
    assert "seg_A" in parts
    assert "baseline" in parts
    assert "merged_classes" in parts
    assert result.name == "run1"


def test_segmenter_gen_multiclass(tmp_path, monkeypatch):
    monkeypatch.setattr(_train_seg, "PROJECT_ROOT", tmp_path)
    cfg = _seg_cfg(out_channels=4, use_generator=True, gen_version="v19")
    result = _train_seg._build_checkpoint_dir(cfg)
    parts = result.parts
    assert "seg_A" in parts
    assert "gen_v19" in parts
    assert "multiclass" in parts
    assert result.name == "run1"


def test_segmenter_finetuned_path(tmp_path, monkeypatch):
    monkeypatch.setattr(_train_seg, "PROJECT_ROOT", tmp_path)
    pretrained = str(tmp_path / "checkpoints" / "brats2017" / "segmenter" / "seg_A" / "baseline" / "multiclass" / "t1w" / "run1" / "last.ckpt")
    cfg = _seg_cfg(out_channels=4, pretrained_ckpt_path=pretrained, freeze_encoder=False)
    result = _train_seg._build_checkpoint_dir(cfg)
    parts = result.parts
    assert "seg_A" in parts
    assert "multiclass" in parts
    assert any("finetuned_from-baseline_t1w_freeze-false" == p for p in parts)


# ── Resume logic ───────────────────────────────────────────────────────────────

def test_resume_selects_latest_run(tmp_path, monkeypatch):
    monkeypatch.setattr(_train_gen, "PROJECT_ROOT", tmp_path)
    cfg = _gen_cfg(dataset="brats2017", contrast="t1w", gen_version="v19", resume=True)
    base = tmp_path / "checkpoints" / "brats2017" / "generator" / "v19" / "t1w"
    for n in (1, 2, 3):
        run = base / f"run{n}"
        run.mkdir(parents=True)
        (run / "last.ckpt").touch()
    result = _train_gen._build_checkpoint_dir(cfg)
    assert result == base / "run3"


def test_resume_no_existing_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(_train_gen, "PROJECT_ROOT", tmp_path)
    cfg = _gen_cfg(dataset="brats2017", contrast="t1w", gen_version="v19", resume=True)
    result = _train_gen._build_checkpoint_dir(cfg)
    assert result == tmp_path / "checkpoints" / "brats2017" / "generator" / "v19" / "t1w" / "run1"


# ── Checkpoint discovery (evaluate.py) ─────────────────────────────────────────

def test_generator_discovery_canonical(tmp_path, monkeypatch):
    monkeypatch.setattr(_evaluate, "PROJECT_ROOT", tmp_path)
    ckpt = tmp_path / "checkpoints" / "brats2017" / "generator" / "v19" / "t1w" / "run1" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    result = _evaluate._resolve_latest_generator_checkpoint(
        source_contrast="t1w",
        runtime_info={"dataset_name": "brats2017", "gen_version": "v19"},
    )
    assert result == ckpt


def test_generator_discovery_legacy_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(_evaluate, "PROJECT_ROOT", tmp_path)
    ckpt = tmp_path / "checkpoints" / "v19" / "generator" / "t1w" / "run1" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    result = _evaluate._resolve_latest_generator_checkpoint(
        source_contrast="t1w",
        runtime_info={"gen_version": "v19"},
    )
    assert result == ckpt


def test_segmenter_discovery_canonical(tmp_path, monkeypatch):
    monkeypatch.setattr(_evaluate, "PROJECT_ROOT", tmp_path)
    ckpt = (
        tmp_path / "checkpoints" / "brats2017" / "segmenter" / "seg_A"
        / "gen_v19" / "multiclass" / "t1w" / "run1" / "last.ckpt"
    )
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    specs = _evaluate._discover_models(tmp_path / "checkpoints" / "brats2017")
    assert len(specs) == 1
    assert specs[0]["source_contrast"] == "t1w"
    assert specs[0]["family"] == "gen_v19"
    assert Path(specs[0]["checkpoint_path"]) == ckpt
