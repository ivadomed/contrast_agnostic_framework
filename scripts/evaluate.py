from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    Spacingd,
)
from torch.utils.data import Subset
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import DEFAULT_CONTRASTS, build_contrast_to_index, normalize_contrast_name
from src.generator import MRI_Synthesis_Net
from src.histogram_ops import DifferentiableHistogram3D, generate_unified_targets

try:
    mp.set_sharing_strategy("file_system")
except (RuntimeError, ValueError):
    # Strategy can be preconfigured by the runtime; keep the current one if setting fails.
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiple segmenters across all MRI contrasts.")
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument(
        "--models-file",
        type=str,
        default=None,
        help="CSV file with columns: model_id,family,source_contrast,checkpoint_path[,enabled].",
    )
    parser.add_argument(
        "--discover-checkpoints",
        type=str,
        default=None,
        help="Directory containing checkpoints named like best_segmenter_<family>_<contrast>.pth.",
    )
    parser.add_argument(
        "--skip-baseline-auto",
        action="store_true",
        help="Skip automatic inclusion of checkpoints/baseline.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Inline model spec: model_id,family,source_contrast,checkpoint_path",
    )
    parser.add_argument(
        "--num-ensemble",
        type=int,
        default=1,
        help=(
            "Total number of models to use in the ensemble, including the best checkpoint. "
            "Use 1 for single-model inference, or 1..5 for standard ensemble sweeps."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "eval"),
        help="Directory for eval_long.csv, eval_wide.csv, and eval_summary.md.",
    )
    parser.add_argument(
        "--task-mode",
        type=str,
        choices=["auto", "binary", "multiclass"],
        default="auto",
        help="Evaluation mode routing. 'auto' infers from loaded segmenter output channels.",
    )
    parser.add_argument(
        "--tta-samples",
        type=int,
        default=0,
        help="Number of synthetic TTA passes to average per batch.",
    )
    parser.add_argument(
        "--exclude-original",
        action="store_true",
        help="Evaluate using synthetic TTA passes only, without the original contrast pass.",
    )

    # Legacy fallback flags (kept for backward compatibility).
    parser.add_argument(
        "--baseline-ckpt",
        type=str,
        default=None,
    )
    parser.add_argument("--baseline-contrast", type=str, default="t1w")
    parser.add_argument(
        "--generator-ckpt",
        type=str,
        default=None,
    )
    parser.add_argument("--generator-contrast", type=str, default="t1w")
    parser.add_argument(
        "--bigaug-ckpt",
        type=str,
        default=None,
    )
    parser.add_argument("--bigaug-contrast", type=str, default="t1w")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--disable-pin-memory",
        action="store_true",
        help="Disable DataLoader pin_memory to avoid host IPC/pin thread crashes on some systems.",
    )
    parser.add_argument(
        "--disable-persistent-workers",
        action="store_true",
        help="Disable DataLoader persistent workers for safer worker lifecycle handling.",
    )
    parser.add_argument("--cache-rate", type=float, default=0.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Evaluation batch size (number of subjects). Increase cautiously to avoid GPU OOM.",
    )
    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=16,
        help="Sliding-window internal batch size. Increase for speed if memory allows.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable mixed precision during evaluation (AMP is enabled by default on CUDA).",
    )
    parser.add_argument(
        "--min-sw-batch-size",
        type=int,
        default=4,
        help="Lower bound for adaptive sliding-window batch size fallback on CUDA OOM.",
    )
    parser.add_argument("--split-file", type=str, default=str(PROJECT_ROOT / "splits" / "brats_subject_split.json"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _default_checkpoint_path(tag: str, contrast: str) -> str:
    if tag == "baseline":
        return str(PROJECT_ROOT / "checkpoints" / "baseline" / f"baseline_{contrast}" / "best_segmenter.pth")
    return str(PROJECT_ROOT / "checkpoints" / f"best_segmenter_{tag}_{contrast}.pth")


def _extract_contrast(image: torch.Tensor, contrast: str, contrast_to_index: dict[str, int]) -> torch.Tensor:
    index = contrast_to_index[normalize_contrast_name(contrast, list(contrast_to_index.keys()))]
    return image[:, index : index + 1, ...]


def _stack_target_contrasts(
    image: torch.Tensor,
    target_contrasts: list[str],
    contrast_to_index: dict[str, int],
) -> tuple[torch.Tensor, int]:
    # Convert [B, C, ...] multi-contrast input to [B*N, 1, ...] in target contrast order.
    batch_size = image.shape[0]
    stacked = torch.cat(
        [_extract_contrast(image, contrast, contrast_to_index) for contrast in target_contrasts],
        dim=0,
    )
    return stacked, batch_size


def _load_hyper_parameters(checkpoint_path: str) -> dict[str, Any] | None:
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(checkpoint, dict):
        return None
    hyper = checkpoint.get("hyper_parameters") or checkpoint.get("hparams")
    return hyper if isinstance(hyper, dict) else None


def _normalize_checkpoint_state_dict(checkpoint: object) -> OrderedDict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format: expected a dict-like state dict.")

    normalized_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        normalized_key = key
        hit = True
        while hit:
            hit = False
            for prefix in ["_orig_mod.", "module.", "model.", "compiled_wrapper.", "compiled_synthesis.", "compiled_segmenter.", "segmenter."]:
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
                    hit = True
        normalized_state_dict[normalized_key] = value

    return normalized_state_dict


def _segmenter_runtime_info(checkpoint_path: str) -> dict[str, Any]:
    hyper = _load_hyper_parameters(checkpoint_path)
    source_contrast = _nested_get(hyper, ["data", "source_contrast"], default=None)
    version = _nested_get(hyper, ["version"], default=None)
    gen_version = _nested_get(hyper, ["model", "segmenter", "gen_version"], default=None)
    if gen_version is None:
        gen_version = _nested_get(hyper, ["model", "generator", "gen_version"], default=version)

    num_bins = _nested_get(hyper, ["model", "segmenter", "num_bins"], default=_nested_get(hyper, ["model", "generator", "num_bins"], default=128))
    num_chunks = _nested_get(hyper, ["model", "segmenter", "num_chunks"], default=_nested_get(hyper, ["model", "generator", "num_chunks"], default=8))
    dark_threshold = _nested_get(hyper, ["model", "segmenter", "dark_threshold"], default=_nested_get(hyper, ["model", "generator", "dark_threshold"], default=0.05))

    return {
        "source_contrast": normalize_contrast_name(str(source_contrast)) if source_contrast is not None else None,
        "version": str(version) if version is not None else None,
        "gen_version": str(gen_version) if gen_version is not None else None,
        "num_bins": int(num_bins),
        "num_chunks": int(num_chunks),
        "dark_threshold": float(dark_threshold),
        "generator_in_channels": int(_nested_get(hyper, ["model", "generator", "in_channels"], default=2)),
        "generator_out_channels": int(_nested_get(hyper, ["model", "generator", "out_channels"], default=1)),
        "generator_base_filters": int(_nested_get(hyper, ["model", "generator", "base_filters"], default=32)),
        "generator_channels_last_3d": bool(_nested_get(hyper, ["model", "generator", "channels_last_3d"], default=True)),
    }


def _build_eval_mode_name(task_mode: str, tta_samples: int, exclude_original: bool) -> str:
    mode_name = "multiclass" if task_mode == "multiclass" else "binary"
    if tta_samples > 0:
        mode_name = f"{mode_name}_tta_n{tta_samples}"
        if exclude_original:
            mode_name += "_synth_only"
    elif exclude_original:
        raise ValueError("--exclude-original requires --tta-samples > 0.")
    return mode_name


def _resolve_latest_generator_checkpoint(source_contrast: str) -> Path:
    generator_root = PROJECT_ROOT / "checkpoints" / "generator" / normalize_contrast_name(source_contrast)
    if not generator_root.exists():
        raise FileNotFoundError(f"Generator checkpoint directory not found: {generator_root}")

    run_pattern = re.compile(r"^run(\d+)$")
    run_dirs: list[tuple[int, Path]] = []
    for child in generator_root.iterdir():
        if not child.is_dir():
            continue
        match = run_pattern.match(child.name)
        if match is not None:
            run_dirs.append((int(match.group(1)), child))

    for _, run_dir in sorted(run_dirs, key=lambda item: item[0], reverse=True):
        candidate = run_dir / "last.ckpt"
        if candidate.exists():
            return candidate

    legacy_candidate = generator_root / "last.ckpt"
    if legacy_candidate.exists():
        return legacy_candidate

    raise FileNotFoundError(f"No generator checkpoint found under: {generator_root}")


def _load_generator_model(device: torch.device, checkpoint_path: Path, runtime_info: dict[str, Any]) -> MRI_Synthesis_Net:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _normalize_checkpoint_state_dict(checkpoint)

    generator = MRI_Synthesis_Net(
        in_channels=int(runtime_info["generator_in_channels"]),
        out_channels=int(runtime_info["generator_out_channels"]),
        base_filters=int(runtime_info["generator_base_filters"]),
    )
    if runtime_info["generator_channels_last_3d"]:
        generator = generator.to(memory_format=torch.channels_last_3d)
    generator.load_state_dict(state_dict, strict=True)
    generator = generator.to(device)
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad = False
    return generator


def _predict_probs_from_ensemble(models: list[UNet], inferer: SlidingWindowInferer, x_input: torch.Tensor) -> torch.Tensor:
    probabilities: list[torch.Tensor] = []
    for model in models:
        logits = _infer_logits_with_oom_retry(inferer=inferer, x_input=x_input, model=model)
        if logits.shape[1] == 1:
            probabilities.append(torch.sigmoid(logits))
        else:
            probabilities.append(F.softmax(logits, dim=1))

    return torch.mean(torch.stack(probabilities, dim=0), dim=0)


def _generate_synth_input(
    generator: MRI_Synthesis_Net,
    hist_module: DifferentiableHistogram3D,
    x_input: torch.Tensor,
    runtime_info: dict[str, Any],
) -> torch.Tensor:
    _, _, guidance_map = generate_unified_targets(
        input_images=x_input,
        num_bins=int(runtime_info["num_bins"]),
        num_chunks=int(runtime_info["num_chunks"]),
        dark_threshold=float(runtime_info["dark_threshold"]),
        hist_module=hist_module,
        gen_version=str(runtime_info["gen_version"] or runtime_info["version"] or ""),
    )

    generator_input = torch.cat([x_input, guidance_map], dim=1)
    synthesized = generator(generator_input)
    return ((synthesized + 1.0) * 0.5).clamp(0.0, 1.0)


_GLOBAL_MIN_SW_BATCH_SIZE = 1


def _is_cuda_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _infer_logits_with_oom_retry(inferer: SlidingWindowInferer, x_input: torch.Tensor, model: UNet) -> torch.Tensor:
    min_sw = max(1, int(_GLOBAL_MIN_SW_BATCH_SIZE))
    while True:
        try:
            return inferer(x_input, model)
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise

            current_sw = int(getattr(inferer, "sw_batch_size", 1))
            if current_sw <= min_sw:
                raise

            next_sw = max(min_sw, current_sw // 2)
            if next_sw >= current_sw:
                raise

            print(f"CUDA OOM in sliding-window inference. Reducing sw_batch_size: {current_sw} -> {next_sw}")
            setattr(inferer, "sw_batch_size", next_sw)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _contrasts_from_hyper_parameters(hyper: dict[str, Any] | None) -> list[str] | None:
    if hyper is None:
        return None
    contrasts = _nested_get(hyper, ["data", "contrasts"], default=None)
    if not isinstance(contrasts, (list, tuple)) or len(contrasts) == 0:
        return None
    return [str(c).strip().lower() for c in contrasts]


def _load_contrasts_from_default_config() -> list[str]:
    cfg_path = PROJECT_ROOT / "conf" / "data" / "brats.yaml"
    if cfg_path.exists():
        try:
            cfg = OmegaConf.load(cfg_path)
            contrasts = cfg.get("contrasts")
            if isinstance(contrasts, (list, tuple)) and len(contrasts) > 0:
                return [str(c).strip().lower() for c in contrasts]
        except Exception:
            pass
    return [str(c) for c in DEFAULT_CONTRASTS]


def _resolve_target_contrasts(model_specs: list[dict[str, Any]]) -> list[str]:
    for spec in model_specs:
        hyper = _load_hyper_parameters(spec["checkpoint_path"])
        contrasts = _contrasts_from_hyper_parameters(hyper)
        if contrasts:
            return contrasts
    return _load_contrasts_from_default_config()


def _load_label_mapping_from_default_config() -> dict[int, int]:
    cfg_path = PROJECT_ROOT / "conf" / "data" / "brats.yaml"
    if cfg_path.exists():
        try:
            cfg = OmegaConf.load(cfg_path)
            mapping = cfg.get("label_mapping")
            if isinstance(mapping, dict):
                return {int(k): int(v) for k, v in mapping.items()}
        except Exception:
            pass
    return {}


def _format_metric(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    return f"{float(value):.4f}"


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _subject_id_from_sample(sample: dict) -> str:
    image_path = sample["image"]
    if isinstance(image_path, (list, tuple)):
        image_path = image_path[0]
    image_name = Path(str(image_path)).name
    if image_name.endswith(".nii.gz"):
        return image_name[:-7]
    return Path(image_name).stem


def build_eval_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            EnsureTyped(keys=["image", "label"], data_type="tensor"),
        ]
    )


def _build_unet(device: torch.device, out_channels: int) -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=out_channels,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)


def build_segmenter(device: torch.device, weights_path: str) -> tuple[UNet, int]:
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    normalized_state_dict = OrderedDict()
    for key, value in state_dict.items():
        # Mixed Lightning checkpoints may contain generator weights; ignore them for UNet loading.
        if "generator." in key and "segmenter." not in key:
            continue

        normalized_key = key
        # Iteratively strip prefixes added by PyTorch Lightning, wrapping, or compiling
        hit = True
        while hit:
            hit = False
            for prefix in ["_orig_mod.", "module.", "compiled_wrapper.", "compiled_synthesis.", "compiled_segmenter.", "segmenter."]:
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
                    hit = True

        if normalized_key.startswith("model.model."):
            normalized_key = "model." + normalized_key[len("model.model.") :]
        normalized_state_dict[normalized_key] = value

    last_error: RuntimeError | None = None
    # If the checkpoint contains hyperparameters, prefer its declared out_channels
    declared_out: int | None = None
    if isinstance(checkpoint, dict):
        hyper = checkpoint.get("hyper_parameters") or checkpoint.get("hparams") or None
        if isinstance(hyper, dict):
            try:
                declared = hyper
                for key in ["model", "segmenter", "out_channels"]:
                    declared = declared.get(key) if isinstance(declared, dict) else None
                if declared is not None:
                    declared_out = int(declared)
            except Exception:
                declared_out = None

    probe_order = []
    if declared_out is not None:
        # Ensure we probe declared first, then fall back to common options
        probe_order = [declared_out] + [c for c in (4, 1) if c != declared_out]
    else:
        probe_order = [4, 1]

    for out_channels in probe_order:
        model = _build_unet(device=device, out_channels=out_channels)
        try:
            model.load_state_dict(normalized_state_dict, strict=True)
            model.eval()
            return model, out_channels
        except RuntimeError as err:
            last_error = err

        try:
            prefixed_state_dict = OrderedDict((f"model.{k}", v) for k, v in normalized_state_dict.items())
            model.load_state_dict(prefixed_state_dict, strict=True)
            model.eval()
            return model, out_channels
        except RuntimeError as err:
            last_error = err

    if last_error is None:
        raise RuntimeError(f"Failed to load checkpoint: {weights_path}")
    raise last_error


def maybe_build_segmenter(device: torch.device, weights_path: str) -> tuple[UNet, int] | None:
    checkpoint_path = Path(weights_path)
    if not checkpoint_path.exists():
        print(f"Skipping missing checkpoint: {checkpoint_path}")
        return None
    return build_segmenter(device=device, weights_path=str(checkpoint_path))


def maybe_build_segmenter_ensemble(
    device: torch.device,
    best_weights_path: str,
    family: str,
    source_contrast: str,
    num_ensemble: int,
) -> tuple[list[UNet], int | None]:
    best_path = Path(best_weights_path)
    if not best_path.exists():
        print(f"Skipping missing checkpoint: {best_path}")
        return [], None

    models: list[UNet] = []
    best_model, out_channels = build_segmenter(device=device, weights_path=str(best_path))
    models.append(best_model)

    # num_ensemble is the TOTAL model count including the best checkpoint.
    extras_to_load = max(0, num_ensemble - 1)

    if extras_to_load > 0:
        ensemble_dir = best_path.parent
        if best_path.suffix == ".ckpt":
            # v5 layout: segmenter_<epoch>_<metric>.ckpt + last.ckpt
            candidates = sorted(
                ensemble_dir.glob("segmenter_*.ckpt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for candidate in candidates[:extras_to_load]:
                candidate_model, candidate_out = build_segmenter(device=device, weights_path=str(candidate))
                if candidate_out != out_channels:
                    print(
                        f"Skipping ensemble checkpoint with mismatched out_channels ({candidate_out}): {candidate}"
                    )
                    continue
                models.append(candidate_model)
        else:
            # v4 and older layout: best_segmenter.pth + last_segmenter_i.pth
            for idx in range(1, extras_to_load + 1):
                candidate = ensemble_dir / f"last_segmenter_{idx}.pth"
                if not candidate.exists():
                    print(f"Skipping missing ensemble checkpoint: {candidate}")
                    continue
                candidate_model, candidate_out = build_segmenter(device=device, weights_path=str(candidate))
                if candidate_out != out_channels:
                    print(
                        f"Skipping ensemble checkpoint with mismatched out_channels ({candidate_out}): {candidate}"
                    )
                    continue
                models.append(candidate_model)

    return models, out_channels


def _predict_from_ensemble(
    models: list[UNet],
    inferer: SlidingWindowInferer,
    x_input: torch.Tensor,
) -> torch.Tensor:
    probabilities: list[torch.Tensor] = []
    for model in models:
        logits = inferer(x_input, model)

        # Binary UNet outputs one channel; convert to two-class logits before softmax.
        if logits.shape[1] == 1:
            logits = torch.cat([torch.zeros_like(logits), logits], dim=1)

        probabilities.append(torch.softmax(logits, dim=1))

    mean_probabilities = torch.mean(torch.stack(probabilities, dim=0), dim=0)
    return torch.argmax(mean_probabilities, dim=1, keepdim=True).long()


def _validate_family(family: str) -> str:
    normalized = family.strip().lower()
    if normalized == "fully_artificial":
        normalized = "fullyartificial"
    valid = {"baseline", "generator", "bigaug", "fullyartificial"}
    if normalized not in valid:
        raise ValueError(f"Unsupported family '{family}'. Expected one of: {sorted(valid)}")
    return normalized


def _parse_enabled(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    if text in {"", "1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid enabled value '{value}'. Use 0/1 or true/false.")


def _parse_bool_for_freezing(value: Any) -> str | None:
    """Parse and canonicalize freeze_encoder bool for output routing."""
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return "true"
    if text in {"0", "false", "no", "n"}:
        return "false"
    raise ValueError(f"Invalid freeze_encoder value '{value}'. Use 0/1 or true/false.")


def _nested_get(mapping: Any, keys: list[str], default: Any = None) -> Any:
    cur = mapping
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
            continue
        if hasattr(cur, key):
            cur = getattr(cur, key)
            continue
        try:
            cur = cur[key]
        except Exception:
            return default
    return cur if cur is not None else default


def _checkpoint_finetune_info(checkpoint_path: str) -> dict[str, Any]:
    path = Path(checkpoint_path)
    info: dict[str, Any] = {
        "is_finetuned": False,
        "freeze_encoder": None,
        "finetune_target_contrast": None,
        "version": None,
    }

    if not path.exists() or path.suffix != ".ckpt":
        return info

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return info

    hyper = checkpoint.get("hyper_parameters") if isinstance(checkpoint, dict) else None
    if hyper is None:
        return info

    pretrained_ckpt_path = _nested_get(hyper, ["model", "segmenter", "pretrained_ckpt_path"], default=None)
    freeze_encoder = _nested_get(hyper, ["model", "segmenter", "freeze_encoder"], default=None)
    target_contrast = _nested_get(hyper, ["data", "source_contrast"], default=None)
    version = _nested_get(hyper, ["version"], default=None)

    is_finetuned = pretrained_ckpt_path not in (None, "", "null")
    freeze_encoder_str = None
    if freeze_encoder is not None:
        freeze_encoder_str = "true" if bool(freeze_encoder) else "false"

    info["is_finetuned"] = bool(is_finetuned)
    info["freeze_encoder"] = freeze_encoder_str
    if target_contrast is not None:
        info["finetune_target_contrast"] = normalize_contrast_name(str(target_contrast))
    if version is not None:
        info["version"] = str(version)

    return info


def _normalize_model_spec(raw: dict[str, Any]) -> dict[str, Any]:
    # Check if this is a fine-tuned model
    is_finetuned = _parse_enabled(raw.get("is_finetuned", False))
    freeze_encoder = None
    if is_finetuned:
        freeze_encoder = _parse_bool_for_freezing(raw.get("freeze_encoder", None))
    
    return {
        "model_id": str(raw["model_id"]).strip(),
        "family": _validate_family(str(raw["family"])),
        "source_contrast": normalize_contrast_name(str(raw["source_contrast"])),
        "checkpoint_path": str(raw["checkpoint_path"]).strip(),
        "enabled": _parse_enabled(raw.get("enabled", True)),
        "is_finetuned": is_finetuned,
        "freeze_encoder": freeze_encoder,
        "finetune_target_contrast": raw.get("finetune_target_contrast"),
        "version": raw.get("version"),
    }


def _load_models_from_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"models-file not found: {path}")

    required = {"model_id", "family", "source_contrast", "checkpoint_path"}
    specs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        missing = required - header
        if missing:
            raise ValueError(f"models-file missing required columns: {sorted(missing)}")
        for row in reader:
            spec = _normalize_model_spec(row)
            if spec["enabled"]:
                specs.append(spec)
    return specs


def _load_models_from_inline(inline_specs: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for item in inline_specs:
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 4:
            raise ValueError(
                f"Invalid --model '{item}'. Expected format: model_id,family,source_contrast,checkpoint_path"
            )
        spec = _normalize_model_spec(
            {
                "model_id": parts[0],
                "family": parts[1],
                "source_contrast": parts[2],
                "checkpoint_path": parts[3],
                "enabled": True,
            }
        )
        specs.append(spec)
    return specs


def _discover_models(checkpoint_dir: Path) -> list[dict[str, Any]]:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"discover-checkpoints directory not found: {checkpoint_dir}")

    # Pattern matches: best_segmenter_<family>_<contrast>.pth
    # Also matches: best_segmenter_generator_<contrast>.pth inside v1/v2 folders
    pattern = re.compile(r"^best_segmenter_(baseline|generator|bigaug|fullyartificial)_([a-zA-Z0-9]+)\.pth$")
    v4_model_id_pattern = re.compile(r"^(?:v4_)?(baseline|generator|bigaug|fullyartificial)_([a-zA-Z0-9]+)$")
    specs: list[dict[str, Any]] = []
    
    # Legacy pattern: best_segmenter_<family>_<contrast>.pth
    for checkpoint in sorted(checkpoint_dir.rglob("*.pth")):
        match = pattern.match(checkpoint.name)
        if match:
            family = match.group(1)
            contrast = normalize_contrast_name(match.group(2))

            # Determine a useful Model ID by looking at the parent directory name
            parent_dir = checkpoint.parent.name
            if parent_dir in ["v1", "v2", "v3", "baseline"]:
                model_id = f"{parent_dir}_{family}_{contrast}"
            else:
                model_id = f"{family}_{contrast}"

            specs.append(
                _normalize_model_spec(
                    {
                        "model_id": model_id,
                        "family": family,
                        "source_contrast": contrast,
                        "checkpoint_path": str(checkpoint),
                        "enabled": True,
                    }
                )
            )
            continue

        if checkpoint.name == "best_segmenter.pth":
            run_dir_name = checkpoint.parent.name
            v4_match = v4_model_id_pattern.match(run_dir_name)
            if not v4_match:
                continue

            family = v4_match.group(1)
            contrast = normalize_contrast_name(v4_match.group(2))
            specs.append(
                _normalize_model_spec(
                    {
                        "model_id": run_dir_name,
                        "family": family,
                        "source_contrast": contrast,
                        "checkpoint_path": str(checkpoint),
                        "enabled": True,
                    }
                )
            )

    # v5+ and runX pattern:
    # - checkpoints/<version>/segmenter/<family>/<contrast>/runX/{last,checkpoint,checkpoints}.ckpt
    # - checkpoints/<version>/segmenter/<family>/<contrast>/last.ckpt (legacy compatibility)
    # - checkpoints/segmenter/<family>/<contrast>/runX/{last,checkpoint,checkpoints}.ckpt
    latest_segmenter_ckpts: dict[tuple[str, str], Path] = {}
    candidate_ckpts: list[Path] = []
    for marker_name in ("last.ckpt", "checkpoint.ckpt", "checkpoints.ckpt"):
        candidate_ckpts.extend(checkpoint_dir.rglob(marker_name))

    for checkpoint in sorted(candidate_ckpts):
        parts = checkpoint.parts
        if "segmenter" not in parts:
            continue

        seg_idx = parts.index("segmenter")
        if seg_idx + 2 >= len(parts):
            continue

        family = _validate_family(parts[seg_idx + 1])
        hyper = _load_hyper_parameters(str(checkpoint))
        hyper_contrast = _nested_get(hyper, ["data", "source_contrast"], default=None)
        contrast_raw = hyper_contrast if hyper_contrast is not None else parts[seg_idx + 2]
        contrast = normalize_contrast_name(str(contrast_raw))
        key = (family, contrast)

        prev = latest_segmenter_ckpts.get(key)
        if prev is None or checkpoint.stat().st_mtime > prev.stat().st_mtime:
            latest_segmenter_ckpts[key] = checkpoint

    for (family, contrast), checkpoint in sorted(latest_segmenter_ckpts.items()):
        model_id = f"segmenter_{family}_{contrast}"
        hyper = _load_hyper_parameters(str(checkpoint))
        version = _nested_get(hyper, ["version"], default=None)
        specs.append(
            _normalize_model_spec(
                {
                    "model_id": model_id,
                    "family": family,
                    "source_contrast": contrast,
                    "checkpoint_path": str(checkpoint),
                    "enabled": True,
                    "version": version,
                }
            )
        )
    return specs


def _legacy_default_models(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        _normalize_model_spec(
            {
                "model_id": "baseline",
                "family": "baseline",
                "source_contrast": args.baseline_contrast,
                "checkpoint_path": args.baseline_ckpt or _default_checkpoint_path("baseline", args.baseline_contrast),
                "enabled": True,
            }
        ),
        _normalize_model_spec(
            {
                "model_id": "generator",
                "family": "generator",
                "source_contrast": args.generator_contrast,
                "checkpoint_path": args.generator_ckpt or _default_checkpoint_path("generator", args.generator_contrast),
                "enabled": True,
            }
        ),
        _normalize_model_spec(
            {
                "model_id": "bigaug",
                "family": "bigaug",
                "source_contrast": args.bigaug_contrast,
                "checkpoint_path": args.bigaug_ckpt or _default_checkpoint_path("bigaug", args.bigaug_contrast),
                "enabled": True,
            }
        ),
        _normalize_model_spec(
            {
                "model_id": "fullyartificial",
                "family": "fullyartificial",
                "source_contrast": args.generator_contrast,
                "checkpoint_path": _default_checkpoint_path("fullyartificial", args.generator_contrast),
                "enabled": True,
            }
        ),
    ]


def _collect_model_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    # 1. Always include baseline models by default.
    baseline_dir = PROJECT_ROOT / "checkpoints" / "baseline"
    include_baseline_auto = baseline_dir.exists() and not args.skip_baseline_auto
    if include_baseline_auto and args.discover_checkpoints:
        discovery_path = Path(args.discover_checkpoints).resolve()
        baseline_resolved = baseline_dir.resolve()
        if discovery_path == baseline_resolved or baseline_resolved.is_relative_to(discovery_path):
            include_baseline_auto = False

    if include_baseline_auto:
        print(f"Automatically including baseline checkpoints from: {baseline_dir}")
        specs.extend(_discover_models(baseline_dir))

    # 2. Add models from CSV if provided
    if args.models_file:
        specs.extend(_load_models_from_csv(Path(args.models_file)))
    
    # 3. Add models from the explicitly requested discovery directory
    if args.discover_checkpoints:
        discovery_path = Path(args.discover_checkpoints)
        # Avoid double-adding if the user passed the baseline dir manually
        if discovery_path.resolve() != baseline_dir.resolve():
            specs.extend(_discover_models(discovery_path))
            
    # 4. Add inline models
    if args.model:
        specs.extend(_load_models_from_inline(args.model))

    # 5. Fallback if still empty (legacy behavior)
    if not specs:
        default_checkpoint_dir = PROJECT_ROOT / "checkpoints"
        if default_checkpoint_dir.exists():
            specs = _discover_models(default_checkpoint_dir)
        else:
            specs = _legacy_default_models(args)

    # Deduplicate and handle model_id collisions
    model_id_counts: dict[str, int] = {}
    unique_specs: list[dict[str, Any]] = []
    for spec in specs:
        base_id = spec["model_id"]
        count = model_id_counts.get(base_id, 0)
        if count > 0:
            spec = dict(spec)
            spec["model_id"] = f"{base_id}_{count + 1}"
        model_id_counts[base_id] = count + 1
        unique_specs.append(spec)

    return unique_specs


def _print_results_table(results: list[dict[str, Any]], target_contrasts: list[str]) -> None:
    contrast_header = " | ".join(f"{c:<9}" for c in target_contrasts)
    line = "+----------------------+-----------+---------+" + "-----------+" * len(target_contrasts)
    print("\nSegmentation Dice on Held-Out Test Subjects")
    print(line)
    print(f"| Model ID             | Family    | Source  | {contrast_header} |")
    print(line)
    for row in results:
        contrast_values = " | ".join(f"{_format_metric(row[c]):>9}" for c in target_contrasts)
        print(
            f"| {row['model_id']:<20} | {row['family']:<9} | {row['source_contrast']:<7} | {contrast_values} |"
        )
    print(line)


def _write_long_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model_id",
        "family",
        "source_contrast",
        "target_contrast",
        "dice",
        "ckpt_exists",
        "checkpoint_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_wide_csv(path: Path, rows: list[dict[str, Any]], target_contrasts: list[str]) -> None:
    fields = [
        "model_id",
        "family",
        "source_contrast",
        "checkpoint_path",
        "ckpt_exists",
        *target_contrasts,
        "in_domain_dice",
        "ood_mean_dice",
        "ood_worst_dice",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_markdown(path: Path, rows: list[dict[str, Any]], target_contrasts: list[str]) -> None:
    contrast_header = " | ".join(target_contrasts)
    contrast_rule = " | ".join(["---:"] * len(target_contrasts))
    header = [
        f"| model_id | family | source_contrast | ckpt_exists | {contrast_header} | in_domain_dice | ood_mean_dice | ood_worst_dice |",
        f"|---|---|---|---:| {contrast_rule} |---:|---:|---:|",
    ]
    body = []
    for row in rows:
        body.append(
            "| "
            + " | ".join(
                [
                    str(row["model_id"]),
                    str(row["family"]),
                    str(row["source_contrast"]),
                    str(row["ckpt_exists"]),
                    *[_format_metric(row[c]) for c in target_contrasts],
                    _format_metric(row["in_domain_dice"]),
                    _format_metric(row["ood_mean_dice"]),
                    _format_metric(row["ood_worst_dice"]),
                ]
            )
            + " |"
        )

    content = ["# Evaluation Summary", ""] + header + body + [""]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.num_ensemble < 1:
        raise ValueError("--num-ensemble must be >= 1.")
    if args.tta_samples < 0:
        raise ValueError("--tta-samples must be >= 0.")
    if args.min_sw_batch_size < 1:
        raise ValueError("--min-sw-batch-size must be >= 1.")

    global _GLOBAL_MIN_SW_BATCH_SIZE
    _GLOBAL_MIN_SW_BATCH_SIZE = int(args.min_sw_batch_size)

    model_specs = _collect_model_specs(args)
    if not model_specs:
        raise ValueError("No model specifications were provided.")

    target_contrasts = _resolve_target_contrasts(model_specs)
    contrast_to_index = build_contrast_to_index(target_contrasts)
    label_mapping = _load_label_mapping_from_default_config()
    args.baseline_contrast = normalize_contrast_name(args.baseline_contrast, target_contrasts)
    args.generator_contrast = normalize_contrast_name(args.generator_contrast, target_contrasts)
    args.bigaug_contrast = normalize_contrast_name(args.bigaug_contrast, target_contrasts)

    for spec in model_specs:
        runtime_info = _segmenter_runtime_info(spec["checkpoint_path"])
        if runtime_info["source_contrast"] is not None:
            spec["source_contrast"] = normalize_contrast_name(str(runtime_info["source_contrast"]), target_contrasts)
        if runtime_info["version"] is not None:
            spec["version"] = runtime_info["version"]
        spec["_runtime_info"] = runtime_info

    device = torch.device(args.device)

    output_dir = Path(args.output_dir)

    transforms = build_eval_transforms()
    full_dataset = DecathlonDataset(
        root_dir=args.data_dir,
        task="Task01_BrainTumour",
        transform=transforms,
        section="training",
        download=True,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )

    split_path = Path(args.split_file)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}. Run scripts/train_segmenter.py first to create consistent train/val/test splits."
        )

    with split_path.open("r", encoding="utf-8") as f:
        split = json.load(f)

    test_subjects = set(split.get("test_subjects", []))
    if not test_subjects:
        raise ValueError(f"No test subjects found in split file: {split_path}")

    test_indices = [
        idx for idx, sample in enumerate(full_dataset.data) if _subject_id_from_sample(sample) in test_subjects
    ]
    if not test_indices:
        raise ValueError("No matching test indices found in dataset for the provided split file.")

    dataset = Subset(full_dataset, test_indices)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available() and not args.disable_pin_memory,
    }
    if args.num_workers > 0:
        if not args.disable_persistent_workers:
            loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    dataloader = DataLoader(dataset, **loader_kwargs)

    print(f"Using split file: {split_path}")
    print(f"Evaluating on held-out test subjects: {len(test_subjects)} (samples: {len(test_indices)})")
    print(f"Evaluating model specs: {len(model_specs)}")

    inferer = SlidingWindowInferer(roi_size=(128, 128, 128), sw_batch_size=args.sw_batch_size)
    print(
        "Eval settings: "
        f"batch_size={args.batch_size}, sw_batch_size={args.sw_batch_size}, "
        f"amp_enabled={device.type == 'cuda' and not args.disable_amp}, "
        f"num_ensemble={args.num_ensemble}"
    )

    model_instances: dict[str, list[UNet]] = {}
    model_out_channels: dict[str, int | None] = {}
    metrics: dict[str, dict[str, DiceMetric]] = {}
    exists_map: dict[str, int] = {}
    for spec in model_specs:
        model_id = spec["model_id"]
        checkpoint_path = spec["checkpoint_path"]
        print(
            f"Model {model_id}: family={spec['family']} source={spec['source_contrast']} checkpoint={checkpoint_path}"
        )
        models, out_channels = maybe_build_segmenter_ensemble(
            device=device,
            best_weights_path=checkpoint_path,
            family=spec["family"],
            source_contrast=spec["source_contrast"],
            num_ensemble=args.num_ensemble,
        )
        model_instances[model_id] = models
        model_out_channels[model_id] = out_channels
        exists_map[model_id] = 1 if len(models) > 0 else 0
        print(f"Loaded checkpoints for {model_id}: {len(models)}")

        is_multiclass_model = (out_channels or 1) > 1
        metrics[model_id] = {
            contrast: DiceMetric(
                include_background=False,
                reduction="mean_batch" if is_multiclass_model else "mean",
            )
            for contrast in target_contrasts
        }

    loaded_channels = [channels for channels in model_out_channels.values() if channels is not None]
    inferred_mode = "multiclass" if any(channels > 1 for channels in loaded_channels) else "binary"
    task_mode = inferred_mode if args.task_mode == "auto" else args.task_mode

    route_name = _build_eval_mode_name(task_mode=task_mode, tta_samples=int(args.tta_samples), exclude_original=bool(args.exclude_original))
    effective_output_dir = output_dir if route_name == "binary" else output_dir / route_name
    effective_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation task mode: {task_mode}")
    print(f"Saving artifacts under: {effective_output_dir}")

    # Derive fine-tuning metadata directly from checkpoint configs.
    for spec in model_specs:
        ckpt_info = _checkpoint_finetune_info(spec["checkpoint_path"])
        spec["is_finetuned"] = bool(spec.get("is_finetuned", False) or ckpt_info["is_finetuned"])
        if spec.get("freeze_encoder") is None:
            spec["freeze_encoder"] = ckpt_info["freeze_encoder"]
        if spec.get("finetune_target_contrast") is None:
            spec["finetune_target_contrast"] = ckpt_info["finetune_target_contrast"]
        if spec.get("version") is None:
            spec["version"] = ckpt_info["version"]

    amp_enabled = device.type == "cuda" and not args.disable_amp
    autocast_dtype = torch.float16
    generator_cache: dict[str, MRI_Synthesis_Net] = {}
    hist_module_cache: dict[str, DifferentiableHistogram3D] = {}

    def _get_generator_for_spec(spec: dict[str, Any]) -> MRI_Synthesis_Net:
        runtime_info = spec["_runtime_info"]
        source_contrast = str(runtime_info["source_contrast"] or spec["source_contrast"])
        cached = generator_cache.get(source_contrast)
        if cached is not None:
            return cached

        checkpoint_path = _resolve_latest_generator_checkpoint(source_contrast)
        generator = _load_generator_model(device=device, checkpoint_path=checkpoint_path, runtime_info=runtime_info)
        generator_cache[source_contrast] = generator
        print(f"Loaded TTA generator for {source_contrast}: {checkpoint_path}")
        return generator

    def _get_hist_module_for_spec(spec: dict[str, Any]) -> DifferentiableHistogram3D:
        runtime_info = spec["_runtime_info"]
        source_contrast = str(runtime_info["source_contrast"] or spec["source_contrast"])
        cache_key = f"{source_contrast}:{int(runtime_info['num_bins'])}"
        cached = hist_module_cache.get(cache_key)
        if cached is not None:
            return cached

        hist_module = DifferentiableHistogram3D(
            num_bins=int(runtime_info["num_bins"]),
            value_range=(0.0, 1.0),
        ).to(device)
        hist_module_cache[cache_key] = hist_module
        return hist_module

    with torch.inference_mode():
        for batch in dataloader:
            x = batch["image"].to(device, non_blocking=True).float()
            y_raw = batch["label"].to(device, non_blocking=True).long()
            for src, dst in label_mapping.items():
                y_raw = torch.where(y_raw == int(src), torch.full_like(y_raw, int(dst)), y_raw)

            batch_size = x.shape[0]
            x_by_contrast = {
                contrast: _extract_contrast(x, contrast, contrast_to_index)
                for contrast in target_contrasts
            }

            for spec in model_specs:
                model_id = spec["model_id"]
                models = model_instances[model_id]
                if not models:
                    continue

                out_channels = model_out_channels[model_id] or (4 if task_mode == "multiclass" else 1)
                if out_channels == 1:
                    y_target = (y_raw > 0).float()
                else:
                    y_target = y_raw.clamp(min=0, max=out_channels - 1).long()

                runtime_info = spec["_runtime_info"]
                generator = _get_generator_for_spec(spec) if int(args.tta_samples) > 0 else None
                hist_module = _get_hist_module_for_spec(spec) if int(args.tta_samples) > 0 else None

                for contrast in target_contrasts:
                    x_input = x_by_contrast[contrast]
                    probs_sum: torch.Tensor | None = None
                    num_passes = 0

                    if not bool(args.exclude_original):
                        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
                            probs = _predict_probs_from_ensemble(models=models, inferer=inferer, x_input=x_input)
                        probs_sum = probs
                        num_passes += 1

                    if int(args.tta_samples) > 0:
                        assert generator is not None
                        assert hist_module is not None
                        for _ in range(int(args.tta_samples)):
                            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
                                x_synth = _generate_synth_input(
                                    generator=generator,
                                    hist_module=hist_module,
                                    x_input=x_input,
                                    runtime_info=runtime_info,
                                )
                            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
                                probs = _predict_probs_from_ensemble(models=models, inferer=inferer, x_input=x_synth)
                            if probs_sum is None:
                                probs_sum = probs
                            else:
                                probs_sum = probs_sum + probs
                            num_passes += 1
                            del x_synth

                    if num_passes == 0 or probs_sum is None:
                        raise ValueError("No evaluation passes were executed. Set --tta-samples > 0 when using --exclude-original.")

                    avg_probs = probs_sum / float(num_passes)
                    if out_channels > 1:
                        pred_chunk = torch.argmax(avg_probs, dim=1, keepdim=True).long()
                    else:
                        pred_chunk = (avg_probs > 0.5).float()

                    if out_channels > 1:
                        pred_onehot = torch.nn.functional.one_hot(
                            pred_chunk[:, 0].long(),
                            num_classes=out_channels,
                        ).permute(0, 4, 1, 2, 3).float()
                        y_onehot = torch.nn.functional.one_hot(
                            y_target[:, 0].long(),
                            num_classes=out_channels,
                        ).permute(0, 4, 1, 2, 3).float()
                        metrics[model_id][contrast](y_pred=pred_onehot, y=y_onehot)
                    else:
                        metrics[model_id][contrast](y_pred=pred_chunk.float(), y=y_target)

    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    for spec in model_specs:
        model_id = spec["model_id"]
        source_contrast = spec["source_contrast"]
        has_ckpt = exists_map[model_id] == 1

        contrast_scores: dict[str, float | None] = {}
        for target_contrast in target_contrasts:
            dice_value: float | None
            if has_ckpt:
                aggregated = metrics[model_id][target_contrast].aggregate()
                out_channels = model_out_channels[model_id] or (4 if task_mode == "multiclass" else 1)
                if out_channels > 1:
                    per_class = aggregated
                    if per_class.ndim > 1:
                        per_class = per_class.mean(dim=0)
                    dice_value = float(per_class.mean().item())
                else:
                    dice_value = float(aggregated.item())
            else:
                dice_value = None

            contrast_scores[target_contrast] = dice_value
            long_rows.append(
                {
                    "model_id": model_id,
                    "family": spec["family"],
                    "source_contrast": source_contrast,
                    "target_contrast": target_contrast,
                    "dice": "" if dice_value is None else f"{dice_value:.6f}",
                    "ckpt_exists": exists_map[model_id],
                    "checkpoint_path": spec["checkpoint_path"],
                }
            )

        in_domain_dice = contrast_scores.get(source_contrast)
        ood_values = [
            score
            for contrast_name, score in contrast_scores.items()
            if contrast_name != source_contrast and score is not None
        ]
        ood_mean = _safe_mean(ood_values)
        ood_worst = min(ood_values) if ood_values else None

        wide_row = {
            "model_id": model_id,
            "family": spec["family"],
            "source_contrast": source_contrast,
            "checkpoint_path": spec["checkpoint_path"],
            "ckpt_exists": exists_map[model_id],
            "in_domain_dice": "" if in_domain_dice is None else f"{in_domain_dice:.6f}",
            "ood_mean_dice": "" if ood_mean is None else f"{ood_mean:.6f}",
            "ood_worst_dice": "" if ood_worst is None else f"{ood_worst:.6f}",
        }
        for contrast in target_contrasts:
            score = contrast_scores.get(contrast)
            wide_row[contrast] = "" if score is None else f"{score:.6f}"
        wide_rows.append(wide_row)

    table_rows = []
    for row in wide_rows:
        row_for_table = {
            "model_id": row["model_id"],
            "family": row["family"],
            "source_contrast": row["source_contrast"],
        }
        for contrast in target_contrasts:
            value = row.get(contrast, "")
            row_for_table[contrast] = None if value == "" else float(value)
        table_rows.append(row_for_table)

    _print_results_table(table_rows, target_contrasts)

    # Separate rows into fine-tuned and non-fine-tuned groups.
    spec_map = {spec["model_id"]: spec for spec in model_specs}
    finetuned_model_ids = {
        spec["model_id"] for spec in model_specs if bool(spec.get("is_finetuned", False))
    }

    nonfinetuned_long_rows = [row for row in long_rows if row["model_id"] not in finetuned_model_ids]
    nonfinetuned_wide_rows = [row for row in wide_rows if row["model_id"] not in finetuned_model_ids]

    # Standard zero-shot output remains unchanged and protected.
    if nonfinetuned_wide_rows:
        effective_output_dir.mkdir(parents=True, exist_ok=True)
        long_csv = effective_output_dir / "eval_long.csv"
        wide_csv = effective_output_dir / "eval_wide.csv"
        summary_md = effective_output_dir / "eval_summary.md"

        _write_long_csv(long_csv, nonfinetuned_long_rows)
        _write_wide_csv(wide_csv, nonfinetuned_wide_rows, target_contrasts)
        _write_summary_markdown(summary_md, nonfinetuned_wide_rows, target_contrasts)

        print(f"Saved long-form metrics to: {long_csv}")
        print(f"Saved wide metrics to: {wide_csv}")
        print(f"Saved markdown summary to: {summary_md}")

    # Route fine-tuned models to results/eval/<version>/finetuned/<target_contrast>_freeze_<bool>/
    if finetuned_model_ids:
        grouped_ids: dict[tuple[str, str, str], set[str]] = {}
        for model_id in finetuned_model_ids:
            spec = spec_map[model_id]
            version = str(spec.get("version") or output_dir.name)
            target_contrast = str(spec.get("finetune_target_contrast") or spec.get("source_contrast") or "unknown")
            freeze_encoder = str(spec.get("freeze_encoder") or "unknown")
            key = (version, target_contrast, freeze_encoder)
            grouped_ids.setdefault(key, set()).add(model_id)

        for (version, target_contrast, freeze_encoder), model_ids in grouped_ids.items():
            # If the caller already included the version component in the output path,
            # avoid adding it again (prevents nested `.../v15/.../v15/...`).
            if version in effective_output_dir.parts or effective_output_dir.name == version:
                version_root = effective_output_dir
            else:
                version_root = effective_output_dir / version

            finetuned_dir = version_root / "finetuned" / f"{target_contrast}_freeze_{freeze_encoder}"
            finetuned_dir.mkdir(parents=True, exist_ok=True)

            ft_long_rows = [row for row in long_rows if row["model_id"] in model_ids]
            ft_wide_rows = [row for row in wide_rows if row["model_id"] in model_ids]

            long_csv = finetuned_dir / "eval_long.csv"
            wide_csv = finetuned_dir / "eval_wide.csv"
            summary_md = finetuned_dir / "eval_summary.md"

            _write_long_csv(long_csv, ft_long_rows)
            _write_wide_csv(wide_csv, ft_wide_rows, target_contrasts)
            _write_summary_markdown(summary_md, ft_wide_rows, target_contrasts)

            print(f"Saved fine-tuned metrics to: {finetuned_dir}")


if __name__ == "__main__":
    main()
