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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CONTRAST_TO_INDEX, normalize_contrast_name

TARGET_CONTRASTS = ["flair", "t1w", "t1gd", "t2w"]


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
        default=4,
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

    # Legacy fallback flags (kept for backward compatibility).
    parser.add_argument(
        "--baseline-ckpt",
        type=str,
        default=None,
    )
    parser.add_argument("--baseline-contrast", type=str, default="t1w", choices=sorted(CONTRAST_TO_INDEX))
    parser.add_argument(
        "--generator-ckpt",
        type=str,
        default=None,
    )
    parser.add_argument("--generator-contrast", type=str, default="t1w", choices=sorted(CONTRAST_TO_INDEX))
    parser.add_argument(
        "--bigaug-ckpt",
        type=str,
        default=None,
    )
    parser.add_argument("--bigaug-contrast", type=str, default="t1w", choices=sorted(CONTRAST_TO_INDEX))
    parser.add_argument("--num-workers", type=int, default=4)
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
    parser.add_argument("--split-file", type=str, default=str(PROJECT_ROOT / "splits" / "brats_subject_split.json"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _default_checkpoint_path(tag: str, contrast: str) -> str:
    if tag == "baseline":
        return str(PROJECT_ROOT / "checkpoints" / "baseline" / f"baseline_{contrast}" / "best_segmenter.pth")
    return str(PROJECT_ROOT / "checkpoints" / f"best_segmenter_{tag}_{contrast}.pth")


def _extract_contrast(image: torch.Tensor, contrast: str) -> torch.Tensor:
    index = CONTRAST_TO_INDEX[normalize_contrast_name(contrast)]
    return image[:, index : index + 1, ...]


def _stack_target_contrasts(image: torch.Tensor) -> tuple[torch.Tensor, int]:
    # Convert [B, C, ...] multi-contrast input to [B*4, 1, ...] in TARGET_CONTRASTS order.
    batch_size = image.shape[0]
    stacked = torch.cat([_extract_contrast(image, contrast) for contrast in TARGET_CONTRASTS], dim=0)
    return stacked, batch_size


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


def build_segmenter(device: torch.device, weights_path: str) -> UNet:
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)
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

    try:
        model.load_state_dict(normalized_state_dict, strict=True)
    except RuntimeError:
        # Some checkpoints store keys without the top-level "model." prefix.
        prefixed_state_dict = OrderedDict((f"model.{k}", v) for k, v in normalized_state_dict.items())
        model.load_state_dict(prefixed_state_dict, strict=True)
    model.eval()
    return model


def maybe_build_segmenter(device: torch.device, weights_path: str) -> UNet | None:
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
) -> list[UNet]:
    best_path = Path(best_weights_path)
    if not best_path.exists():
        print(f"Skipping missing checkpoint: {best_path}")
        return []

    models: list[UNet] = []
    models.append(build_segmenter(device=device, weights_path=str(best_path)))

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
                models.append(build_segmenter(device=device, weights_path=str(candidate)))
        else:
            # v4 and older layout: best_segmenter.pth + last_segmenter_i.pth
            for idx in range(1, extras_to_load + 1):
                candidate = ensemble_dir / f"last_segmenter_{idx}.pth"
                if not candidate.exists():
                    print(f"Skipping missing ensemble checkpoint: {candidate}")
                    continue
                models.append(build_segmenter(device=device, weights_path=str(candidate)))

    return models


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
    return torch.argmax(mean_probabilities, dim=1, keepdim=True).float()


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


def _normalize_model_spec(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": str(raw["model_id"]).strip(),
        "family": _validate_family(str(raw["family"])),
        "source_contrast": normalize_contrast_name(str(raw["source_contrast"])),
        "checkpoint_path": str(raw["checkpoint_path"]).strip(),
        "enabled": _parse_enabled(raw.get("enabled", True)),
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
        contrast = normalize_contrast_name(parts[seg_idx + 2])
        key = (family, contrast)

        prev = latest_segmenter_ckpts.get(key)
        if prev is None or checkpoint.stat().st_mtime > prev.stat().st_mtime:
            latest_segmenter_ckpts[key] = checkpoint

    for (family, contrast), checkpoint in sorted(latest_segmenter_ckpts.items()):
        model_id = f"segmenter_{family}_{contrast}"
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


def _print_results_table(results: list[dict[str, Any]]) -> None:
    line = "+----------------------+-----------+---------+-----------+-----------+-----------+-----------+"
    print("\nSegmentation Dice on Held-Out Test Subjects")
    print(line)
    print("| Model ID             | Family    | Source  | FLAIR     | T1w       | T1gd      | T2w       |")
    print(line)
    for row in results:
        print(
            f"| {row['model_id']:<20} | {row['family']:<9} | {row['source_contrast']:<7} | "
            f"{_format_metric(row['flair']):>9} | {_format_metric(row['t1w']):>9} | {_format_metric(row['t1gd']):>9} | {_format_metric(row['t2w']):>9} |"
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


def _write_wide_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model_id",
        "family",
        "source_contrast",
        "checkpoint_path",
        "ckpt_exists",
        "flair",
        "t1w",
        "t1gd",
        "t2w",
        "in_domain_dice",
        "ood_mean_dice",
        "ood_worst_dice",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    header = [
        "| model_id | family | source_contrast | ckpt_exists | flair | t1w | t1gd | t2w | in_domain_dice | ood_mean_dice | ood_worst_dice |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
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
                    _format_metric(row["flair"]),
                    _format_metric(row["t1w"]),
                    _format_metric(row["t1gd"]),
                    _format_metric(row["t2w"]),
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

    args.baseline_contrast = normalize_contrast_name(args.baseline_contrast)
    args.generator_contrast = normalize_contrast_name(args.generator_contrast)
    args.bigaug_contrast = normalize_contrast_name(args.bigaug_contrast)
    device = torch.device(args.device)

    model_specs = _collect_model_specs(args)
    if not model_specs:
        raise ValueError("No model specifications were provided.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
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
    metrics: dict[str, dict[str, DiceMetric]] = {}
    exists_map: dict[str, int] = {}
    for spec in model_specs:
        model_id = spec["model_id"]
        checkpoint_path = spec["checkpoint_path"]
        print(
            f"Model {model_id}: family={spec['family']} source={spec['source_contrast']} checkpoint={checkpoint_path}"
        )
        models = maybe_build_segmenter_ensemble(
            device=device,
            best_weights_path=checkpoint_path,
            family=spec["family"],
            source_contrast=spec["source_contrast"],
            num_ensemble=args.num_ensemble,
        )
        model_instances[model_id] = models
        exists_map[model_id] = 1 if len(models) > 0 else 0
        print(f"Loaded checkpoints for {model_id}: {len(models)}")
        metrics[model_id] = {
            contrast: DiceMetric(include_background=False, reduction="mean")
            for contrast in TARGET_CONTRASTS
        }

    amp_enabled = device.type == "cuda" and not args.disable_amp
    autocast_dtype = torch.float16

    with torch.inference_mode():
        for batch in dataloader:
            x = batch["image"].to(device, non_blocking=True).float()
            y = (batch["label"].to(device, non_blocking=True) > 0).float()

            x_stacked, batch_size = _stack_target_contrasts(x)

            for spec in model_specs:
                model_id = spec["model_id"]
                models = model_instances[model_id]
                if not models:
                    continue

                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
                    pred = _predict_from_ensemble(models=models, inferer=inferer, x_input=x_stacked)

                for idx, contrast in enumerate(TARGET_CONTRASTS):
                    start = idx * batch_size
                    end = (idx + 1) * batch_size
                    metrics[model_id][contrast](y_pred=pred[start:end], y=y)

    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    for spec in model_specs:
        model_id = spec["model_id"]
        source_contrast = spec["source_contrast"]
        has_ckpt = exists_map[model_id] == 1

        contrast_scores: dict[str, float | None] = {}
        for target_contrast in TARGET_CONTRASTS:
            dice_value: float | None
            if has_ckpt:
                dice_value = float(metrics[model_id][target_contrast].aggregate().item())
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

        wide_rows.append(
            {
                "model_id": model_id,
                "family": spec["family"],
                "source_contrast": source_contrast,
                "checkpoint_path": spec["checkpoint_path"],
                "ckpt_exists": exists_map[model_id],
                "flair": "" if contrast_scores["flair"] is None else f"{contrast_scores['flair']:.6f}",
                "t1w": "" if contrast_scores["t1w"] is None else f"{contrast_scores['t1w']:.6f}",
                "t1gd": "" if contrast_scores["t1gd"] is None else f"{contrast_scores['t1gd']:.6f}",
                "t2w": "" if contrast_scores["t2w"] is None else f"{contrast_scores['t2w']:.6f}",
                "in_domain_dice": "" if in_domain_dice is None else f"{in_domain_dice:.6f}",
                "ood_mean_dice": "" if ood_mean is None else f"{ood_mean:.6f}",
                "ood_worst_dice": "" if ood_worst is None else f"{ood_worst:.6f}",
            }
        )

    table_rows = []
    for row in wide_rows:
        table_rows.append(
            {
                "model_id": row["model_id"],
                "family": row["family"],
                "source_contrast": row["source_contrast"],
                "flair": None if row["flair"] == "" else float(row["flair"]),
                "t1w": None if row["t1w"] == "" else float(row["t1w"]),
                "t1gd": None if row["t1gd"] == "" else float(row["t1gd"]),
                "t2w": None if row["t2w"] == "" else float(row["t2w"]),
            }
        )

    _print_results_table(table_rows)

    long_csv = output_dir / "eval_long.csv"
    wide_csv = output_dir / "eval_wide.csv"
    summary_md = output_dir / "eval_summary.md"
    _write_long_csv(long_csv, long_rows)
    _write_wide_csv(wide_csv, wide_rows)
    _write_summary_markdown(summary_md, wide_rows)

    print(f"Saved long-form metrics to: {long_csv}")
    print(f"Saved wide metrics to: {wide_csv}")
    print(f"Saved markdown summary to: {summary_md}")


if __name__ == "__main__":
    main()
