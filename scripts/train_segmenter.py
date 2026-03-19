from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from torch.utils.data import Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CONTRAST_TO_INDEX, get_preprocessing_transforms, normalize_contrast_name
from src.generator import MRI_Synthesis_Net
from src.histogram_ops import (
    DifferentiableHistogram3D,
    apply_gaussian_blur_3d,
    create_range_translation_guidance_map,
    generate_unified_targets,
)

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')


SUPPORTED_GENERATOR_VERSIONS = ("v1", "v2", "v3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3D UNet segmenter on BraTS Task01.")
    parser.add_argument(
        "--fully-artificial",
        action="store_true",
        help="Train a fully artificial segmenter (forces generator augmentation with aug-prob=1.0).",
    )
    parser.add_argument("--use-generator", action="store_true", help="Enable generator-based contrast augmentation.")
    parser.add_argument("--gen-weights", type=str, default=None, help="Path to pre-trained MRI_Synthesis_Net .pth file.")
    parser.add_argument(
        "--gen-version",
        type=str,
        default="v2",
        choices=SUPPORTED_GENERATOR_VERSIONS,
        help="Generator pipeline version: v1/v2 legacy flow, v3 percentile-synchronized targets.",
    )
    parser.add_argument("--aug-prob-train", type=float, default=0.7, help="Probability of applying generator augmentation during training.")
    parser.add_argument("--aug-prob-val", type=float, default=0.0, help="Probability of applying generator augmentation during validation.")
    parser.add_argument(
        "--baseline-contrast",
        type=str,
        default="t1w",
        choices=sorted(CONTRAST_TO_INDEX),
        help="Source MRI contrast to use for training and validation.",
    )

    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--batch-size", type=int, default=8)
    # OPTIMIZATION: Bumped default val batch size to 4 to speed up evaluation
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--cache-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-file", type=str, default=str(PROJECT_ROOT / "splits" / "brats_subject_split.json"))

    parser.add_argument("--num-bins", type=int, default=128)
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--dark-threshold", type=float, default=0.05)

    parser.add_argument("--project-name", type=str, default="brats-segmenter")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--version", type=str, default="v2", help="Checkpoint version directory (only used with --use-generator).")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--image-log-every", type=int, default=50)
    parser.add_argument("--val-image-log-every", type=int, default=1)
    
    parser.add_argument("--compile-model", dest="compile_model", action="store_true")
    parser.add_argument("--no-compile-model", dest="compile_model", action="store_false")
    parser.set_defaults(compile_model=True)
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _build_logging_panel(
    original: torch.Tensor,
    unet_input: torch.Tensor,
    target_mask: torch.Tensor,
    pred_mask: torch.Tensor,
) -> np.ndarray:
    # Ensure all tensors are float32 before converting to numpy
    original_np = original.detach().float().cpu().numpy()
    unet_input_np = unet_input.detach().float().cpu().numpy()
    target_np = target_mask.detach().float().cpu().numpy()
    pred_np = pred_mask.detach().float().cpu().numpy()

    panel = np.concatenate([original_np, unet_input_np, target_np, pred_np], axis=1)
    panel = np.clip(panel, 0.0, 1.0)
    return panel


def _extract_normalized_state_dict(checkpoint: object) -> OrderedDict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        raw_state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        raw_state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        raw_state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format: expected a dict-like state dict.")

    normalized_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in raw_state_dict.items():
        normalized_key = key
        if normalized_key.startswith("_orig_mod."):
            normalized_key = normalized_key[len("_orig_mod.") :]
        if normalized_key.startswith("module."):
            normalized_key = normalized_key[len("module.") :]
        normalized_state_dict[normalized_key] = value

    return normalized_state_dict


def _subject_id_from_sample(sample: dict) -> str:
    image_path = sample["image"]
    if isinstance(image_path, (list, tuple)):
        image_path = image_path[0]
    image_name = Path(str(image_path)).name
    if image_name.endswith(".nii.gz"):
        return image_name[:-7]
    return Path(image_name).stem


def _load_or_create_split(
    samples: list[dict],
    split_file: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict:
    subjects = [_subject_id_from_sample(sample) for sample in samples]
    unique_subjects = sorted(set(subjects))

    split_file.parent.mkdir(parents=True, exist_ok=True)
    if split_file.exists():
        with split_file.open("r", encoding="utf-8") as f:
            split = json.load(f)
    else:
        shuffled = unique_subjects[:]
        rng = random.Random(seed)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_train = max(1, min(n_train, n_total - 2))
        n_val = max(1, min(n_val, n_total - n_train - 1))

        split = {
            "seed": seed,
            "train_subjects": shuffled[:n_train],
            "val_subjects": shuffled[n_train : n_train + n_val],
            "test_subjects": shuffled[n_train + n_val :],
        }
        with split_file.open("w", encoding="utf-8") as f:
            json.dump(split, f, indent=2)

    train_set = set(split["train_subjects"])
    val_set = set(split["val_subjects"])
    test_set = set(split["test_subjects"])

    train_indices, val_indices, test_indices = [], [], []
    for idx, subject in enumerate(subjects):
        if subject in train_set:
            train_indices.append(idx)
        elif subject in val_set:
            val_indices.append(idx)
        elif subject in test_set:
            test_indices.append(idx)

    split["train_indices"] = train_indices
    split["val_indices"] = val_indices
    split["test_indices"] = test_indices
    split["dataset_size"] = len(samples)

    with split_file.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    return split


def _build_generator_guidance(
    *,
    x: torch.Tensor,
    args: argparse.Namespace,
    hist_module: DifferentiableHistogram3D,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    target_hist: torch.Tensor | None = None

    if args.gen_version == "v3":
        target_hist, _, guidance_map = generate_unified_targets(
            input_images=x,
            num_bins=args.num_bins,
            num_chunks=args.num_chunks,
            dark_threshold=args.dark_threshold,
            hist_module=hist_module,
            return_guidance_map=True,
        )
    else:
        target_hist, perms = generate_unified_targets(
            input_images=x,
            num_bins=args.num_bins,
            num_chunks=args.num_chunks,
            dark_threshold=args.dark_threshold,
            hist_module=hist_module,
        )
        guidance_map = create_range_translation_guidance_map(
            input_image=x,
            perms=perms,
            num_chunks=args.num_chunks,
            dark_threshold=args.dark_threshold,
        )

    if args.gen_version != "v1":
        guidance_map = apply_gaussian_blur_3d(guidance_map)

    return guidance_map, target_hist


def main() -> None:
    args = parse_args()
    args.baseline_contrast = normalize_contrast_name(args.baseline_contrast)

    if args.fully_artificial:
        args.use_generator = True
        args.aug_prob_train = 1.0
        args.aug_prob_val = 1.0

    if args.use_generator and args.gen_weights is None:
        args.gen_weights = str(
            PROJECT_ROOT / "checkpoints" / args.gen_version / f"mri_generator_{args.baseline_contrast}_epoch_30.pth"
        )
    device = torch.device(args.device)
    
    use_amp = device.type == "cuda"

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        # OPTIMIZATION: Enable TF32 for up to 3x faster 3D Convolutions on Ampere+ GPUs
        torch.backends.cudnn.allow_tf32 = True

    if args.use_generator and not args.gen_weights:
        raise ValueError("--gen-weights is required when --use-generator is enabled.")

    # Auto-generate run name if not provided
    if args.run_name is None:
        if args.fully_artificial:
            args.run_name = f"fully-artificial-{args.baseline_contrast}-based-segmenter-{args.version}"
        else:
            task = "generator" if args.use_generator else "baseline"
            suffix = f"-{args.version}" if args.use_generator else ""
            args.run_name = f"{task}-{args.baseline_contrast}{suffix}"

    wandb.init(project=args.project_name, name=args.run_name, config=vars(args))

    base_dataset = DecathlonDataset(
        root_dir=args.data_dir,
        task="Task01_BrainTumour",
        transform=get_preprocessing_transforms(
            mode="val",
            patch_size=tuple(args.patch_size),
            source_contrast=args.baseline_contrast,
        ),
        section="training",
        download=True,
        cache_rate=0.0,
        num_workers=args.num_workers,
    )

    split = _load_or_create_split(
        samples=base_dataset.data,
        split_file=Path(args.split_file),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_dataset_full = DecathlonDataset(
        root_dir=args.data_dir,
        task="Task01_BrainTumour",
        transform=get_preprocessing_transforms(
            mode="train",
            patch_size=tuple(args.patch_size),
            source_contrast=args.baseline_contrast,
        ),
        section="training",
        download=True,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )
    val_dataset_full = DecathlonDataset(
        root_dir=args.data_dir,
        task="Task01_BrainTumour",
        transform=get_preprocessing_transforms(
            mode="val",
            patch_size=tuple(args.patch_size),
            source_contrast=args.baseline_contrast,
        ),
        section="training",
        download=True,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )

    dataset = Subset(train_dataset_full, split["train_indices"])
    val_dataset = Subset(val_dataset_full, split["val_indices"])
    
    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        # OPTIMIZATION: Prevents graph recompilation on the last, smaller batch of the epoch
        "drop_last": True,  
    }
    if args.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 2
    dataloader = DataLoader(dataset, **train_loader_kwargs)

    val_loader_kwargs = {
        "batch_size": args.val_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        val_loader_kwargs["persistent_workers"] = True
        val_loader_kwargs["prefetch_factor"] = 2
    val_dataloader = DataLoader(val_dataset, **val_loader_kwargs)

    print(
        f"Using split file: {args.split_file}\n"
        f"Train subjects: {len(split['train_subjects'])} | "
        f"Val subjects: {len(split['val_subjects'])} | "
        f"Test subjects: {len(split['test_subjects'])}"
    )

    # OPTIMIZATION: Push UNet to channels_last_3d memory format for heavily optimized cuDNN kernels
    segmenter = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device=device, memory_format=torch.channels_last_3d)

    if args.compile_model and hasattr(torch, "compile") and device.type == "cuda":
        try:
            segmenter = torch.compile(segmenter)
            print("Successfully wrapped segmenter with torch.compile")
        except Exception as compile_error:
            print(f"torch.compile failed, continuing eagerly: {compile_error}")

    generator = None
    hist_module = None
    if args.use_generator:
        generator = MRI_Synthesis_Net(in_channels=2, out_channels=1).to(device=device, memory_format=torch.channels_last_3d)
        checkpoint = torch.load(args.gen_weights, map_location=device)
        state_dict = _extract_normalized_state_dict(checkpoint)
        generator.load_state_dict(state_dict, strict=True)
        generator.eval()
        for parameter in generator.parameters():
            parameter.requires_grad = False

        hist_module = DifferentiableHistogram3D(num_bins=args.num_bins, value_range=(0.0, 1.0)).to(device)

    loss_fn = DiceCELoss(sigmoid=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    # OPTIMIZATION: Fused AdamW batches kernel updates, reducing CPU-GPU overhead
    optimizer = torch.optim.AdamW(segmenter.parameters(), lr=args.lr, fused=use_amp)
    
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    if args.fully_artificial:
        tag = "fullyartificial"
    else:
        tag = "generator" if args.use_generator else "baseline"

    if args.use_generator:
        if args.gen_version == "v4":
            model_id = f"v4_{tag}_{args.baseline_contrast}"
            checkpoint_dir = PROJECT_ROOT / "checkpoints" / "v4" / model_id
        else:
            checkpoint_dir = PROJECT_ROOT / "checkpoints" / args.version
    else:
        checkpoint_dir = PROJECT_ROOT / "checkpoints" / "baseline"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.use_generator and args.gen_version == "v4":
        best_checkpoint_path = checkpoint_dir / "best_segmenter.pth"
    else:
        best_checkpoint_path = checkpoint_dir / f"best_segmenter_{tag}_{args.baseline_contrast}.pth"

    best_val_dice = float("-inf")

    global_step = 0
    for epoch in range(args.epochs):
        segmenter.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            # OPTIMIZATION: .as_tensor() strips MONAI wrappers. channels_last_3d speeds up memory reading.
            x = batch["image"].as_tensor().to(device=device, memory_format=torch.channels_last_3d).float()
            y = (batch["label"].as_tensor().to(device=device, memory_format=torch.channels_last_3d) > 0).float()

            unet_input = x
            used_generator = False

            if args.use_generator and random.random() < args.aug_prob_train:
                used_generator = True
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                    guidance_map, target_hist = _build_generator_guidance(
                        x=x,
                        args=args,
                        hist_module=hist_module,
                    )
                    generator_input = torch.cat([x, guidance_map], dim=1)
                    generator_output = generator(generator_input)
                    synthesized = ((generator_output + 1.0) * 0.5).clamp(0.0, 1.0)
                    
                    # OPTIMIZATION: Cast back to float32 to prevent dtype graph breaks in torch.compile
                    unet_input = synthesized.to(memory_format=torch.channels_last_3d).float()
                    
            else:
                target_hist = None
                guidance_map = None

            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = segmenter(unet_input)
                loss = loss_fn(logits, y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            global_step += 1

            if batch_idx % args.log_every == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/used_generator": float(used_generator),
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                    step=global_step,
                )

            if batch_idx % args.image_log_every == 0:
                with torch.no_grad():
                    pred_mask = (torch.sigmoid(logits) > 0.5).float()
                    depth_idx = x.shape[2] // 2
                    original_slice = x[0, 0, depth_idx]
                    input_slice = unet_input[0, 0, depth_idx]
                    gt_slice = y[0, 0, depth_idx]
                    pred_slice = pred_mask[0, 0, depth_idx]

                    panel = _build_logging_panel(
                        original=original_slice,
                        unet_input=input_slice,
                        target_mask=gt_slice,
                        pred_mask=pred_slice,
                    )

                    log_data = {
                        "train/mid_axial_panel": wandb.Image(
                            panel,
                            caption="[Original T1w | UNet Input | GT Mask | Pred Mask]",
                        ),
                        "epoch": epoch,
                        "global_step": global_step,
                    }

                    if used_generator and target_hist is not None and guidance_map is not None:
                        log_data["train/generator_used_for_panel"] = 1.0

                    wandb.log(log_data, step=global_step)

        mean_loss = running_loss / max(len(dataloader), 1)

        segmenter.eval()
        val_running_loss = 0.0
        dice_metric.reset()
        with torch.no_grad():
            for val_batch_idx, val_batch in enumerate(val_dataloader):
                # OPTIMIZATION: Same handling as training inputs
                x_val = val_batch["image"].as_tensor().to(device=device, memory_format=torch.channels_last_3d).float()
                y_val = (val_batch["label"].as_tensor().to(device=device, memory_format=torch.channels_last_3d) > 0).float()

                val_input = x_val
                if args.use_generator and random.random() < args.aug_prob_val:
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        guidance_map_val, _ = _build_generator_guidance(
                            x=x_val,
                            args=args,
                            hist_module=hist_module,
                        )
                        generator_input_val = torch.cat([x_val, guidance_map_val], dim=1)
                        generator_output_val = generator(generator_input_val)
                        val_synthesized = ((generator_output_val + 1.0) * 0.5).clamp(0.0, 1.0)
                        val_input = val_synthesized.to(memory_format=torch.channels_last_3d).float()

                with torch.amp.autocast('cuda', enabled=use_amp):
                    val_logits = segmenter(val_input)
                    val_loss = loss_fn(val_logits, y_val)
                    
                val_running_loss += val_loss.item()
                val_pred = (torch.sigmoid(val_logits) > 0.5).float()
                dice_metric(y_pred=val_pred, y=y_val)

                if val_batch_idx == 0 and epoch % args.val_image_log_every == 0:
                    depth_idx_val = x_val.shape[2] // 2
                    val_panel = _build_logging_panel(
                        original=x_val[0, 0, depth_idx_val],
                        unet_input=val_input[0, 0, depth_idx_val],
                        target_mask=y_val[0, 0, depth_idx_val],
                        pred_mask=val_pred[0, 0, depth_idx_val],
                    )
                    wandb.log(
                        {
                            "val/mid_axial_panel": wandb.Image(
                                val_panel,
                                caption="[Val Original T1w | UNet Input | GT Mask | Pred Mask]",
                            ),
                            "epoch": epoch,
                            "global_step": global_step,
                        },
                        step=global_step,
                    )

        mean_val_loss = val_running_loss / max(len(val_dataloader), 1)
        mean_val_dice = float(dice_metric.aggregate().item())
        dice_metric.reset()

        wandb.log(
            {
                "train/epoch_loss": mean_loss,
                "val/loss": mean_val_loss,
                "val/dice": mean_val_dice,
                "epoch": epoch,
            },
            step=global_step,
        )

        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            torch.save(segmenter.state_dict(), best_checkpoint_path)
            print(
                f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
                f"train_loss={mean_loss:.4f} | val_loss={mean_val_loss:.4f} | "
                f"val_dice={mean_val_dice:.4f} | ✓ saved best model → {best_checkpoint_path}"
            )
        else:
            print(
                f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
                f"train_loss={mean_loss:.4f} | val_loss={mean_val_loss:.4f} | val_dice={mean_val_dice:.4f}"
            )

        if args.use_generator and args.gen_version == "v4":
            # Keep a strict rolling buffer of the latest 4 checkpoints.
            slot = ((epoch + 1 - 1) % 4) + 1
            last_checkpoint_path = checkpoint_dir / f"last_segmenter_{slot}.pth"
            torch.save(segmenter.state_dict(), last_checkpoint_path)

            # Defensive cleanup in case stale files exist from older naming schemes.
            for stale_path in checkpoint_dir.glob("last_segmenter_*.pth"):
                stem = stale_path.stem
                try:
                    idx = int(stem.rsplit("_", 1)[-1])
                except ValueError:
                    stale_path.unlink(missing_ok=True)
                    continue
                if idx < 1 or idx > 4:
                    stale_path.unlink(missing_ok=True)

    wandb.finish()


if __name__ == "__main__":
    main()