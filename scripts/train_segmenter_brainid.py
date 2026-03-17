from __future__ import annotations

import argparse
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

# Add Brain-ID to sys.path
BRAIN_ID_DIR = PROJECT_ROOT / "additional_contents" / "Brain-ID"
if str(BRAIN_ID_DIR) not in sys.path:
    sys.path.insert(0, str(BRAIN_ID_DIR))

from src.dataset import build_train_dataset, get_preprocessing_transforms

# Import Brain-ID utilities
import utils.misc as brainid_misc
from BrainID.models import build_feat_model
from utils.checkpoint import load_checkpoint

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')


import os

class BrainIDWrapper(torch.nn.Module):
    def __init__(self, ckp_path: str, device: torch.device):
        super().__init__()
        
        # Safely change the Current Working Directory to the Brain-ID folder
        # so their internal YAML parsers don't crash on relative paths.
        original_cwd = os.getcwd()
        os.chdir(BRAIN_ID_DIR)
        
        try:
            # Now we can just use their native relative paths
            default_cfg_file = 'cfgs/default_train.yaml'
            default_data_file = 'cfgs/default_dataset.yaml'
            submit_cfg_file = 'cfgs/submit.yaml'

            args = brainid_misc.preprocess_cfg([default_cfg_file, default_data_file, submit_cfg_file])
            
            # PyTorch 2.6+ fix: allow numpy multiarray scalar for this checkpoint
            if hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([np.core.multiarray.scalar])

            _, model, _, _, _ = build_feat_model(args, device=device)
            load_checkpoint(ckp_path, [model], None, ['model'], to_print=False)
            
            self.model = model
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        finally:
            # Always return to your project root!
            os.chdir(original_cwd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Brain-ID expects a list of dictionaries
        samples = [{'input': x}]
        outputs, _ = self.model(samples)
        synthesized = outputs[0]['image']
        return synthesized.clamp(0.0, 1.0)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3D UNet segmenter on BraTS Task01 with Brain-ID augmentation.")
    parser.add_argument("--use-generator", action="store_true", help="Enable Brain-ID contrast augmentation.")
    parser.add_argument("--brainid-weights", type=str, default=str(BRAIN_ID_DIR / "assets" / "brain_id_pretrained.pth"), help="Path to pre-trained Brain-ID checkpoint.")
    parser.add_argument("--aug-prob", type=float, default=0.7, help="Probability of applying Brain-ID augmentation.")

    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--cache-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-file", type=str, default=str(PROJECT_ROOT / "splits" / "brats_subject_split.json"))

    parser.add_argument("--project-name", type=str, default="brats-segmenter")
    parser.add_argument("--run-name", type=str, default="brainid-augmentation-run")
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
    original_np = original.detach().float().cpu().numpy()
    unet_input_np = unet_input.detach().float().cpu().numpy()
    target_np = target_mask.detach().float().cpu().numpy()
    pred_np = pred_mask.detach().float().cpu().numpy()

    panel = np.concatenate([original_np, unet_input_np, target_np, pred_np], axis=1)
    panel = np.clip(panel, 0.0, 1.0)
    return panel


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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    
    use_amp = device.type == "cuda"

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True

    if args.use_generator and not args.brainid_weights:
        raise ValueError("--brainid-weights is required when --use-generator is enabled.")

    wandb.init(project=args.project_name, name=args.run_name, config=vars(args))

    base_dataset = DecathlonDataset(
        root_dir=args.data_dir,
        task="Task01_BrainTumour",
        transform=get_preprocessing_transforms(mode="val", patch_size=tuple(args.patch_size)),
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
        transform=get_preprocessing_transforms(mode="train", patch_size=tuple(args.patch_size)),
        section="training",
        download=True,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )
    val_dataset_full = DecathlonDataset(
        root_dir=args.data_dir,
        task="Task01_BrainTumour",
        transform=get_preprocessing_transforms(mode="val", patch_size=tuple(args.patch_size)),
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

    brainid_wrapper = None
    if args.use_generator:
        brainid_wrapper = BrainIDWrapper(ckp_path=args.brainid_weights, device=device).to(device)

    loss_fn = DiceCELoss(sigmoid=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    optimizer = torch.optim.AdamW(segmenter.parameters(), lr=args.lr, fused=use_amp)
    
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_dice = float("-inf")

    global_step = 0
    for epoch in range(args.epochs):
        segmenter.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            x = batch["image"].as_tensor().to(device=device, memory_format=torch.channels_last_3d).float()
            y = (batch["label"].as_tensor().to(device=device, memory_format=torch.channels_last_3d) > 0).float()

            unet_input = x
            used_generator = False

            if args.use_generator and random.random() < args.aug_prob and brainid_wrapper is not None:
                used_generator = True
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                    synthesized = brainid_wrapper(x)
                    unet_input = synthesized.to(memory_format=torch.channels_last_3d).float()
                    
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
                            caption="[Original T1w | Brain-ID Input | GT Mask | Pred Mask]",
                        ),
                        "epoch": epoch,
                        "global_step": global_step,
                    }

                    if used_generator:
                        log_data["train/generator_used_for_panel"] = 1.0

                    wandb.log(log_data, step=global_step)

        mean_loss = running_loss / max(len(dataloader), 1)

        segmenter.eval()
        val_running_loss = 0.0
        dice_metric.reset()
        with torch.no_grad():
            for val_batch_idx, val_batch in enumerate(val_dataloader):
                x_val = val_batch["image"].as_tensor().to(device=device, memory_format=torch.channels_last_3d).float()
                y_val = (val_batch["label"].as_tensor().to(device=device, memory_format=torch.channels_last_3d) > 0).float()

                val_input = x_val
                if args.use_generator and random.random() < args.aug_prob and brainid_wrapper is not None:
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        val_synthesized = brainid_wrapper(x_val)
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
                                caption="[Val Original T1w | Brain-ID Input | GT Mask | Pred Mask]",
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
            tag = "brainid_augmented" if args.use_generator else "baseline"
            save_path = checkpoint_dir / f"best_segmenter_{tag}.pth"
            torch.save(segmenter.state_dict(), save_path)
            print(
                f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
                f"train_loss={mean_loss:.4f} | val_loss={mean_val_loss:.4f} | "
                f"val_dice={mean_val_dice:.4f} | ✓ saved best model → {save_path}"
            )
        else:
            print(
                f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
                f"train_loss={mean_loss:.4f} | val_loss={mean_val_loss:.4f} | val_dice={mean_val_dice:.4f}"
            )

    wandb.finish()


if __name__ == "__main__":
    main()