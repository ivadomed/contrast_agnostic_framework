from __future__ import annotations
import os 

import argparse
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import wandb
from monai.data import DataLoader
from PIL import Image
from torchvision.utils import make_grid

# --- SPEED OPTIMIZATION 1: Multiprocessing File Sharing ---
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

# Setup project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import build_train_dataset
from src.dataset import CONTRAST_TO_INDEX, normalize_contrast_name
from src.generator import MRI_Synthesis_Net
from src.histogram_ops import (
    DifferentiableHistogram3D,
    apply_gaussian_blur_3d,
    create_range_translation_guidance_map,
    generate_unified_targets,
)
from src.losses import (
    DiceEdgeLoss3D,
    DifferentiableWassersteinLoss,
    RangeLoss,
    TotalVariationLoss3D,
    GuidanceLoss3D,
)

from monai.transforms import RandAffine, Rand3DElastic

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 3D MRI contrast synthesis model.")
    parser.add_argument(
        "--source-contrast",
        type=str,
        default="t1w",
        choices=sorted(CONTRAST_TO_INDEX),
        help="Source MRI contrast to train the generator on.",
    )
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-rate", type=float, default=0.0)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--num-bins", type=int, default=128)
    
    # --- ARTIFACT FIX 1: Lower the number of chunks to mimic actual tissue clusters ---
    parser.add_argument("--num-chunks", type=int, default=8)
    
    parser.add_argument("--dark-threshold", type=float, default=0.05)
    parser.add_argument("--base-filters", type=int, default=32)
    parser.add_argument("--wasserstein-weight", type=float, default=0.0)
    parser.add_argument("--tv-weight", type=float, default=2.0)
    parser.add_argument("--edge-weight", type=float, default=20.0)
    parser.add_argument("--range-weight", type=float, default=100.0)
    parser.add_argument("--guidance-weight", type=float, default=20.0)
    parser.add_argument("--guidance-weight-sharp", type=float, default=40.0)
    parser.add_argument("--project-name", type=str, default="mri-synthesis-3d")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--image-log-every", type=int, default=100)
    parser.add_argument("--version", type=str, default="v2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def log_visualizations_to_wandb(
    x: torch.Tensor,
    guidance_map: torch.Tensor,
    synthesized_01: torch.Tensor,
    target_hist: torch.Tensor,
    generated_hist: torch.Tensor,
    *,
    global_step: int,
    epoch: int,
    batch_idx: int,
    max_items: int = 4,
    prefix: str = "train",
) -> None:
    """Log a side-by-side slice grid and histogram overlay to Weights & Biases."""
    with torch.no_grad():
        b = min(max_items, x.shape[0])
        depth_idx = x.shape[2] // 2

        x_slices = x[:b, :, depth_idx].detach().float().cpu().clamp(0.0, 1.0)
        g_slices = guidance_map[:b, :, depth_idx].detach().float().cpu().clamp(0.0, 1.0)
        s_slices = synthesized_01[:b, :, depth_idx].detach().float().cpu().clamp(0.0, 1.0)

        triplets = []
        for idx in range(b):
            triplets.extend([x_slices[idx], g_slices[idx], s_slices[idx]])
        triplet_tensor = torch.stack(triplets, dim=0)

        grid = make_grid(triplet_tensor, nrow=3, padding=2)
        grid_np = grid.permute(1, 2, 0).numpy()
        if grid_np.shape[-1] == 1:
            grid_np = grid_np[..., 0]

        target_1d = target_hist[0, 0].detach().float().cpu().numpy()
        generated_1d = generated_hist[0, 0].detach().float().cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(target_1d, label="target_hist", linewidth=2)
        ax.plot(generated_1d, label="generated_hist", linewidth=2)
        ax.set_title("Target vs Generated Histogram")
        ax.set_xlabel("Bin")
        ax.set_ylabel("Density")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=140)
        buffer.seek(0)
        with Image.open(buffer) as hist_image:
            hist_image = hist_image.convert("RGB")
            wandb.log(
                {
                    f"{prefix}/slice_grid": wandb.Image(
                        grid_np,
                        caption="Rows: sample 0..N | Cols: [Original | Guidance | Synthesized]",
                    ),
                    f"{prefix}/hist_overlay": wandb.Image(
                        hist_image,
                        caption="target_hist vs generated_hist (sample 0)",
                    ),
                    "epoch": epoch,
                    "global_step": global_step,
                    "batch_idx": batch_idx,
                },
                step=global_step,
            )

        plt.close(fig)
        buffer.close()


def main() -> None:
    args = parse_args()
    args.source_contrast = normalize_contrast_name(args.source_contrast)
    device = torch.device(args.device)
    use_amp = device.type == "cuda"

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        # --- SPEED OPTIMIZATION 2: Enable TF32 for Ampere+ GPUs ---
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Initialize Checkpoint Directory
    version_dir = PROJECT_ROOT / "checkpoints" / args.version
    version_dir.mkdir(parents=True, exist_ok=True)

    # Auto-generate run name if not provided
    if args.run_name is None:
        args.run_name = f"generator-{args.source_contrast}-{args.version}"

    wandb.init(project=args.project_name, name=args.run_name, config=vars(args))

    dataset = build_train_dataset(
        data_dir=args.data_dir,
        patch_size=tuple(args.patch_size),
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
        source_contrast=args.source_contrast,
    )
    
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        # --- SPEED OPTIMIZATION 3: Drop last batch to keep dimensions static for compilation ---
        "drop_last": True, 
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    dataloader = DataLoader(dataset, **loader_kwargs)

    model = MRI_Synthesis_Net(in_channels=2, out_channels=1, base_filters=args.base_filters).to(device)

    # --- SPEED OPTIMIZATION 4: torch.compile the generator ---
    if hasattr(torch, "compile") and device.type == "cuda":
        print("Compiling the generator to speed up training...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed, continuing eagerly: {e}")

    histogram_module = DifferentiableHistogram3D(num_bins=args.num_bins, value_range=(0.0, 1.0)).to(device)

    # Initialize Losses
    wasserstein_loss_fn = DifferentiableWassersteinLoss(dark_threshold=args.dark_threshold).to(device)
    edge_loss_fn = DiceEdgeLoss3D().to(device)
    tv_loss_fn = TotalVariationLoss3D().to(device)
    range_loss_fn = RangeLoss(min_value=-1.0, max_value=1.0).to(device)
    guidance_loss_fn = GuidanceLoss3D(kernel_size=5, sigma=2.0).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # --- SPEED OPTIMIZATION 5: Automatic Mixed Precision Scaler ---
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    global_step = 0
    for epoch in range(args.epochs):
        
        gpu_affine = RandAffine(
            prob=0.8,
            rotate_range=(0.35, 0.35, 0.35),
            scale_range=(0.2, 0.2, 0.2),
            mode="bilinear",
            padding_mode="border",
            device=device,
        )
        
        gpu_elastic = Rand3DElastic(
            prob=0.5,
            sigma_range=(10, 13),
            magnitude_range=(0, 1000),
            mode="bilinear",
            padding_mode="border",
            device=device,
        )
    
    
        model.train()
        epoch_loss = 0.0

            
        for batch_idx, batch in enumerate(dataloader):
            # 1. Load clean cropped patch to GPU
            x = batch["image"].to(device).float()
            optimizer.zero_grad(set_to_none=True)

            # --- NEW: APPLY SPATIAL AUGMENTATIONS ON GPU ---
            with torch.no_grad():
                # Strip MetaTensor metadata to prevent affine tracking crashes
                x_pure = x.as_tensor() 

                augmented_items = []
                # Loop through the batch dimension
                for i in range(x_pure.shape[0]):
                    item = x_pure[i]
                    item = gpu_affine(item)
                    item = gpu_elastic(item)
                    augmented_items.append(item)

                # Re-stack into a batch and clamp
                x = torch.stack(augmented_items).clamp(0.0, 1.0)
                
            
            # 1. Generate target histogram via chunk shuffling (Kept in FP32 for precision)
            target_hist, perms = generate_unified_targets(
                input_images=x,
                num_bins=args.num_bins,
                num_chunks=args.num_chunks,
                dark_threshold=args.dark_threshold,
                hist_module=histogram_module,
            )
            
            # 2. Map original intensities to new chunks spatially (Kept in FP32)
            guidance_map = create_range_translation_guidance_map(
                input_image=x,
                perms=perms,
                num_chunks=args.num_chunks,
                dark_threshold=args.dark_threshold,
            )

            # --- ARTIFACT FIX 2: Pre-blur the guidance map to remove "cliffs" ---
            with torch.no_grad():
                blurred_guidance = apply_gaussian_blur_3d(guidance_map, kernel_size=5, sigma=2.0)

            # 3. Forward + Losses (Wrapped in AMP Autocast block)
            with torch.amp.autocast('cuda', enabled=use_amp):
                model_input = torch.cat([x, blurred_guidance], dim=1)
                synthesized = model(model_input) 

                synthesized_01 = ((synthesized + 1.0) * 0.5).clamp(0.0, 1.0)
                generated_hist = histogram_module(synthesized_01)

                wasserstein_loss = wasserstein_loss_fn(generated_hist, target_hist)
                edge_loss = edge_loss_fn(synthesized_01, x)
                tv_loss = tv_loss_fn(synthesized)
                range_loss = range_loss_fn(synthesized)
                
                # 1. The original blurred guidance loss (keeps macro-regions accurate)
                guidance_loss_blurred = guidance_loss_fn(synthesized_01, guidance_map)
                
                # 2. NEW: The sharp guidance constraint (closes the spiking loophole)
                # We use torch.nn.functional.l1_loss directly on the unblurred tensors
                guidance_loss_sharp = torch.nn.functional.l1_loss(synthesized_01, guidance_map)

                # Combine them. 
                # (e.g., 20.0 for blurred to drive the main colors, 10.0 for sharp to prevent spikes)
                total_guidance_loss = (args.guidance_weight * guidance_loss_blurred) + (args.guidance_weight_sharp * guidance_loss_sharp)

                total_loss = (
                    args.wasserstein_weight * wasserstein_loss
                    + args.edge_weight * edge_loss
                    + args.tv_weight * tv_loss
                    + args.range_weight * range_loss
                    + total_guidance_loss  # <-- Use the combined guidance loss
                )

            # 4. Scaled Backward Pass
            scaler.scale(total_loss).backward()
            
            # Unscale gradients before clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            global_step += 1

            # 5. Logging
            if batch_idx % args.log_every == 0:
                wandb.log(
                    {
                        "train/total_loss": total_loss.item(),
                        "train/wasserstein_loss": wasserstein_loss.item(),
                        "train/edge_loss": edge_loss.item(),
                        "train/tv_loss": tv_loss.item(),
                        "train/range_loss": range_loss.item(),
                        "train/guidance_loss_blurred": guidance_loss_blurred.item(),
                        "train/guidance_loss_sharp": guidance_loss_sharp.item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                    step=global_step,
                )

            if batch_idx % args.image_log_every == 0:
                log_visualizations_to_wandb(
                    x=x,
                    guidance_map=blurred_guidance, # Log the blurred guidance map you are passing
                    synthesized_01=synthesized_01,
                    target_hist=target_hist,
                    generated_hist=generated_hist,
                    global_step=global_step,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    max_items=min(4, x.shape[0]),
                    prefix="train",
                )

        # Step the learning rate scheduler
        scheduler.step()
        
        mean_epoch_loss = epoch_loss / max(len(dataloader), 1)
        wandb.log({"train/epoch_loss": mean_epoch_loss, "epoch": epoch}, step=global_step)
        print(f"Epoch {epoch + 1:03d}/{args.epochs:03d} | loss={mean_epoch_loss:.4f} | LR={scheduler.get_last_lr()[0]:.6f}")

        # Model Checkpointing
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            save_path = version_dir / f"mri_generator_{args.source_contrast}_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")

    wandb.finish()

if __name__ == "__main__":
    main()