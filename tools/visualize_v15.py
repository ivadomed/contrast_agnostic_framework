import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path

def generate_v15_2d(
    input_image: torch.Tensor, 
    num_chunks: int = 4, 
    grid_size: tuple = (2, 2), 
    dark_threshold: float = 0.01,
    interpolate: bool = True,
    seed: int = 42
):
    b, c, h, w = input_image.shape
    gy, gx = grid_size
    
    # 1. Pad the image so it is perfectly divisible by the grid size
    pad_h = (gy - (h % gy)) % gy
    pad_w = (gx - (w % gx)) % gx
    x_pad = F.pad(input_image, (0, pad_w, 0, pad_h), mode="replicate")
    
    # 2. Reshape into local spatial blocks
    bh = x_pad.shape[2] // gy
    bw = x_pad.shape[3] // gx
    block_values = (
        x_pad.view(b, c, gy, bh, gx, bw)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(b, gy * gx, -1)
    )
    
    # 3. Mask out the background
    block_values = block_values.clone()
    block_values[block_values <= dark_threshold] = float("nan")
    
    # 4. Calculate local quantiles (thresholds per zone)
    q_probs = torch.linspace(0.0, 1.0, num_chunks + 1, device=input_image.device, dtype=torch.float32)
    local_edges = torch.nanquantile(block_values, q_probs, dim=-1).to(input_image.dtype).permute(1, 0, 2)
    
    local_edges = torch.nan_to_num(local_edges, nan=float(dark_threshold))
    local_edges[:, 0, :] = torch.clamp(local_edges[:, 0, :], max=dark_threshold)
    local_edges[:, -1, :] = torch.clamp(local_edges[:, -1, :], min=1.0)
    
    # 5. Distribute thresholds across the image (Interpolated vs Hard Blocks)
    local_edges_grid = local_edges.view(b, num_chunks + 1, gy, gx)
    
    mode = "bilinear" if interpolate else "nearest"
    align_corners = True if interpolate else None
    
    dense_edges = F.interpolate(
        local_edges_grid,
        size=(h, w),
        mode=mode,
        align_corners=align_corners,
    )
    dense_edges = torch.cummax(dense_edges, dim=1).values
    
    # 6. Hard Bin Assignment
    x_vals = input_image.squeeze(1)
    bin_idx = torch.sum(x_vals.unsqueeze(1) > dense_edges, dim=1) - 1
    bin_idx = torch.clamp(bin_idx, 0, num_chunks - 1)
    
    # 7. Non-Monotonic Random Target Assignment
    # We use a manual seed here so the colors stay exactly the same across comparisons!
    torch.manual_seed(seed)
    random_targets = torch.rand((b, num_chunks), device=input_image.device, dtype=input_image.dtype)
    target_vals = torch.gather(random_targets, 1, bin_idx.reshape(b, -1)).reshape(b, h, w)
    
    # 8. Strict Background Masking
    mapped_img = target_vals.unsqueeze(1).expand(-1, c, -1, -1)
    guidance_map = torch.where(input_image > dark_threshold, mapped_img, torch.zeros_like(mapped_img))
    guidance_map = guidance_map.clamp(0.0, 1.0)
    
    return guidance_map, bin_idx


def draw_grid_lines(ax, shape, grid_size):
    """Draws red dashed lines to show the exact boundaries of the zones."""
    h, w = shape
    gy, gx = grid_size
    for i in range(1, gy):
        ax.axhline(i * h / gy, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    for j in range(1, gx):
        ax.axvline(j * w / gx, color='red', linestyle='--', linewidth=1.5, alpha=0.8)


def main(file_path: str):
    print(f"Loading {file_path}...")
    nii = nib.load(file_path)
    data = nii.get_fdata()
    
    if data.ndim == 4:
        slice_2d = data[:, :, data.shape[2] // 2, 0]
    else:
        slice_2d = data[:, :, data.shape[2] // 2]
        
    slice_2d = np.rot90(slice_2d)

    # Normalize
    slice_min, slice_max = slice_2d.min(), slice_2d.max()
    slice_2d = (slice_2d - slice_min) / (slice_max - slice_min + 1e-8)
    image_tensor = torch.tensor(slice_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    h, w = slice_2d.shape

    print("Generating comparisons...")
    # 1. One Zone (Global)
    g_1x1, b_1x1 = generate_v15_2d(image_tensor, grid_size=(1, 1), interpolate=False)
    # 2. Four Zones (No Interpolation)
    g_2x2_hard, b_2x2_hard = generate_v15_2d(image_tensor, grid_size=(2, 2), interpolate=False)
    # 3. Four Zones (Interpolated)
    g_2x2_smooth, b_2x2_smooth = generate_v15_2d(image_tensor, grid_size=(2, 2), interpolate=True)
    # 4. Sixteen Zones (Interpolated - Standard v15)
    g_4x4_smooth, b_4x4_smooth = generate_v15_2d(image_tensor, grid_size=(4, 4), interpolate=True)

    # Plotting Setup
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle("v15 Generation: Grid Size & Interpolation Comparison", fontsize=20, y=0.98)
    
    # Col 0: Original
    axes[0, 0].imshow(slice_2d, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Original MRI", fontsize=14)
    axes[0, 0].axis("off")
    axes[1, 0].axis("off") # Leave bottom left empty

    configs = [
        ("1 Zone (1x1)\nGlobal Thresholds", b_1x1, g_1x1, (1, 1)),
        ("4 Zones (2x2)\nNO Interpolation (Hard Blocks)", b_2x2_hard, g_2x2_hard, (2, 2)),
        ("4 Zones (2x2)\nWITH Interpolation (Blobs)", b_2x2_smooth, g_2x2_smooth, (2, 2)),
        ("16 Zones (4x4)\nWITH Interpolation (v15 Default)", b_4x4_smooth, g_4x4_smooth, (4, 4))
    ]

    for col_idx, (title, bin_map, guide_map, grid_shape) in enumerate(configs, start=1):
        # Top Row: Bin Assignments
        ax_bin = axes[0, col_idx]
        ax_bin.imshow(bin_map[0].numpy(), cmap="tab10", vmin=0, vmax=3)
        ax_bin.set_title(title + "\n\nBin Assignments", fontsize=12)
        ax_bin.axis("off")
        draw_grid_lines(ax_bin, (h, w), grid_shape)

        # Bottom Row: Final Guidance Map
        ax_guide = axes[1, col_idx]
        ax_guide.imshow(guide_map[0, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax_guide.set_title("Final Guidance Map", fontsize=12)
        ax_guide.axis("off")
        draw_grid_lines(ax_guide, (h, w), grid_shape)

    plt.tight_layout()
    
    # Save the output
    out_dir = Path("tools/outputs/v15_2d_visualization")
    out_dir.mkdir(parents=True, exist_ok=True)
    input_filename = Path(file_path).name.replace(".nii.gz", "").replace(".nii", "")
    out_file = out_dir / f"v15_comparison_{input_filename}.png"
    
    plt.savefig(out_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Comparison successfully saved to: {out_file}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, help="Path to a BraTS .nii.gz file")
    args = parser.parse_args()
    main(args.file_path)