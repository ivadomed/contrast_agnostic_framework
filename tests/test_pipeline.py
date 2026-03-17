from __future__ import annotations

import sys
from pathlib import Path
import torch

# Setup project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import MRI_Synthesis_Net
from src.histogram_ops import (
    DifferentiableHistogram3D,
    create_range_translation_guidance_map,
    generate_unified_targets,
)
from src.losses import (
    DiceEdgeLoss3D,
    DifferentiableWassersteinLoss,
    RangeLoss,
    TotalVariationLoss3D,
)

def test_full_pipeline():
    print("Starting 3D Pipeline Test...")
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Define Hyperparameters
    batch_size = 2
    channels = 1
    # Using a 64x64x64 patch to ensure it runs on most hardware during testing
    # You can bump this to 96 or 128 to test your specific GPU's limits
    depth, height, width = 64, 64, 64 
    num_bins = 64
    num_chunks = 8
    dark_threshold = 0.05

    # 3. Create Dummy Data
    # torch.rand generates values in [0, 1), simulating our normalized MRI inputs
    print(f"Generating dummy input tensor of shape ({batch_size}, {channels}, {depth}, {height}, {width})...")
    dummy_input = torch.rand((batch_size, channels, depth, height, width), device=device)

    # 4. Initialize Modules
    print("Initializing Generator and Modules...")
    model = MRI_Synthesis_Net(in_channels=2, out_channels=1, base_filters=16).to(device)
    histogram_module = DifferentiableHistogram3D(num_bins=num_bins, value_range=(0.0, 1.0)).to(device)

    wasserstein_loss_fn = DifferentiableWassersteinLoss(dark_threshold=dark_threshold).to(device)
    edge_loss_fn = DiceEdgeLoss3D().to(device)
    tv_loss_fn = TotalVariationLoss3D().to(device)
    range_loss_fn = RangeLoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ==========================================
    # FORWARD PASS
    # ==========================================
    print("\n--- Forward Pass ---")
    
    # Generate Target Histogram and Permutations
    print("1. Generating unified targets...")
    target_hist, perms = generate_unified_targets(
        input_images=dummy_input,
        num_bins=num_bins,
        num_chunks=num_chunks,
        dark_threshold=dark_threshold,
        hist_module=histogram_module,
    )
    assert target_hist.shape == (batch_size, channels, num_bins), f"Unexpected target_hist shape: {target_hist.shape}"

    # Generate Guidance Map
    print("2. Creating guidance map...")
    guidance_map = create_range_translation_guidance_map(
        input_image=dummy_input,
        perms=perms,
        num_chunks=num_chunks,
        dark_threshold=dark_threshold,
    )
    assert guidance_map.shape == dummy_input.shape, f"Unexpected guidance_map shape: {guidance_map.shape}"

    # Generator Inference
    print("3. Running Generator...")
    model_input = torch.cat([dummy_input, guidance_map], dim=1)
    synthesized = model(model_input)
    assert synthesized.shape == dummy_input.shape, f"Unexpected synthesized shape: {synthesized.shape}"
    
    # Denormalize output for histogram and edge operations
    synthesized_01 = ((synthesized + 1.0) * 0.5).clamp(0.0, 1.0)
    generated_hist = histogram_module(synthesized_01)

    # Calculate Losses
    print("4. Calculating Losses...")
    wasserstein_loss = wasserstein_loss_fn(generated_hist, target_hist)
    edge_loss = edge_loss_fn(synthesized_01, dummy_input)
    tv_loss = tv_loss_fn(synthesized)
    range_loss = range_loss_fn(synthesized)

    total_loss = wasserstein_loss + edge_loss + tv_loss + range_loss
    
    print(f"   Wasserstein Loss: {wasserstein_loss.item():.4f}")
    print(f"   Edge Loss:        {edge_loss.item():.4f}")
    print(f"   TV Loss:          {tv_loss.item():.4f}")
    print(f"   Range Loss:       {range_loss.item():.4f}")
    print(f"   Total Loss:       {total_loss.item():.4f}")

    assert not torch.isnan(total_loss), "Total loss resulted in NaN!"

    # ==========================================
    # BACKWARD PASS
    # ==========================================
    print("\n--- Backward Pass ---")
    optimizer.zero_grad()
    total_loss.backward()
    
    # Check if gradients flow correctly
    has_gradients = any(p.grad is not None for p in model.parameters())
    assert has_gradients, "No gradients computed! The computation graph might be broken."
    
    optimizer.step()
    print("Backward pass successful. Gradients flowed through the network.")

    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\nMax GPU Memory used: {max_mem:.2f} MB")

    print("\n✅ PIPELINE TEST PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_full_pipeline()