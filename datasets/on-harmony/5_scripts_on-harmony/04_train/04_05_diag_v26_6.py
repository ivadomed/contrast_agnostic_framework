"""
Quick diagnostic: load V26_6 fold2 checkpoint, run forward on a synthesized sample,
check what the model actually predicts.
"""
import os, sys, torch
import numpy as np

sys.path.insert(0, "/home/ge.polymtl.ca/pahoa/mri_synthesis_project")

from pathlib import Path
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 1. Load checkpoint ────────────────────────────────────────────────────────
_results = Path(os.environ.get("nnUNet_results", ""))
CKPT = str(
    _results / "v26_6_20260530_120921"
    / "Dataset030_OnHarmonyT1w"
    / "nnUNetTrainerOnHarmonyV26_6__nnUNetPlans__3d_fullres"
    / "fold_2/checkpoint_final.pth"
)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"Epoch: {ck['current_epoch']}")

# ── 2. Rebuild network ────────────────────────────────────────────────────────
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import LabelManager

import json
from pathlib import Path
_pre = Path(os.environ.get("nnUNet_preprocessed", "")) / "Dataset030_OnHarmonyT1w"
plans_file = str(_pre / "nnUNetPlans.json")
with open(plans_file) as f:
    plans = json.load(f)

dataset_json_file = str(_pre / "dataset.json")
with open(dataset_json_file) as f:
    dataset_json = json.load(f)

plans_manager = PlansManager(plans)
conf = plans_manager.get_configuration("3d_fullres")
label_manager = plans_manager.get_label_manager(dataset_json)

network = get_network_from_plans(
    conf.network_arch_class_name,
    conf.network_arch_init_kwargs,
    conf.network_arch_init_kwargs_req_import,
    input_channels=1,
    output_channels=label_manager.num_segmentation_heads,
)
network.load_state_dict(ck["network_weights"])
network = network.to(device).eval()
print(f"Network: {label_manager.num_segmentation_heads} classes")

# ── 3. Load a real val case ───────────────────────────────────────────────────
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

dataset = nnUNetDatasetBlosc2(
    str(_pre / "nnUNetPlans_3d_fullres"),
)
cases = sorted(dataset.identifiers)
cid = cases[0]
print(f"Case: {cid}")

data, seg, _, props = dataset.load_case(cid)
data_np = np.array(data, dtype=np.float32)
seg_np  = np.array(seg,  dtype=np.int16)

data_t = torch.from_numpy(data_np).unsqueeze(0).to(device)  # (1,1,D,H,W)
seg_t  = torch.from_numpy(seg_np).unsqueeze(0).to(device)   # (1,1,D,H,W)

print(f"Data shape: {data_t.shape}, range: [{data_t.min():.2f}, {data_t.max():.2f}]")
print(f"Seg unique values: {seg_t.unique().tolist()}")

# ── 4. Synthesize ─────────────────────────────────────────────────────────────
from src.synthesis.histogram_ops import DifferentiableHistogram3D
from src.synthesis.target_generators import V26_6SignedAlphaTargetGenerator
from src.nnunet.transforms.synth_aug import synthesize_volume, center_crop_pair

generator   = V26_6SignedAlphaTargetGenerator()
hist_module = DifferentiableHistogram3D(64).to(device)

print("Running synthesis (A: full-volume — same as original V26_6 intent)...")
with torch.no_grad():
    synth_z, synth_01 = synthesize_volume(data_t, seg_t, generator, hist_module)

print(f"synth_z range: [{synth_z.min():.2f}, {synth_z.max():.2f}]")
print(f"synth_z brain mean: {synth_z[seg_t > 0].mean():.3f}, std: {synth_z[seg_t > 0].std():.3f}")
print(f"synth_z background (should be 0): unique vals = {synth_z[seg_t == 0].unique().numel()} distinct (want 1)")

# ── 5A. Forward on full-volume synthesis → center crop ────────────────────────
patch_size = (128, 160, 112)
patch, patch_seg = center_crop_pair(synth_z, seg_t, patch_size, n_crops=1)
print("\n[A] Full-vol synthesis → center crop (NOT the training distribution):")
print(f"\nPatch shape: {patch.shape}")

with torch.no_grad():
    from torch.amp import autocast
    with autocast("cuda", enabled=device.type == "cuda"):
        output = network(patch)

logits = output[0] if isinstance(output, (list, tuple)) else output
pred   = logits.argmax(1, keepdim=True)  # (1,1,D,H,W)
print(f"Prediction shape: {pred.shape}")
print(f"Predicted classes: {pred.unique().tolist()}")

# Dice per class vs GT
gt_patch, _ = center_crop_pair(seg_t, seg_t, patch_size, n_crops=1)
print(f"GT classes present: {gt_patch.unique().tolist()}")

for cls in range(1, label_manager.num_segmentation_heads):
    pred_cls = (pred == cls).float()
    gt_cls   = (gt_patch == cls).float()
    intersection = (pred_cls * gt_cls).sum()
    denom = pred_cls.sum() + gt_cls.sum() + 1e-6
    dice = (2 * intersection / denom).item()
    print(f"  Class {cls}: Dice={dice:.3f}  (pred_vox={pred_cls.sum().int()}, gt_vox={gt_cls.sum().int()})")

# ── 5B. PATCH-level synthesis → same as training distribution ────────────────
print("\n[B] Patch-level synthesis (same as training in OnHarmonyBatchPool):")
from src.nnunet.transforms.synth_aug import synthesize_patch

# full-vol min-max → [0,1], crop patch, synthesize patch only
v_min = data_t.min(); v_max = data_t.max()
image_01_full = (data_t - v_min) / (v_max - v_min + 1e-7)
image_01_full = image_01_full.clamp(0.0, 1.0)

# Center crop (same location as above)
from src.nnunet.transforms.synth_aug import center_crop_pair as _ccrop
patch_01_c, patch_seg_c = _ccrop(image_01_full, seg_t, patch_size, n_crops=1)
print(f"patch_01 range: [{patch_01_c.min():.3f}, {patch_01_c.max():.3f}]  (should be [0,1])")

with torch.no_grad():
    synth_z_patch, _ = synthesize_patch(patch_01_c, patch_seg_c, generator, hist_module)

print(f"synth_z_patch brain mean: {synth_z_patch[patch_seg_c > 0].mean():.3f}, std: {synth_z_patch[patch_seg_c > 0].std():.3f}")

with torch.no_grad():
    with autocast("cuda", enabled=device.type == "cuda"):
        output_p = network(synth_z_patch)

logits_p = output_p[0] if isinstance(output_p, (list, tuple)) else output_p
pred_p   = logits_p.argmax(1, keepdim=True)
print(f"Predicted classes: {pred_p.unique().tolist()}")
gt_c, _ = _ccrop(seg_t, seg_t, patch_size, n_crops=1)
for cls in range(1, label_manager.num_segmentation_heads):
    pred_cls = (pred_p == cls).float()
    gt_cls   = (gt_c == cls).float()
    intersection = (pred_cls * gt_cls).sum()
    denom = pred_cls.sum() + gt_cls.sum() + 1e-6
    dice = (2 * intersection / denom).item()
    print(f"  Class {cls}: Dice={dice:.3f}  (pred_vox={pred_cls.sum().int()}, gt_vox={gt_cls.sum().int()})")

# ── 6. Also test on RAW T1w (no synthesis) to compare ────────────────────────
print("\n--- Same test on RAW T1w (no synthesis) ---")
raw_patch, raw_seg = center_crop_pair(data_t, seg_t, patch_size, n_crops=1)
print(f"Raw patch range: [{raw_patch.min():.2f}, {raw_patch.max():.2f}]")

with torch.no_grad():
    with autocast("cuda", enabled=device.type == "cuda"):
        output_raw = network(raw_patch)

logits_raw = output_raw[0] if isinstance(output_raw, (list, tuple)) else output_raw
pred_raw   = logits_raw.argmax(1, keepdim=True)
print(f"Predicted classes on raw T1w: {pred_raw.unique().tolist()}")

for cls in range(1, label_manager.num_segmentation_heads):
    pred_cls = (pred_raw == cls).float()
    gt_cls   = (raw_seg == cls).float()
    intersection = (pred_cls * gt_cls).sum()
    denom = pred_cls.sum() + gt_cls.sum() + 1e-6
    dice = (2 * intersection / denom).item()
    print(f"  Class {cls}: Dice={dice:.3f}")

print("\nDone.")
