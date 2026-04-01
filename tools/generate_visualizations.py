"""
Generate deterministic 3D synthetic MRI volumes from a trained generator.

This script loads a trained generator checkpoint, fetches validation samples,
synthesizes artificial MRI contrasts, and saves them as NIfTI files for
inspection in medical viewers like ITK-SNAP or 3D Slicer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datamodule import BraTSDataModule
from src.generator import MRI_Synthesis_Net
from src.histogram_ops import DifferentiableHistogram3D
from src.lightning_modules import CompiledSynthesisWrapper, _extract_normalized_state_dict

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
)


def _resolve_path(path_like: str) -> Path:
    """Resolve path relative to PROJECT_ROOT if not absolute."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _nested_get(mapping: dict | None, keys: list[str], default=None):
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def _checkpoint_metadata(checkpoint_path: Path, checkpoint: dict | None) -> tuple[str, str, str]:
    hyper = checkpoint.get("hyper_parameters") if isinstance(checkpoint, dict) else None
    if not isinstance(hyper, dict):
        hyper = checkpoint.get("hparams") if isinstance(checkpoint, dict) else None

    version = _nested_get(hyper, ["version"], default=None)
    contrast = _nested_get(hyper, ["data", "source_contrast"], default=None)

    if version is None or contrast is None:
        parts = list(checkpoint_path.resolve().parts)
        if "generator" in parts:
            gen_idx = parts.index("generator")
            if version is None and gen_idx >= 2:
                version = parts[gen_idx - 2]
            if contrast is None and gen_idx + 1 < len(parts):
                contrast = parts[gen_idx + 1]

    if version is None:
        version = "unknown_version"
    if contrast is None:
        contrast = "t1w"

    run_dir = checkpoint_path.parent.name
    model_id = f"{contrast}_{run_dir}"
    return str(version), str(model_id), str(contrast)


def _load_generator(checkpoint_path: Path, cfg: DictConfig) -> tuple[MRI_Synthesis_Net, str, str, dict | None]:
    """Load the generator model from checkpoint."""
    checkpoint_path = _resolve_path(str(checkpoint_path))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    generator = MRI_Synthesis_Net(
        in_channels=int(cfg.model.generator.in_channels),
        out_channels=int(cfg.model.generator.out_channels),
        base_filters=int(cfg.model.generator.base_filters),
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    gen_version, _, checkpoint_contrast = _checkpoint_metadata(checkpoint_path, checkpoint)
    state_dict = _extract_normalized_state_dict(checkpoint)
    generator.load_state_dict(state_dict, strict=True)

    print(f"Loaded generator from: {checkpoint_path}")
    print(f"Detected gen_version: {gen_version}")
    return generator, gen_version, checkpoint_contrast, checkpoint

def _save_volume(volume: torch.Tensor, output_path: Path, affine: torch.Tensor | None = None) -> None:
    """Saves a 3D tensor to NIfTI format properly using nibabel."""
    # Squeeze down to exactly (H, W, D)
    vol_np = volume.detach().cpu().squeeze().numpy()
    
    # Use the provided affine, or default to identity matrix if none exists
    aff_np = affine.detach().cpu().squeeze().numpy() if affine is not None else np.eye(4)
    
    nib_img = nib.Nifti1Image(vol_np, affine=aff_np)
    nib.save(nib_img, str(output_path))


def main(args: argparse.Namespace) -> None:
    seed = int(args.seed)
    pl.seed_everything(seed, workers=True)
    
    checkpoint_path = _resolve_path(str(args.checkpoint))
    config_path = _resolve_path(str(args.config))
    
    config_dir = str(config_path.parent)
    config_name = config_path.stem
    
    with initialize_config_dir(version_base=None, config_dir=config_dir, job_name="visualize"):
        cfg = compose(config_name=config_name)

    # =====================================================================
    # THE FIX: Override config to force val set creation and KILL caching
    # =====================================================================
    with open_dict(cfg):
        cfg.task = "segmenter"           # Forces datamodule to build val_dataset
        cfg.data.cache_rate = 0.0        # Prevents loading the whole dataset into RAM
        cfg.data.num_workers = 0         # Prevents multiprocess hanging for small jobs
        cfg.data.val_batch_size = 1      # Process one by one for clean saving
    
    generator, gen_version, checkpoint_contrast, checkpoint = _load_generator(checkpoint_path, cfg)
    _, model_id, _ = _checkpoint_metadata(checkpoint_path, checkpoint)
    
    # =====================================================================
    # THE FIX: Override config to force val set creation and KILL caching
    # =====================================================================
    with open_dict(cfg):
        cfg.task = "segmenter"           
        cfg.data.cache_rate = 0.0        
        cfg.data.num_workers = 0         
        cfg.data.val_batch_size = 1      
        # FORCE the dataset to use the contrast the model was trained on!
        cfg.data.source_contrast = checkpoint_contrast
        
        
    base_output_dir = PROJECT_ROOT / "results" / "visualizations" / gen_version / model_id
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device).eval()
    
    # Instantiate datamodule with the overridden config
    datamodule = BraTSDataModule(cfg)
    datamodule.setup(stage="fit")
    
    from monai.transforms import Compose

    underlying_dataset = datamodule.val_dataset.dataset
    
    # Grab the original list of transforms from the validation pipeline
    original_transforms = underlying_dataset.transform.transforms
    
    inference_transforms = []
    for t in original_transforms:
        t_name = type(t).__name__
        # Keep everything EXCEPT spatial manipulators
        if not any(keyword in t_name for keyword in ["Crop", "Pad", "Resize"]):
            inference_transforms.append(t)
            
    # Apply the filtered pipeline back to the dataset
    underlying_dataset.transform = Compose(inference_transforms)
    
    val_dataloader = datamodule.val_dataloader()
    
    histogram_module = DifferentiableHistogram3D(
        num_bins=int(cfg.model.generator.num_bins),
        value_range=(0.0, 1.0),
    ).to(device).eval()

    target_generator_cfg = cfg.model.generator.get(
        "target_generator",
        {"_target_": "src.target_generators.LegacyChunkTargetGenerator"},
    )
    guidance_perturber_cfg = cfg.model.generator.get("guidance_perturber", None)
    apply_guidance_blur = bool(cfg.model.generator.get("apply_guidance_blur", True))

    target_generator = instantiate(target_generator_cfg)
    guidance_perturber = instantiate(guidance_perturber_cfg)
    
    synthesis_wrapper = CompiledSynthesisWrapper(
        generator=generator,
        hist_module=histogram_module,
        target_generator=target_generator,
        guidance_perturber=guidance_perturber,
        apply_guidance_blur=apply_guidance_blur,
    ).to(device).eval()
    
    sample_count = 0
    with torch.no_grad(), torch.inference_mode():
        for batch in val_dataloader:
            if sample_count >= args.num_samples:
                break
            
            source = batch["image"].to(device).float()
            label = batch.get("label")
            # MONAI often stores the original affine matrix in the batch dictionary
            affine = batch.get("image_meta_dict", {}).get("affine") 
            
            synthesized = synthesis_wrapper(
                source,
                num_bins=int(cfg.model.generator.num_bins),
                num_chunks=int(cfg.model.generator.num_chunks),
                dark_threshold=float(cfg.model.generator.dark_threshold),
            )
            
            output_dir = base_output_dir / f"sample_{sample_count:03d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            _save_volume(source, output_dir / "source.nii.gz", affine)
            _save_volume(synthesized, output_dir / "synthetic.nii.gz", affine)
            
            if label is not None:
                _save_volume(label.to(device).float(), output_dir / "label.nii.gz", affine)
            
            print(f"Saved sample {sample_count} to: {output_dir}")
            sample_count += 1

    print(f"Successfully generated {sample_count} visualizations in {base_output_dir}")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic MRI volumes.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint")
    parser.add_argument("--config", type=str, default="conf/config.yaml", help="Path to Hydra config")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of subjects to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)