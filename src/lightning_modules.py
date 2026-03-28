from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any
import re

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import wandb
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from omegaconf import DictConfig
from PIL import Image
from torchvision.utils import make_grid

from src.generator import MRI_Synthesis_Net
from src.histogram_ops import (
    DifferentiableHistogram3D,
    apply_gaussian_blur_3d,
    generate_unified_targets,
)
from src.bigaug_augmentations import build_bigaug_augmentation
from src.losses import (
    DiceEdgeLoss3D,
    DifferentiableWassersteinLoss,
    GuidanceLoss3D,
    RangeLoss,
    TotalVariationLoss3D,
)
from src.kornia_augmentations import RandomFourierAmplitude3D, build_kornia_augmentation
from src.filters import AnatomicalUnsharpMask3D
from src.intensity_ops import (
    RandomAnisotropicDegradation3D,
    RandomBezierIntensityWarp,
    RandomGMMHistogramMatching,
    RandomSoftQuantileShuffling,
)
from src.noise_ops import generate_fractal_noise_3d




class CompiledSynthesisWrapper(nn.Module):
    def __init__(self, generator, hist_module, gen_version):
        super().__init__()
        self.generator = generator
        self.hist_module = hist_module
        self.gen_version = gen_version
        self.noise_strength = 0.2
        self.bezier_warp = RandomBezierIntensityWarp(p=1.0)
        self.gmm_hist_match = RandomGMMHistogramMatching(p=1.0)
        self.soft_quantile_shuffle = RandomSoftQuantileShuffling(p=1.0)

    def forward(self, x: torch.Tensor, 
                num_bins: int, num_chunks: int, dark_threshold: float):
        
        if self.gen_version in ("v3", "v4"):
            target_hist, _, guidance_map = generate_unified_targets(
                input_images=x,
                num_bins=num_bins,
                num_chunks=num_chunks,
                dark_threshold=dark_threshold,
                hist_module=self.hist_module,
                return_guidance_map=True,
                gen_version=self.gen_version,
            )
        else:
            target_hist, _, guidance_map = generate_unified_targets(
                input_images=x,
                num_bins=num_bins,
                num_chunks=num_chunks,
                dark_threshold=dark_threshold,
                hist_module=self.hist_module,
                gen_version=self.gen_version,
            )

        if self.gen_version == "v9":
            with torch.no_grad():
                procedural_noise = generate_fractal_noise_3d(
                    guidance_map.detach(),
                    noise_dtype=torch.float16,
                )
            guidance_map = (guidance_map + self.noise_strength * procedural_noise).clamp(0.0, 1.0)

        if self.gen_version == "v11":
            guidance_map = self.bezier_warp(guidance_map)
        if self.gen_version == "v12":
            guidance_map = self.gmm_hist_match(guidance_map)
        if self.gen_version == "v13":
            guidance_map = self.soft_quantile_shuffle(guidance_map)

        if self.gen_version != "v1":
            guidance_map = apply_gaussian_blur_3d(guidance_map)

        generator_input = torch.cat([x, guidance_map], dim=1)
        generator_output = self.generator(generator_input)
        synthesized = ((generator_output + 1.0) * 0.5).clamp(0.0, 1.0)
        return synthesized

class CompiledSegmenterWrapper(nn.Module):
    def __init__(self, segmenter, loss_fn):
        super().__init__()
        self.segmenter = segmenter
        self.loss_fn = loss_fn

    def forward(self, x, y):
        logits = self.segmenter(x)
        loss = self.loss_fn(logits, y)
        return loss

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
        # Iteratively strip prefixes added by PyTorch Lightning, wrapping, or compiling
        hit = True
        while hit:
            hit = False
            for prefix in ["_orig_mod.", "module.", "model.", "compiled_wrapper.", "compiled_synthesis.", "compiled_segmenter."]:
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
                    hit = True
        normalized_state_dict[normalized_key] = value

    return normalized_state_dict


class CompiledLossWrapper(nn.Module):
    def __init__(self, model, histogram_module, wasserstein_loss_fn, edge_loss_fn, tv_loss_fn, range_loss_fn, guidance_loss_fn, gen_version: str):
        super().__init__()
        self.model = model
        self.histogram_module = histogram_module
        self.wasserstein_loss_fn = wasserstein_loss_fn
        self.edge_loss_fn = edge_loss_fn
        self.tv_loss_fn = tv_loss_fn
        self.range_loss_fn = range_loss_fn
        self.guidance_loss_fn = guidance_loss_fn
        self.gen_version = gen_version
        self.noise_strength = 0.2
        self.bezier_warp = RandomBezierIntensityWarp(p=1.0)
        self.gmm_hist_match = RandomGMMHistogramMatching(p=1.0)
        self.soft_quantile_shuffle = RandomSoftQuantileShuffling(p=1.0)

    def forward(self, x: torch.Tensor, 
                num_bins: int, num_chunks: int, dark_threshold: float,
                guidance_blur_k: int, guidance_blur_s: float,
                w_edge: float, w_tv: float, w_range: float, w_wass: float,
                w_guide_blur: float, w_guide_sharp: float) -> dict[str, torch.Tensor]:
        
        # Generate targets once - reuses guidance_map internally
        target_hist, perms, guidance_map = generate_unified_targets(
            input_images=x,
            num_bins=num_bins,
            num_chunks=num_chunks,
            dark_threshold=dark_threshold,
            hist_module=self.histogram_module,
            gen_version=self.gen_version,
        )
        
        guidance_for_generator = guidance_map
        if self.gen_version == "v9":
            with torch.no_grad():
                procedural_noise = generate_fractal_noise_3d(
                    guidance_map.detach(),
                    noise_dtype=torch.float16,
                )
            guidance_for_generator = (guidance_for_generator + self.noise_strength * procedural_noise).clamp(0.0, 1.0)

        if self.gen_version == "v11":
            guidance_for_generator = self.bezier_warp(guidance_for_generator)
        if self.gen_version == "v12":
            guidance_for_generator = self.gmm_hist_match(guidance_for_generator)
        if self.gen_version == "v13":
            guidance_for_generator = self.soft_quantile_shuffle(guidance_for_generator)

        # Stop propagating gradient through guidance map blur
        bg_guidance = guidance_for_generator.detach()
        blurred_guidance = apply_gaussian_blur_3d(
            bg_guidance,
            kernel_size=guidance_blur_k,
            sigma=guidance_blur_s,
        )
        # Cannot conditionally branch on config inside compile well if we want static graph,
        # but x comes in channels_last_3d so simple operations keep it.
        blurred_guidance = blurred_guidance.contiguous(memory_format=torch.channels_last_3d)

        model_input = torch.cat([x, blurred_guidance], dim=1)
        synthesized = self.model(model_input)

        synthesized_01 = ((synthesized + 1.0) * 0.5).clamp(0.0, 1.0)
        generated_hist = self.histogram_module(synthesized_01)

        wasserstein_loss = self.wasserstein_loss_fn(generated_hist, target_hist)
        edge_loss = self.edge_loss_fn(synthesized_01, x)
        tv_loss = self.tv_loss_fn(synthesized)
        range_loss = self.range_loss_fn(synthesized)
        guidance_loss_blurred = self.guidance_loss_fn(synthesized_01, guidance_for_generator)
        guidance_loss_sharp = F.l1_loss(synthesized_01, guidance_for_generator)

        total_guidance_loss = (
            w_guide_blur * guidance_loss_blurred + w_guide_sharp * guidance_loss_sharp
        )
        total_loss = (
            w_wass * wasserstein_loss
            + w_edge * edge_loss
            + w_tv * tv_loss
            + w_range * range_loss
            + total_guidance_loss
        )
        
        return {
            "total_loss": total_loss,
            "wasserstein_loss": wasserstein_loss,
            "edge_loss": edge_loss,
            "tv_loss": tv_loss,
            "range_loss": range_loss,
            "guidance_loss_blurred": guidance_loss_blurred,
            "guidance_loss_sharp": guidance_loss_sharp,
            "blurred_guidance": blurred_guidance,
            "synthesized_01": synthesized_01,
            "target_hist": target_hist,
            "generated_hist": generated_hist
        }


class MRISynthesisLightning(pl.LightningModule):
    """Lightning module for 3D MRI contrast synthesis.

    Args:
        cfg: Hydra configuration.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        if bool(cfg.model.generator.tf32) and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model = MRI_Synthesis_Net(
            in_channels=int(cfg.model.generator.in_channels),
            out_channels=int(cfg.model.generator.out_channels),
            base_filters=int(cfg.model.generator.base_filters),
        )
        if bool(cfg.model.generator.channels_last_3d):
            self.model = self.model.to(memory_format=torch.channels_last_3d)

        self.histogram_module = DifferentiableHistogram3D(
            num_bins=int(cfg.model.generator.num_bins),
            value_range=(0.0, 1.0),
        )
        self.wasserstein_loss_fn = DifferentiableWassersteinLoss(
            dark_threshold=float(cfg.model.generator.dark_threshold)
        )
        self.edge_loss_fn = DiceEdgeLoss3D()
        self.tv_loss_fn = TotalVariationLoss3D()
        self.range_loss_fn = RangeLoss(min_value=-1.0, max_value=1.0)
        self.guidance_loss_fn = GuidanceLoss3D(
            kernel_size=int(cfg.model.generator.guidance_blur.kernel_size),
            sigma=float(cfg.model.generator.guidance_blur.sigma),
        )
        
        self.compiled_wrapper = CompiledLossWrapper(
            self.model, self.histogram_module, self.wasserstein_loss_fn,
            self.edge_loss_fn, self.tv_loss_fn, self.range_loss_fn, self.guidance_loss_fn,
            self._resolved_generator_version(),
        )
        if bool(self.cfg.training.generator.compile_model) and hasattr(torch, "compile"):
            self.compiled_wrapper = torch.compile(self.compiled_wrapper, mode="reduce-overhead")

        self._gpu_aug: nn.Module | None = None
        self._fourier_aug: nn.Module | None = None
        self._anatomical_unsharp: nn.Module | None = None

    def _ensure_gpu_aug(self) -> None:
        if self._gpu_aug is not None:
            return
        self._gpu_aug = build_kornia_augmentation(self.cfg, task="generator").to(self.device)

    def _resolved_generator_version(self) -> str:
        configured = None
        if hasattr(self.cfg.model, "generator") and hasattr(self.cfg.model.generator, "gen_version"):
            configured = self.cfg.model.generator.gen_version
        if configured is None:
            return str(self.cfg.version)
        return str(configured)

    def _uses_fourier_generator(self) -> bool:
        return self._resolved_generator_version() in ("v7", "v8", "v9", "v10", "v11")

    def _uses_unsharp_generator(self) -> bool:
        return self._resolved_generator_version() == "v10"

    def _ensure_unsharp_generator(self) -> None:
        if self._anatomical_unsharp is not None:
            return
        self._anatomical_unsharp = AnatomicalUnsharpMask3D(alpha=2.0, sigma=1.0).to(self.device)

    def _ensure_fourier_aug(self) -> None:
        if self._fourier_aug is not None:
            return
        p = 1.0
        low_freq_ratio = 0.15
        scale_range = (0.5, 1.5)
        version = self._resolved_generator_version()
        if version in ("v8", "v9", "v10", "v11"):
            # v8 keeps Fourier as an occasional regularizer.
            p = 0.3
        if hasattr(self.cfg.model.generator, "fourier"):
            fourier_cfg = self.cfg.model.generator.fourier
            if hasattr(fourier_cfg, "p") and version not in ("v8", "v9", "v10", "v11"):
                p = float(fourier_cfg.p)
            if hasattr(fourier_cfg, "low_freq_ratio"):
                low_freq_ratio = float(fourier_cfg.low_freq_ratio)
            if hasattr(fourier_cfg, "scale_range"):
                scale_range = tuple(float(v) for v in fourier_cfg.scale_range)
        self._fourier_aug = RandomFourierAmplitude3D(
            p=p,
            low_freq_ratio=low_freq_ratio,
            scale_range=scale_range,
        ).to(self.device)

    def on_after_batch_transfer(self, batch: dict[str, Any], dataloader_idx: int) -> dict[str, Any]:
        if "image" not in batch:
            return batch

        image = batch["image"]
        if hasattr(image, "as_tensor"):
            image = image.as_tensor()
        image = image.float()
        if bool(self.cfg.model.generator.channels_last_3d):
            image = image.to(memory_format=torch.channels_last_3d)

        if self.training and bool(self.cfg.training.generator.gpu_aug.enabled):
            self._ensure_gpu_aug()
            if self._gpu_aug is not None:
                with torch.no_grad():
                    # Avoid forcing contiguous here - let Kornia handle the memory format
                    image = self._gpu_aug(image)

                # Ensure output is in desired memory format if it changed
                if bool(self.cfg.model.generator.channels_last_3d):
                    image = image.to(memory_format=torch.channels_last_3d)

        batch["image"] = image
        return batch

    def configure_optimizers(self) -> dict[str, Any]:
        use_fused_adamw = (
            bool(self.cfg.model.generator.fused_adamw)
            and (self.device.type == "cuda")
            and float(self.cfg.training.generator.gradient_clip_val) <= 0.0
        )
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg.training.lr.generator),
            betas=tuple(float(v) for v in self.cfg.training.betas.generator),
            weight_decay=float(self.cfg.training.weight_decay.generator),
            fused=use_fused_adamw,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.cfg.training.max_epochs.generator),
            eta_min=float(self.cfg.training.eta_min),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["image"]
        if bool(self.cfg.model.generator.channels_last_3d):
            x = x.to(memory_format=torch.channels_last_3d)

        if self._uses_fourier_generator():
            self._ensure_fourier_aug()
            if self._fourier_aug is not None:
                x = self._fourier_aug(x)

        if self._uses_unsharp_generator():
            self._ensure_unsharp_generator()
            if self._anatomical_unsharp is not None:
                x = self._anatomical_unsharp(x)

        # Unpack scalars to avoid graphing dictionary/config lookups
        num_bins = int(self.cfg.model.generator.num_bins)
        num_chunks = int(self.cfg.model.generator.num_chunks)
        dark_threshold=float(self.cfg.model.generator.dark_threshold)
        guidance_blur_k=int(self.cfg.model.generator.guidance_blur.kernel_size)
        guidance_blur_s=float(self.cfg.model.generator.guidance_blur.sigma)
        w_edge=float(self.cfg.model.generator.loss_weights.edge)
        w_tv=float(self.cfg.model.generator.loss_weights.tv)
        w_range=float(self.cfg.model.generator.loss_weights.range)
        w_wass=float(self.cfg.model.generator.loss_weights.wasserstein)
        w_guide_blur=float(self.cfg.model.generator.loss_weights.guidance_blurred)
        w_guide_sharp=float(self.cfg.model.generator.loss_weights.guidance_sharp)

        outs = self.compiled_wrapper(
            x, num_bins, num_chunks, dark_threshold, 
            guidance_blur_k, guidance_blur_s,
            w_edge, w_tv, w_range, w_wass, 
            w_guide_blur, w_guide_sharp
        )

        total_loss = outs["total_loss"]
        
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        if batch_idx % int(self.cfg.training.generator.log_aux_every_n_steps) == 0:
            self.log("train/wasserstein_loss", outs["wasserstein_loss"], on_step=True, on_epoch=False)
            self.log("train/edge_loss", outs["edge_loss"], on_step=True, on_epoch=False)
            self.log("train/tv_loss", outs["tv_loss"], on_step=True, on_epoch=False)
            self.log("train/range_loss", outs["range_loss"], on_step=True, on_epoch=False)
            self.log("train/guidance_loss_blurred", outs["guidance_loss_blurred"], on_step=True, on_epoch=False)
            self.log("train/guidance_loss_sharp", outs["guidance_loss_sharp"], on_step=True, on_epoch=False)

        if (
            bool(self.cfg.training.generator.enable_image_logging)
            and self.trainer.is_global_zero
            and self.logger is not None
            and isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run)
            and batch_idx % int(self.cfg.training.generator.image_log_every) == 0
        ):
            self._log_train_images(
                x=x,
                guidance_map=outs["blurred_guidance"],
                synthesized_01=outs["synthesized_01"],
                target_hist=outs["target_hist"],
                generated_hist=outs["generated_hist"],
                batch_idx=batch_idx,
            )

        return total_loss

    def _log_train_images(
        self,
        *,
        x: torch.Tensor,
        guidance_map: torch.Tensor,
        synthesized_01: torch.Tensor,
        target_hist: torch.Tensor,
        generated_hist: torch.Tensor,
        batch_idx: int,
    ) -> None:
        with torch.no_grad():
            max_items = min(4, x.shape[0])
            depth_idx = x.shape[2] // 2

            x_slices = x[:max_items, :, depth_idx].detach().float().cpu().clamp(0.0, 1.0)
            g_slices = guidance_map[:max_items, :, depth_idx].detach().float().cpu().clamp(0.0, 1.0)
            s_slices = synthesized_01[:max_items, :, depth_idx].detach().float().cpu().clamp(0.0, 1.0)

            triplets = []
            for idx in range(max_items):
                triplets.extend([x_slices[idx], g_slices[idx], s_slices[idx]])
            grid_tensor = torch.stack(triplets, dim=0)
            grid = make_grid(grid_tensor, nrow=3, padding=2)
            grid_np = grid.permute(1, 2, 0).numpy()
            if grid_np.shape[-1] == 1:
                grid_np = grid_np[..., 0]

            target_1d = target_hist[0, 0].detach().float().cpu().numpy()
            generated_1d = generated_hist[0, 0].detach().float().cpu().numpy()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(target_1d, label="target_hist", linewidth=2)
            ax.plot(generated_1d, label="generated_hist", linewidth=2)
            ax.set_title("Target vs Generated Histogram")
            ax.legend(loc="best")
            ax.grid(alpha=0.2)
            fig.canvas.draw()

            width, height = fig.canvas.get_width_height()
            hist_np = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
            plt.close(fig)

            self.logger.experiment.log(
                {
                    "train/slice_grid": wandb.Image(grid_np),
                    "train/hist_overlay": wandb.Image(Image.fromarray(hist_np)),
                    "global_step": self.global_step,
                    "batch_idx": batch_idx,
                },
                step=self.global_step,
            )


class MRISegmenterLightning(pl.LightningModule):
    """Lightning module for BraTS 3D tumor segmentation.

    Args:
        cfg: Hydra configuration.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        if bool(cfg.model.segmenter.tf32) and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.segmenter = UNet(
            spatial_dims=3,
            in_channels=int(cfg.model.segmenter.in_channels),
            out_channels=int(cfg.model.segmenter.out_channels),
            channels=tuple(int(c) for c in cfg.model.segmenter.channels),
            strides=tuple(int(s) for s in cfg.model.segmenter.strides),
            num_res_units=int(cfg.model.segmenter.num_res_units),
        )
        if bool(cfg.model.segmenter.channels_last_3d):
            self.segmenter = self.segmenter.to(memory_format=torch.channels_last_3d)

        self.loss_fn = DiceCELoss(sigmoid=True)
        self.compiled_segmenter = CompiledSegmenterWrapper(self.segmenter, self.loss_fn)
        if bool(self.cfg.model.segmenter.compile_model) and hasattr(torch, "compile"):
            self.compiled_segmenter = torch.compile(self.compiled_segmenter, mode="max-autotune-no-cudagraphs")
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")

        self.generator: MRI_Synthesis_Net | None = None
        self.hist_module: DifferentiableHistogram3D | None = None
        self._generator_is_ready = False
        self._gpu_aug: nn.Module | None = None
        self._fourier_aug: nn.Module | None = None
        self._anatomical_unsharp: nn.Module | None = None
        self._anisotropic_degradation: nn.Module | None = None


    def _ensure_gpu_aug(self) -> None:
        if self._gpu_aug is not None:
            return
        if str(self.cfg.task) == "segmenter" and str(self.cfg.version) == "v16_bigaug":
            self._gpu_aug = build_bigaug_augmentation().to(self.device)
            return
        self._gpu_aug = build_kornia_augmentation(self.cfg, task="segmenter").to(self.device)

    def _segmenter_gpu_aug_enabled(self) -> bool:
        if hasattr(self.cfg.training, "segmenter") and hasattr(self.cfg.training.segmenter, "gpu_aug"):
            return bool(self.cfg.training.segmenter.gpu_aug.enabled)
        return bool(self.cfg.training.generator.gpu_aug.enabled)

    def on_after_batch_transfer(self, batch: dict[str, Any], dataloader_idx: int) -> dict[str, Any]:
        if "image" not in batch or "label" not in batch:
            return batch

        image = batch["image"]
        label = batch["label"]
        if hasattr(image, "as_tensor"):
            image = image.as_tensor()
        if hasattr(label, "as_tensor"):
            label = label.as_tensor()
            
        image = image.float()
        label = label.float()

        if image.ndim != 5:
            image = image.unsqueeze(0) if image.ndim == 4 else image
        if label.ndim != 5:
            label = label.unsqueeze(0) if label.ndim == 4 else label

        if bool(self.cfg.model.segmenter.channels_last_3d):
            image = image.to(memory_format=torch.channels_last_3d)

        # Apply augmentation only during training, same as generator
        if self.training and self._segmenter_gpu_aug_enabled():
            self._ensure_gpu_aug()
            if self._gpu_aug is not None:
                with torch.no_grad():
                    image, label = self._gpu_aug(image.contiguous(), label.contiguous())

                if image.ndim != 5:
                    image = image.unsqueeze(0) if image.ndim == 4 else image
                if label.ndim != 5:
                    label = label.unsqueeze(0) if label.ndim == 4 else label

                if bool(self.cfg.model.segmenter.channels_last_3d):
                    image = image.to(memory_format=torch.channels_last_3d)

        batch["image"] = image
        batch["label"] = label
        return batch

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit"):
            return

# Removed standard compile from setup to use CompiledSegmenterWrapper

        if self._generator_is_ready:
            return

        if bool(self.cfg.model.segmenter.fully_artificial):
            self.cfg.model.segmenter.use_generator = True
            self.cfg.model.segmenter.aug_prob_train = 1.0
            self.cfg.model.segmenter.aug_prob_val = 1.0

        if not bool(self.cfg.model.segmenter.use_generator):
            return

        generator_weights = self.cfg.model.segmenter.gen_weights
        if generator_weights is None:
            project_root = Path(__file__).resolve().parents[1]
            contrast = self.cfg.data.source_contrast
            gen_version = self._resolved_segmenter_gen_version()
            generator_root = project_root / "checkpoints" / str(gen_version) / "generator" / str(contrast)

            run_pattern = re.compile(r"^run(\d+)$")
            latest_ckpt: Path | None = None
            if generator_root.exists():
                run_dirs = []
                for child in generator_root.iterdir():
                    if not child.is_dir():
                        continue
                    match = run_pattern.match(child.name)
                    if match is None:
                        continue
                    run_dirs.append((int(match.group(1)), child))
                if run_dirs:
                    latest_run_dir = max(run_dirs, key=lambda item: item[0])[1]
                    candidate = latest_run_dir / "last.ckpt"
                    if candidate.exists():
                        latest_ckpt = candidate

            if latest_ckpt is None:
                legacy_candidate = generator_root / "last.ckpt"
                latest_ckpt = legacy_candidate

            generator_weights = str(latest_ckpt)
            print(f"Loading generator from: {generator_weights}")

        self.generator = MRI_Synthesis_Net(in_channels=2, out_channels=1)
        if bool(self.cfg.model.segmenter.channels_last_3d):
            self.generator = self.generator.to(memory_format=torch.channels_last_3d)

        checkpoint = torch.load(generator_weights, map_location="cpu", weights_only=False)
        state_dict = _extract_normalized_state_dict(checkpoint)
        self.generator.load_state_dict(state_dict, strict=True)
        self.generator.eval()
        for p in self.generator.parameters():
            p.requires_grad = False
        self.hist_module = DifferentiableHistogram3D(
            num_bins=int(self.cfg.model.segmenter.num_bins),
            value_range=(0.0, 1.0),
        )
        self.compiled_synthesis = CompiledSynthesisWrapper(
            self.generator,
            self.hist_module,
            self._resolved_segmenter_gen_version(),
        )
        if bool(self.cfg.model.segmenter.compile_model) and hasattr(torch, "compile"):
            self.compiled_synthesis = torch.compile(self.compiled_synthesis, mode="reduce-overhead")
        self._generator_is_ready = True

    def _resolved_segmenter_gen_version(self) -> str:
        configured = self.cfg.model.segmenter.gen_version
        if configured is None and hasattr(self.cfg.model, "generator") and hasattr(self.cfg.model.generator, "gen_version"):
            if self.cfg.model.generator.gen_version is not None:
                configured = self.cfg.model.generator.gen_version
        if configured is None:
            return str(self.cfg.version)
        return str(configured)

    def _segmenter_uses_fourier_generator(self) -> bool:
        return self._resolved_segmenter_gen_version() in ("v7", "v8", "v9", "v10", "v11")

    def _segmenter_uses_unsharp_generator(self) -> bool:
        return self._resolved_segmenter_gen_version() == "v10"

    def _segmenter_uses_anisotropic_degradation(self) -> bool:
        return self._resolved_segmenter_gen_version() in ("v11", "v12", "v13", "v14", "v15")

    def _segmenter_uses_consistency_regularization(self) -> bool:
        return self._resolved_segmenter_gen_version() in ("v13", "v14", "v15")

    def _ensure_segmenter_unsharp(self) -> None:
        if self._anatomical_unsharp is not None:
            return
        self._anatomical_unsharp = AnatomicalUnsharpMask3D(alpha=2.0, sigma=1.0).to(self.device)

    def _ensure_segmenter_anisotropic_degradation(self) -> None:
        if self._anisotropic_degradation is not None:
            return
        self._anisotropic_degradation = RandomAnisotropicDegradation3D(
            p=0.5,
            min_factor=4,
            max_factor=8,
        ).to(self.device)

    def _ensure_segmenter_fourier_aug(self) -> None:
        if self._fourier_aug is not None:
            return
        p = 1.0
        low_freq_ratio = 0.15
        scale_range = (0.5, 1.5)
        resolved_version = self._resolved_segmenter_gen_version()
        if resolved_version in ("v8", "v9", "v10", "v11"):
            # v8 keeps Fourier as an occasional regularizer.
            p = 0.3
        if hasattr(self.cfg.model.segmenter, "fourier"):
            fourier_cfg = self.cfg.model.segmenter.fourier
            if hasattr(fourier_cfg, "p") and resolved_version not in ("v8", "v9", "v10", "v11"):
                p = float(fourier_cfg.p)
            if hasattr(fourier_cfg, "low_freq_ratio"):
                low_freq_ratio = float(fourier_cfg.low_freq_ratio)
            if hasattr(fourier_cfg, "scale_range"):
                scale_range = tuple(float(v) for v in fourier_cfg.scale_range)
        self._fourier_aug = RandomFourierAmplitude3D(
            p=p,
            low_freq_ratio=low_freq_ratio,
            scale_range=scale_range,
        ).to(self.device)

    def _to_plain_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(x, "as_tensor"):
            x = x.as_tensor()
        x = x.float()
        if bool(self.cfg.model.segmenter.channels_last_3d):
            x = x.to(memory_format=torch.channels_last_3d)
        return x

    def _build_generator_guidance(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self.hist_module is not None
        target_hist: torch.Tensor | None = None
        gen_version = self._resolved_segmenter_gen_version()

        if gen_version in ("v3", "v4"):
            target_hist, _, guidance_map = generate_unified_targets(
                input_images=x,
                num_bins=int(self.cfg.model.segmenter.num_bins),
                num_chunks=int(self.cfg.model.segmenter.num_chunks),
                dark_threshold=float(self.cfg.model.segmenter.dark_threshold),
                hist_module=self.hist_module,
                return_guidance_map=True,
                gen_version=gen_version,
            )
        else:
            # Now generate_unified_targets always returns guidance_map
            target_hist, perms, guidance_map = generate_unified_targets(
                input_images=x,
                num_bins=int(self.cfg.model.segmenter.num_bins),
                num_chunks=int(self.cfg.model.segmenter.num_chunks),
                dark_threshold=float(self.cfg.model.segmenter.dark_threshold),
                hist_module=self.hist_module,
                gen_version=gen_version,
            )

        if gen_version != "v1":
            guidance_map = apply_gaussian_blur_3d(guidance_map)

        return guidance_map, target_hist

    def _maybe_apply_generator(self, x: torch.Tensor, prob: float) -> tuple[torch.Tensor, bool]:
        use_generator = bool(self.cfg.model.segmenter.use_generator) and self.generator is not None
        apply_generator = bool(torch.rand((), device=x.device) < float(prob))
        if not use_generator or not apply_generator:
            return x, False

        if self._segmenter_uses_fourier_generator():
            self._ensure_segmenter_fourier_aug()
            if self._fourier_aug is not None:
                x = self._fourier_aug(x)

        if self._segmenter_uses_unsharp_generator():
            self._ensure_segmenter_unsharp()
            if self._anatomical_unsharp is not None:
                x = self._anatomical_unsharp(x)

        # Synthesis pass doesn't need gradients for segmenter
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
            
        with torch.inference_mode():
            synthesized = self.compiled_synthesis(
                x,
                num_bins=int(self.cfg.model.segmenter.num_bins),
                num_chunks=int(self.cfg.model.segmenter.num_chunks),
                dark_threshold=float(self.cfg.model.segmenter.dark_threshold)
            )

        if self._segmenter_uses_anisotropic_degradation():
            self._ensure_segmenter_anisotropic_degradation()
            if self._anisotropic_degradation is not None:
                synthesized = self._anisotropic_degradation(synthesized)

        if bool(self.cfg.model.segmenter.channels_last_3d):
            synthesized = synthesized.contiguous(memory_format=torch.channels_last_3d)
        return synthesized.clone().float(), True

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.segmenter.parameters(),
            lr=float(self.cfg.training.lr.segmenter),
            betas=tuple(float(v) for v in self.cfg.training.betas.segmenter),
            weight_decay=float(self.cfg.training.weight_decay.segmenter),
            fused=(self.device.type == "cuda"),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.cfg.training.max_epochs.segmenter) if hasattr(self.cfg.training.max_epochs, "segmenter") else int(self.cfg.training.max_epochs),
            eta_min=float(self.cfg.training.eta_min),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
            
        x = self._to_plain_tensor(batch["image"])
        y = (self._to_plain_tensor(batch["label"]) > 0).float()
        if bool(self.cfg.model.segmenter.channels_last_3d):
            y = y.to(memory_format=torch.channels_last_3d)

        unet_input, used_generator = self._maybe_apply_generator(
            x,
            prob=float(self.cfg.model.segmenter.aug_prob_train),
        )

        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        train_logits_for_logging: torch.Tensor | None = None
        if self._segmenter_uses_consistency_regularization() and used_generator:
            x_combined = torch.cat([x, unet_input], dim=0)
            logits_combined = self.segmenter(x_combined)
            logits_raw, logits_synth = torch.chunk(logits_combined, 2, dim=0)

            supervised_loss = self.loss_fn(logits_synth, y)

            eps = torch.finfo(logits_synth.dtype).eps
            raw_prob = torch.sigmoid(logits_raw.detach()).clamp(min=eps, max=1.0 - eps)
            synth_prob = torch.sigmoid(logits_synth).clamp(min=eps, max=1.0 - eps)

            # Memory-lean Bernoulli KL: KL(p_raw || p_synth) without channel concatenation.
            consistency_map = raw_prob * (torch.log(raw_prob) - torch.log(synth_prob))
            consistency_map = consistency_map + (1.0 - raw_prob) * (
                torch.log(1.0 - raw_prob) - torch.log(1.0 - synth_prob)
            )
            consistency_loss = consistency_map.mean()
            consistency_weight = float(self.cfg.training.segmenter.get("consistency_loss_weight", 0.1))
            loss = supervised_loss + consistency_weight * consistency_loss
            self.log("train/loss_supervised", supervised_loss, on_step=True, on_epoch=False)
            self.log("train/loss_consistency", consistency_loss, on_step=True, on_epoch=False)
            train_logits_for_logging = logits_synth.detach()
        else:
            loss = self.compiled_segmenter(unet_input, y)

        if (
            self.trainer.is_global_zero
            and self.logger is not None
            and isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run)
            and bool(self.cfg.training.segmenter.enable_train_image_logging)
            and int(self.cfg.training.segmenter.train_image_log_every) > 0
            and (self.global_step % int(self.cfg.training.segmenter.train_image_log_every) == 0)
        ):
            with torch.no_grad():
                if train_logits_for_logging is None:
                    train_logits = self.segmenter(unet_input)
                else:
                    train_logits = train_logits_for_logging
                train_pred = (torch.sigmoid(train_logits) > 0.5).float()
            self._log_segmenter_train_images(
                raw_input=x,
                train_input=unet_input,
                y=y,
                train_pred=train_pred,
                batch_idx=batch_idx,
            )

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/used_generator", float(used_generator), on_step=True, on_epoch=False)
        return loss

    def _log_segmenter_train_images(
        self,
        *,
        raw_input: torch.Tensor,
        train_input: torch.Tensor,
        y: torch.Tensor,
        train_pred: torch.Tensor,
        batch_idx: int,
    ) -> None:
        with torch.no_grad():
            max_items = min(4, train_input.shape[0])
            depth_idx = train_input.shape[2] // 2

            raw_input_slices = raw_input[:max_items, :1, depth_idx].detach().float().cpu().clamp(0.0, 1.0)
            train_input_slices = train_input[:max_items, :1, depth_idx].detach().float().cpu().clamp(0.0, 1.0)
            target_slices = y[:max_items, :1, depth_idx].detach().float().cpu().clamp(0.0, 1.0)
            pred_slices = train_pred[:max_items, :1, depth_idx].detach().float().cpu().clamp(0.0, 1.0)

            rows = []
            for idx in range(max_items):
                rows.extend([
                    raw_input_slices[idx],
                    train_input_slices[idx],
                    target_slices[idx],
                    pred_slices[idx],
                ])

            grid_tensor = torch.stack(rows, dim=0)
            grid = make_grid(grid_tensor, nrow=4, padding=2)
            grid_np = grid.permute(1, 2, 0).numpy()
            if grid_np.shape[-1] == 1:
                grid_np = grid_np[..., 0]

            self.logger.experiment.log(
                {
                    "train/slice_grid_raw_model_target_pred": wandb.Image(grid_np),
                    "train/epoch": int(self.current_epoch) + 1,
                    "global_step": self.global_step,
                    "batch_idx": batch_idx,
                },
                step=self.global_step,
            )

    def on_validation_epoch_start(self) -> None:
        self.dice_metric.reset()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = self._to_plain_tensor(batch["image"])
        y = (self._to_plain_tensor(batch["label"]) > 0).float()
        if bool(self.cfg.model.segmenter.channels_last_3d):
            y = y.to(memory_format=torch.channels_last_3d)

        val_input, _ = self._maybe_apply_generator(
            x,
            prob=float(self.cfg.model.segmenter.aug_prob_val),
        )

        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        # Keep validation eager to avoid retaining compiled graph outputs for metric state.
        logits = self.segmenter(val_input)
        val_loss = self.loss_fn(logits, y)
        val_pred = (torch.sigmoid(logits) > 0.5).float()
        self.dice_metric(y_pred=val_pred, y=y)

        if (
            self.trainer.is_global_zero
            and self.logger is not None
            and isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run)
            and batch_idx == 0
            and int(self.cfg.training.segmenter.val_image_log_every) > 0
            and ((int(self.current_epoch) + 1) % int(self.cfg.training.segmenter.val_image_log_every) == 0)
        ):
            self._log_segmenter_val_images(val_input=val_input, y=y, val_pred=val_pred, batch_idx=batch_idx)

        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return val_loss

    def _log_segmenter_val_images(
        self,
        *,
        val_input: torch.Tensor,
        y: torch.Tensor,
        val_pred: torch.Tensor,
        batch_idx: int,
    ) -> None:
        with torch.no_grad():
            max_items = min(4, val_input.shape[0])
            depth_idx = val_input.shape[2] // 2

            input_slices = val_input[:max_items, :1, depth_idx].detach().float().cpu().clamp(0.0, 1.0)
            target_slices = y[:max_items, :1, depth_idx].detach().float().cpu().clamp(0.0, 1.0)
            pred_slices = val_pred[:max_items, :1, depth_idx].detach().float().cpu().clamp(0.0, 1.0)

            triplets = []
            for idx in range(max_items):
                triplets.extend([input_slices[idx], target_slices[idx], pred_slices[idx]])
            grid_tensor = torch.stack(triplets, dim=0)
            grid = make_grid(grid_tensor, nrow=3, padding=2)
            grid_np = grid.permute(1, 2, 0).numpy()
            if grid_np.shape[-1] == 1:
                grid_np = grid_np[..., 0]

            self.logger.experiment.log(
                {
                    "val/slice_grid": wandb.Image(grid_np),
                    "val/epoch": int(self.current_epoch) + 1,
                    "global_step": self.global_step,
                    "batch_idx": batch_idx,
                },
                step=self.global_step,
            )

    def on_validation_epoch_end(self) -> None:
        mean_val_dice = float(self.dice_metric.aggregate().item())
        self.log("val/dice", mean_val_dice, on_epoch=True, prog_bar=True, sync_dist=True)
        self.dice_metric.reset()
