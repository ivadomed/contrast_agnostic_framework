from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any
import re

import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
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
)
from src.guidance_perturbers import BaseGuidancePerturber, IdentityGuidancePerturber
from src.target_generators import BaseTargetGenerator


def _get_hydra_choice(group_name: str) -> str | None:
    try:
        choices = HydraConfig.get().runtime.choices
        choice = choices.get(group_name, None)
        if choice is not None:
            return str(choice)
    except Exception:
        return None
    return None




class CompiledSynthesisWrapper(nn.Module):
    def __init__(
        self,
        generator: nn.Module,
        hist_module: DifferentiableHistogram3D,
        target_generator: BaseTargetGenerator,
        guidance_perturber: BaseGuidancePerturber | None,
        apply_guidance_blur: bool,
    ):
        super().__init__()
        self.generator = generator
        self.hist_module = hist_module
        self.target_generator = target_generator
        self.guidance_perturber = guidance_perturber or IdentityGuidancePerturber()
        self.apply_guidance_blur = bool(apply_guidance_blur)

    def forward(self, x: torch.Tensor, 
                num_bins: int, num_chunks: int, dark_threshold: float, labels: torch.Tensor | None = None):
        _, _, guidance_map = self.target_generator(
            input_images=x,
            num_bins=num_bins,
            num_chunks=num_chunks,
            dark_threshold=dark_threshold,
            hist_module=self.hist_module,
            return_guidance_map=True,
            labels=labels,
        )

        guidance_map = self.guidance_perturber(guidance_map)

        if self.apply_guidance_blur:
            guidance_map = apply_gaussian_blur_3d(guidance_map)

        generator_input = torch.cat([x, guidance_map], dim=1)
        generator_output = self.generator(generator_input)
        synthesized = ((generator_output + 1.0) * 0.5).clamp(0.0, 1.0)
        bg_mask = x[:, :1] < dark_threshold
        synthesized = torch.where(bg_mask, torch.zeros_like(synthesized), synthesized)
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
    def __init__(
        self,
        model,
        histogram_module,
        wasserstein_loss_fn,
        edge_loss_fn,
        tv_loss_fn,
        range_loss_fn,
        guidance_loss_fn,
        target_generator: BaseTargetGenerator,
        guidance_perturber: BaseGuidancePerturber | None,
        apply_guidance_blur: bool,
    ):
        super().__init__()
        self.model = model
        self.histogram_module = histogram_module
        self.wasserstein_loss_fn = wasserstein_loss_fn
        self.edge_loss_fn = edge_loss_fn
        self.tv_loss_fn = tv_loss_fn
        self.range_loss_fn = range_loss_fn
        self.guidance_loss_fn = guidance_loss_fn
        self.target_generator = target_generator
        self.guidance_perturber = guidance_perturber or IdentityGuidancePerturber()
        self.apply_guidance_blur = bool(apply_guidance_blur)

    def forward(self, x: torch.Tensor, 
                num_bins: int, num_chunks: int, dark_threshold: float,
                guidance_blur_k: int, guidance_blur_s: float,
                w_edge: float, w_tv: float, w_range: float, w_wass: float,
                w_guide_blur: float, w_guide_sharp: float, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        
        target_hist, perms, guidance_map = self.target_generator(
            input_images=x,
            num_bins=num_bins,
            num_chunks=num_chunks,
            dark_threshold=dark_threshold,
            hist_module=self.histogram_module,
            return_guidance_map=True,
            labels=labels,
        )

        guidance_for_generator = self.guidance_perturber(guidance_map)

        # Stop propagating gradient through guidance map blur
        bg_guidance = guidance_for_generator.detach()
        if self.apply_guidance_blur:
            blurred_guidance = apply_gaussian_blur_3d(
                bg_guidance,
                kernel_size=guidance_blur_k,
                sigma=guidance_blur_s,
            )
        else:
            blurred_guidance = bg_guidance
        # Cannot conditionally branch on config inside compile well if we want static graph,
        # but x comes in channels_last_3d so simple operations keep it.
        blurred_guidance = blurred_guidance.contiguous(memory_format=torch.channels_last_3d)

        model_input = torch.cat([x, blurred_guidance], dim=1)
        synthesized = self.model(model_input)

        synthesized_01 = ((synthesized + 1.0) * 0.5).clamp(0.0, 1.0)
        bg_mask = x[:, :1] < dark_threshold
        synthesized_01 = torch.where(bg_mask, torch.zeros_like(synthesized_01), synthesized_01)
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

        target_generator = instantiate(self._generator_target_generator_cfg())
        guidance_perturber = instantiate(self._generator_guidance_perturber_cfg())
        
        self.compiled_wrapper = CompiledLossWrapper(
            self.model, self.histogram_module, self.wasserstein_loss_fn,
            self.edge_loss_fn, self.tv_loss_fn, self.range_loss_fn, self.guidance_loss_fn,
            target_generator,
            guidance_perturber,
            self._generator_apply_guidance_blur(),
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

    def _default_target_generator_cfg(self, version: str) -> dict[str, Any]:
        version = str(version)
        if version in ("v8", "v9", "v10", "v11"):
            return {"_target_": "src.target_generators.V8GridTargetGenerator", "grid_size": [4, 4, 4]}
        if version == "v15":
            return {
                "_target_": "src.target_generators.V15GridTargetGenerator",
                "grid_size": [4, 4, 4],
                "background_threshold": 0.01,
            }
        if version == "v17_micro_anchor":
            return {
                "_target_": "src.target_generators.V17MicroAnchorTargetGenerator",
                "tau": 0.05,
                "num_peaks": 4,
                "background_threshold": 0.01,
            }
        return {"_target_": "src.target_generators.LegacyChunkTargetGenerator"}

    def _generator_target_generator_cfg(self) -> Any:
        if hasattr(self.cfg.model.generator, "target_generator") and self.cfg.model.generator.target_generator is not None:
            return self.cfg.model.generator.target_generator
        return self._default_target_generator_cfg(self._resolved_generator_version())

    def _generator_guidance_perturber_cfg(self) -> Any:
        if hasattr(self.cfg.model.generator, "guidance_perturber"):
            return self.cfg.model.generator.guidance_perturber
        return None

    def _generator_apply_guidance_blur(self) -> bool:
        if hasattr(self.cfg.model.generator, "apply_guidance_blur"):
            return bool(self.cfg.model.generator.apply_guidance_blur)
        return self._resolved_generator_version() not in ("v1", "v17_micro_anchor")

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
            w_guide_blur, w_guide_sharp, labels=batch.get("label", None)
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
        
        # Load pretrained checkpoint if provided
        pretrained_ckpt_path = getattr(cfg.model.segmenter, 'pretrained_ckpt_path', None)
        if pretrained_ckpt_path is not None:
            self._load_pretrained_checkpoint(str(pretrained_ckpt_path))
        
        # Freeze encoder if requested
        freeze_encoder = getattr(cfg.model.segmenter, 'freeze_encoder', False)
        if freeze_encoder:
            self._freeze_encoder()
        
        if bool(cfg.model.segmenter.channels_last_3d):
            self.segmenter = self.segmenter.to(memory_format=torch.channels_last_3d)

        self._num_seg_classes = int(cfg.model.segmenter.out_channels)
        self._is_multiclass = self._num_seg_classes > 1
        if self._is_multiclass:
            self.loss_fn = DiceCELoss(
                to_onehot_y=True,
                softmax=True,
                include_background=False,
            )
            self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        else:
            self.loss_fn = DiceCELoss(sigmoid=True)
            self.dice_metric = DiceMetric(include_background=False, reduction="mean")

        self.compiled_segmenter = CompiledSegmenterWrapper(self.segmenter, self.loss_fn)
        if bool(self.cfg.model.segmenter.compile_model) and hasattr(torch, "compile"):
            self.compiled_segmenter = torch.compile(self.compiled_segmenter, mode="max-autotune-no-cudagraphs")

        self.generator: MRI_Synthesis_Net | None = None
        self.hist_module: DifferentiableHistogram3D | None = None
        self._generator_is_ready = False
        self._gpu_aug: nn.Module | None = None
        self._fourier_aug: nn.Module | None = None
        self._anatomical_unsharp: nn.Module | None = None
        self._anisotropic_degradation: nn.Module | None = None

    def _load_pretrained_checkpoint(self, checkpoint_path: str) -> None:
        """Load pretrained segmenter weights from a checkpoint file."""
        print(f"Loading pretrained segmenter checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = _extract_normalized_state_dict(checkpoint)
        
        # The state dict may have "segmenter." prefix from Lightning; extract just the UNet weights
        segmenter_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("segmenter."):
                # Strip the "segmenter." prefix
                new_key = key[len("segmenter."):]
                segmenter_state_dict[new_key] = value
            elif not key.startswith("generator."):
                # Include keys without generator prefix (these are our UNet weights)
                segmenter_state_dict[key] = value

        model_state = self.segmenter.state_dict()
        compatible_state_dict = OrderedDict()
        skipped_shape = []
        skipped_missing = []
        for key, value in segmenter_state_dict.items():
            if key not in model_state:
                skipped_missing.append(key)
                continue
            if model_state[key].shape != value.shape:
                skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
                continue
            compatible_state_dict[key] = value

        missing_after_load, unexpected_after_load = self.segmenter.load_state_dict(
            compatible_state_dict,
            strict=False,
        )

        print(
            "Pretrained checkpoint loaded with compatibility filtering: "
            f"loaded={len(compatible_state_dict)}, "
            f"skipped_shape={len(skipped_shape)}, "
            f"skipped_missing={len(skipped_missing)}, "
            f"missing_after_load={len(missing_after_load)}, "
            f"unexpected_after_load={len(unexpected_after_load)}"
        )
        if skipped_shape:
            for key, src_shape, dst_shape in skipped_shape[:8]:
                print(f"  - shape mismatch skipped: {key} src={src_shape} dst={dst_shape}")

    def _freeze_encoder(self) -> None:
        """Freeze the encoder blocks of the U-Net segmenter."""
        # MONAI UNet names down-path params as:
        # - model.0.* (first down block)
        # - *.submodule.0.* (recursive down blocks)
        # Keep bottleneck/decoder trainable.
        num_frozen = 0
        for name, param in self.segmenter.named_parameters():
            if name.startswith("model.0.") or ".submodule.0." in name:
                param.requires_grad = False
                num_frozen += 1
        print(f"Froze {num_frozen} encoder parameters")

    def _ensure_gpu_aug(self) -> None:
        if self._gpu_aug is not None:
            return
        task = str(getattr(self.cfg, "task", ""))
        version = str(getattr(self.cfg, "version", ""))
        if task == "segmenter":
            if version == "v16_bigaug":
                self._gpu_aug = build_bigaug_augmentation(self.cfg).to(self.device)
                return
            if version == "v20_1" or version == "v20":
                from src.intensity_ops import PartialSynthSegAugmentation3D
                class BaselineAugmentationWrapper(nn.Module):
                    def __init__(self, cfg, device):
                        super().__init__()
                        self.partial_aug = PartialSynthSegAugmentation3D(sigma=1.0).to(device)
                        self.kornia_aug = build_kornia_augmentation(cfg, task="segmenter").to(device)
                    def forward(self, images, labels):
                        # First replace tumor intensities, smoothly blending boundaries via PSF simulation
                        y, l = self.partial_aug(images, labels)
                        # Then apply standard downsampling (low-res thick slice) and geometric augs
                        return self.kornia_aug(y, l)
                self._gpu_aug = BaselineAugmentationWrapper(self.cfg, self.device)
                return
            if version == "v21":
                from src.intensity_ops import SparseSynthSegAugmentation3D
                class SparseSynthSegWrapper(nn.Module):
                    def __init__(self, cfg, device):
                        super().__init__()
                        self.sparse_synth_aug = SparseSynthSegAugmentation3D(sigma=1.5).to(device)
                        self.kornia_aug = build_kornia_augmentation(cfg, task="segmenter").to(device)
                    def forward(self, images, labels):
                        y, l = self.sparse_synth_aug(images, labels)
                        return self.kornia_aug(y, l)
                self._gpu_aug = SparseSynthSegWrapper(self.cfg, self.device)
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
                # Stash the pre-augmentation scan so image logging can show a
                # true before/after comparison (e.g. SparseSynthSegAugmentation3D).
                batch["raw_image"] = image.contiguous().clone()
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
            dataset_name = str(getattr(self.cfg.data, "name", "unknown_dataset"))
            contrast = self.cfg.data.source_contrast
            gen_version = self._resolved_segmenter_gen_version()
            gen_choice = _get_hydra_choice("generator") or str(getattr(getattr(self.cfg, "generator", None), "name", gen_version))
            generator_roots = [
                project_root / "checkpoints" / dataset_name / f"{dataset_name}_{gen_choice}" / "generator" / str(contrast),
                project_root / "checkpoints" / dataset_name / f"{dataset_name}_{gen_version}" / "generator" / str(contrast),
                project_root / "checkpoints" / dataset_name / str(gen_version) / "generator" / str(contrast),
                project_root / "checkpoints" / str(gen_version) / "generator" / str(contrast),
                project_root / "checkpoints" / "generator" / str(contrast),
            ]

            run_pattern = re.compile(r"^run(\d+)$")
            latest_ckpt: Path | None = None
            for generator_root in generator_roots:
                if not generator_root.exists():
                    continue
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
                        break

                legacy_candidate = generator_root / "last.ckpt"
                if legacy_candidate.exists():
                    latest_ckpt = legacy_candidate
                    break

            if latest_ckpt is None:
                latest_ckpt = generator_roots[0] / "last.ckpt"

            generator_weights = str(latest_ckpt)
            print(f"Loading generator from: {generator_weights}")

        if not Path(generator_weights).exists():
            project_root = Path(__file__).resolve().parents[1]
            dataset_name = str(getattr(self.cfg.data, "name", "unknown_dataset"))
            gen_version = self._resolved_segmenter_gen_version()
            generator_base = project_root / "checkpoints" / str(gen_version) / "generator"
            available_contrasts: list[str] = []
            if generator_base.exists():
                available_contrasts = sorted([p.name for p in generator_base.iterdir() if p.is_dir()])
            raise FileNotFoundError(
                "Missing generator checkpoint for segmenter synthesis path. "
                f"Expected checkpoint: {generator_weights}. "
                f"Requested source contrast: {self.cfg.data.source_contrast}. "
                f"Dataset: {dataset_name}. Available generator contrasts for {gen_version}: {available_contrasts}. "
                "Either run generator training first for the requested contrast or switch to an available contrast/version."
            )

        self.generator = MRI_Synthesis_Net(
            in_channels=2,
            out_channels=1,
            base_filters=int(self.cfg.model.generator.base_filters),
        )
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
            instantiate(self._segmenter_target_generator_cfg()),
            instantiate(self._segmenter_guidance_perturber_cfg()),
            bool(self._segmenter_apply_guidance_blur()),
        )
        compile_synthesis = bool(getattr(self.cfg.model.segmenter, "compile_synthesis", self.cfg.model.segmenter.compile_model))
        if compile_synthesis and hasattr(torch, "compile"):
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

    def _segmenter_target_generator_cfg(self) -> Any:
        if hasattr(self.cfg.model.segmenter, "target_generator") and self.cfg.model.segmenter.target_generator is not None:
            return self.cfg.model.segmenter.target_generator
        if hasattr(self.cfg.model.generator, "target_generator") and self.cfg.model.generator.target_generator is not None:
            return self.cfg.model.generator.target_generator
        return self._default_target_generator_cfg(self._resolved_segmenter_gen_version())

    def _segmenter_guidance_perturber_cfg(self) -> Any:
        if hasattr(self.cfg.model.segmenter, "guidance_perturber"):
            return self.cfg.model.segmenter.guidance_perturber
        if hasattr(self.cfg.model.generator, "guidance_perturber"):
            return self.cfg.model.generator.guidance_perturber
        return None

    def _segmenter_apply_guidance_blur(self) -> bool:
        if hasattr(self.cfg.model.segmenter, "apply_guidance_blur"):
            return bool(self.cfg.model.segmenter.apply_guidance_blur)
        if hasattr(self.cfg.model.generator, "apply_guidance_blur"):
            return bool(self.cfg.model.generator.apply_guidance_blur)
        return self._resolved_segmenter_gen_version() not in ("v1", "v17_micro_anchor")

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

    def _prepare_segmentation_target(self, label: torch.Tensor) -> torch.Tensor:
        y = self._to_plain_tensor(label).long()
        if y.ndim == 4:
            y = y.unsqueeze(1)
        elif y.ndim == 5 and y.shape[1] > 1:
            # Convert one-hot-like targets back to class indices expected by DiceCELoss(to_onehot_y=True).
            y = torch.argmax(y, dim=1, keepdim=True)
        elif y.ndim > 5:
            y = y.reshape(y.shape[0], 1, *y.shape[-3:])

        label_mapping = getattr(self.cfg.data, "label_mapping", None)
        if label_mapping is not None:
            for src, dst in dict(label_mapping).items():
                y = torch.where(y == int(src), torch.full_like(y, int(dst)), y)
        if self._is_multiclass:
            y = y.clamp(min=0, max=self._num_seg_classes - 1)
            return y
        return (y > 0).float()

    def _align_target_to_input(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if any(int(dim) <= 0 for dim in y.shape[2:]):
            # Some converted Spider samples can carry degenerate label tensors (e.g. W=0).
            # Replace with an empty-background target matching the model input shape.
            if self._is_multiclass:
                return torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device, dtype=torch.long)
            return torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device, dtype=x.dtype)
        if y.shape[2:] == x.shape[2:]:
            return y
        if self._is_multiclass:
            y = F.interpolate(y.float(), size=x.shape[2:], mode="nearest").long()
            return y.clamp(min=0, max=self._num_seg_classes - 1)
        return F.interpolate(y.float(), size=x.shape[2:], mode="nearest")

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
                labels=labels,
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
                labels=labels,
                hist_module=self.hist_module,
                gen_version=gen_version,
            )

        if gen_version != "v1":
            guidance_map = apply_gaussian_blur_3d(guidance_map)

        return guidance_map, target_hist

    def _maybe_apply_generator(self, x: torch.Tensor, prob: float, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, bool]:
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
                dark_threshold=float(self.cfg.model.segmenter.dark_threshold),
                labels=labels
            )

        if self._segmenter_uses_anisotropic_degradation():
            self._ensure_segmenter_anisotropic_degradation()
            if self._anisotropic_degradation is not None:
                synthesized = self._anisotropic_degradation(synthesized)

        if bool(self.cfg.model.segmenter.channels_last_3d):
            synthesized = synthesized.contiguous(memory_format=torch.channels_last_3d)
        return synthesized.clone().float(), True

    def configure_optimizers(self) -> dict[str, Any]:
        total_params = sum(p.numel() for p in self.segmenter.parameters())
        trainable_params = sum(p.numel() for p in self.segmenter.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(
            "Segmenter parameter summary: "
            f"total={total_params}, trainable={trainable_params}, frozen={frozen_params}"
        )

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

        # Use the pre-augmentation scan for the "raw" image log column when
        # available (stashed by on_after_batch_transfer for GPU-aug versions).
        raw_image_for_log = batch.get("raw_image", None)

        x = self._to_plain_tensor(batch["image"])
        y = self._prepare_segmentation_target(batch["label"])
        if bool(self.cfg.model.segmenter.channels_last_3d) and not self._is_multiclass:
            y = y.to(memory_format=torch.channels_last_3d)

        unet_input, used_generator = self._maybe_apply_generator(
            x,
            prob=float(self.cfg.model.segmenter.aug_prob_train),
            labels=y
        )
        y = self._align_target_to_input(y, unet_input)

        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        train_logits_for_logging: torch.Tensor | None = None
        if self._segmenter_uses_consistency_regularization() and used_generator:
            x_combined = torch.cat([x, unet_input], dim=0)
            logits_combined = self.segmenter(x_combined)
            logits_raw, logits_synth = torch.chunk(logits_combined, 2, dim=0)

            supervised_loss = self.loss_fn(logits_synth, y)

            if logits_synth.shape[1] == 1:
                eps = torch.finfo(logits_synth.dtype).eps
                raw_prob = torch.sigmoid(logits_raw.detach()).clamp(min=eps, max=1.0 - eps)
                synth_prob = torch.sigmoid(logits_synth).clamp(min=eps, max=1.0 - eps)

                # Memory-lean Bernoulli KL: KL(p_raw || p_synth) without channel concatenation.
                consistency_map = raw_prob * (torch.log(raw_prob) - torch.log(synth_prob))
                consistency_map = consistency_map + (1.0 - raw_prob) * (
                    torch.log(1.0 - raw_prob) - torch.log(1.0 - synth_prob)
                )
                consistency_loss = consistency_map.mean()
            else:
                raw_prob = torch.softmax(logits_raw.detach(), dim=1)
                synth_log_prob = torch.log_softmax(logits_synth, dim=1)
                consistency_loss = F.kl_div(synth_log_prob, raw_prob, reduction="batchmean")

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

                if self._is_multiclass:
                    train_pred = torch.argmax(train_logits, dim=1, keepdim=True).float()
                    scale = float(max(1, self._num_seg_classes - 1))
                    train_pred = train_pred / scale
                    y_for_logging = y.float() / scale
                else:
                    train_pred = (torch.sigmoid(train_logits) > 0.5).float()
                    y_for_logging = y

            log_raw = (
                self._to_plain_tensor(raw_image_for_log)
                if raw_image_for_log is not None
                else x
            )
            self._log_segmenter_train_images(
                raw_input=log_raw,
                train_input=unet_input,
                y=y_for_logging,
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
        y = self._prepare_segmentation_target(batch["label"])
        if bool(self.cfg.model.segmenter.channels_last_3d) and not self._is_multiclass:
            y = y.to(memory_format=torch.channels_last_3d)

        val_input, _ = self._maybe_apply_generator(
            x,
            prob=float(self.cfg.model.segmenter.aug_prob_val),
            labels=y
        )
        y = self._align_target_to_input(y, val_input)

        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        # Keep validation eager to avoid retaining compiled graph outputs for metric state.
        logits = self.segmenter(val_input)
        val_loss = self.loss_fn(logits, y)
        if self._is_multiclass:
            val_pred_idx = torch.argmax(logits, dim=1, keepdim=True).long()
            val_pred = F.one_hot(
                val_pred_idx[:, 0],
                num_classes=self._num_seg_classes,
            ).permute(0, 4, 1, 2, 3).float()
            y_onehot = F.one_hot(
                y[:, 0].long(),
                num_classes=self._num_seg_classes,
            ).permute(0, 4, 1, 2, 3).float()
            self.dice_metric(y_pred=val_pred, y=y_onehot)

            scale = float(max(1, self._num_seg_classes - 1))
            val_pred_for_logging = val_pred_idx.float() / scale
            y_for_logging = y.float() / scale
        else:
            val_pred_for_logging = (torch.sigmoid(logits) > 0.5).float()
            y_for_logging = y
            self.dice_metric(y_pred=val_pred_for_logging, y=y)

        if (
            self.trainer.is_global_zero
            and self.logger is not None
            and isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run)
            and batch_idx == 0
            and int(self.cfg.training.segmenter.val_image_log_every) > 0
            and ((int(self.current_epoch) + 1) % int(self.cfg.training.segmenter.val_image_log_every) == 0)
        ):
            self._log_segmenter_val_images(
                val_input=val_input,
                y=y_for_logging,
                val_pred=val_pred_for_logging,
                batch_idx=batch_idx,
            )

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
        aggregated = self.dice_metric.aggregate()
        if self._is_multiclass:
            per_class = aggregated
            if per_class.ndim > 1:
                per_class = per_class.mean(dim=0)

            class_names = list(getattr(self.cfg.data, "class_names", []))
            if not class_names:
                if str(getattr(self.cfg.data, "name", "")) == "brats":
                    class_names = ["ncr", "ed", "et"]
                else:
                    class_names = [f"class_{i + 1}" for i in range(int(per_class.numel()))]
            for idx, value in enumerate(per_class.tolist()):
                class_name = class_names[idx] if idx < len(class_names) else f"class_{idx + 1}"
                self.log(f"val/dice_{class_name}", float(value), on_epoch=True, prog_bar=False, sync_dist=True)

            mean_val_dice = float(per_class.mean().item())
        else:
            mean_val_dice = float(aggregated.item())

        self.log("val/dice", mean_val_dice, on_epoch=True, prog_bar=True, sync_dist=True)
        self.dice_metric.reset()
