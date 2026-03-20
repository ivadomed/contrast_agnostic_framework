from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import random
from typing import Any

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
    create_range_translation_guidance_map,
    generate_unified_targets,
)
from src.losses import (
    DiceEdgeLoss3D,
    DifferentiableWassersteinLoss,
    GuidanceLoss3D,
    RangeLoss,
    TotalVariationLoss3D,
)
from src.kornia_augmentations import build_kornia_augmentation


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


class CompiledLossWrapper(nn.Module):
    def __init__(self, model, histogram_module, wasserstein_loss_fn, edge_loss_fn, tv_loss_fn, range_loss_fn, guidance_loss_fn):
        super().__init__()
        self.model = model
        self.histogram_module = histogram_module
        self.wasserstein_loss_fn = wasserstein_loss_fn
        self.edge_loss_fn = edge_loss_fn
        self.tv_loss_fn = tv_loss_fn
        self.range_loss_fn = range_loss_fn
        self.guidance_loss_fn = guidance_loss_fn

    def forward(self, x: torch.Tensor, 
                num_bins: int, num_chunks: int, dark_threshold: float,
                guidance_blur_k: int, guidance_blur_s: float,
                w_edge: float, w_tv: float, w_range: float, w_wass: float,
                w_guide_blur: float, w_guide_sharp: float) -> dict[str, torch.Tensor]:
        
        target_hist, perms = generate_unified_targets(
            input_images=x,
            num_bins=num_bins,
            num_chunks=num_chunks,
            dark_threshold=dark_threshold,
            hist_module=self.histogram_module,
        )
        guidance_map = create_range_translation_guidance_map(
            input_image=x,
            perms=perms,
            num_chunks=num_chunks,
            dark_threshold=dark_threshold,
        )
        
        # Stop propagating gradient through guidance map blur
        bg_guidance = guidance_map.detach()
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
        guidance_loss_blurred = self.guidance_loss_fn(synthesized_01, guidance_map)
        guidance_loss_sharp = F.l1_loss(synthesized_01, guidance_map)

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
            self.edge_loss_fn, self.tv_loss_fn, self.range_loss_fn, self.guidance_loss_fn
        )
        if bool(self.cfg.training.generator.compile_model) and hasattr(torch, "compile"):
            self.compiled_wrapper = torch.compile(self.compiled_wrapper, mode="reduce-overhead")

        self._gpu_aug: nn.Module | None = None

    def _ensure_gpu_aug(self) -> None:
        if self._gpu_aug is not None:
            return
        self._gpu_aug = build_kornia_augmentation(self.cfg).to(self.device)

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
                    image = self._gpu_aug(image.contiguous())

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
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")

        self.generator: MRI_Synthesis_Net | None = None
        self.hist_module: DifferentiableHistogram3D | None = None
        self._generator_is_ready = False

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit"):
            return

        if bool(self.cfg.model.segmenter.compile_model) and hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                self.segmenter = torch.compile(self.segmenter)
            except Exception:
                pass

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
            gen_version = self.cfg.model.segmenter.gen_version
            generator_weights = str(project_root / "checkpoints" / gen_version / f"mri_generator_{contrast}_epoch_30.pth")

        self.generator = MRI_Synthesis_Net(in_channels=2, out_channels=1)
        if bool(self.cfg.model.segmenter.channels_last_3d):
            self.generator = self.generator.to(memory_format=torch.channels_last_3d)

        checkpoint = torch.load(generator_weights, map_location="cpu")
        state_dict = _extract_normalized_state_dict(checkpoint)
        self.generator.load_state_dict(state_dict, strict=True)
        self.generator.eval()
        for p in self.generator.parameters():
            p.requires_grad = False

        self.hist_module = DifferentiableHistogram3D(
            num_bins=int(self.cfg.model.segmenter.num_bins),
            value_range=(0.0, 1.0),
        )
        self._generator_is_ready = True

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
        gen_version = str(self.cfg.model.segmenter.gen_version)

        if gen_version in ("v3", "v4"):
            target_hist, _, guidance_map = generate_unified_targets(
                input_images=x,
                num_bins=int(self.cfg.model.segmenter.num_bins),
                num_chunks=int(self.cfg.model.segmenter.num_chunks),
                dark_threshold=float(self.cfg.model.segmenter.dark_threshold),
                hist_module=self.hist_module,
                return_guidance_map=True,
            )
        else:
            target_hist, perms = generate_unified_targets(
                input_images=x,
                num_bins=int(self.cfg.model.segmenter.num_bins),
                num_chunks=int(self.cfg.model.segmenter.num_chunks),
                dark_threshold=float(self.cfg.model.segmenter.dark_threshold),
                hist_module=self.hist_module,
            )
            guidance_map = create_range_translation_guidance_map(
                input_image=x,
                perms=perms,
                num_chunks=int(self.cfg.model.segmenter.num_chunks),
                dark_threshold=float(self.cfg.model.segmenter.dark_threshold),
            )

        if gen_version != "v1":
            guidance_map = apply_gaussian_blur_3d(guidance_map)

        return guidance_map, target_hist

    def _maybe_apply_generator(self, x: torch.Tensor, prob: float) -> tuple[torch.Tensor, bool]:
        use_generator = bool(self.cfg.model.segmenter.use_generator) and self.generator is not None
        if not use_generator or random.random() >= float(prob):
            return x, False

        with torch.no_grad():
            guidance_map, _ = self._build_generator_guidance(x=x)
            generator_input = torch.cat([x, guidance_map], dim=1)
            generator_output = self.generator(generator_input)
            synthesized = ((generator_output + 1.0) * 0.5).clamp(0.0, 1.0)
            if bool(self.cfg.model.segmenter.channels_last_3d):
                synthesized = synthesized.to(memory_format=torch.channels_last_3d)
            return synthesized.float(), True

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
            T_max=int(self.cfg.training.max_epochs.segmenter),
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
        x = self._to_plain_tensor(batch["image"])
        y = (self._to_plain_tensor(batch["label"]) > 0).float()

        unet_input, used_generator = self._maybe_apply_generator(
            x,
            prob=float(self.cfg.model.segmenter.aug_prob_train),
        )

        logits = self.segmenter(unet_input)
        loss = self.loss_fn(logits, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/used_generator", float(used_generator), on_step=True, on_epoch=False)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.dice_metric.reset()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = self._to_plain_tensor(batch["image"])
        y = (self._to_plain_tensor(batch["label"]) > 0).float()

        val_input, _ = self._maybe_apply_generator(
            x,
            prob=float(self.cfg.model.segmenter.aug_prob_val),
        )

        logits = self.segmenter(val_input)
        val_loss = self.loss_fn(logits, y)
        val_pred = (torch.sigmoid(logits) > 0.5).float()
        self.dice_metric(y_pred=val_pred, y=y)

        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return val_loss

    def on_validation_epoch_end(self) -> None:
        mean_val_dice = float(self.dice_metric.aggregate().item())
        self.log("val/dice", mean_val_dice, on_epoch=True, prog_bar=True, sync_dist=True)
        self.dice_metric.reset()
