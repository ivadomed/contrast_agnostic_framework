from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import RandomAffine3D
from kornia.filters import get_gaussian_kernel3d
from kornia.utils import create_meshgrid3d
from omegaconf import DictConfig


class RandomElasticTransform3D(nn.Module):
    """Batched 3D elastic deformation implemented with Kornia/Torch ops.

    The transform expects a 5D tensor (B, C, D, H, W) and applies a smooth
    random displacement field per sample.
    """

    def __init__(
        self,
        *,
        p: float,
        sigma_range: tuple[float, float],
        magnitude_range: tuple[float, float],
        mode: str = "bilinear",
        padding_mode: str = "border",
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.p = float(p)
        self.sigma_range = tuple(float(v) for v in sigma_range)
        self.magnitude_range = tuple(float(v) for v in magnitude_range)
        self.mode = mode
        self.padding_mode = padding_mode
        
        
        self.align_corners = align_corners
        
        # Cache for meshgrid and kernel to avoid repeated allocations
        self._cached_grid = None
        self._cached_kernel = None

        
        # Cache for meshgrid and kernel to avoid repeated allocations
        self._cached_grid = None
        self._cached_kernel = None


    def _sample_uniform(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        low, high = self.magnitude_range
        if high <= low:
            return torch.full((batch_size,), low, device=device, dtype=dtype)
        return torch.empty((batch_size,), device=device, dtype=dtype).uniform_(low, high)

    def _gaussian_kernel_size(self, sigma: float) -> int:
        # Roughly cover +/- 3 sigma and keep kernel odd.
        kernel = int(max(3, round(sigma * 6)))
        return kernel + 1 if kernel % 2 == 0 else kernel

    def _smooth_noise(
        self,
        noise: torch.Tensor,
        *,
        kernel_size: int,
        sigma: float,
    ) -> torch.Tensor:
        channels = noise.shape[1]
        
        if self._cached_kernel is None or self._cached_kernel.shape[-1] != kernel_size or self._cached_kernel.device != noise.device:
            coords = torch.arange(kernel_size, dtype=noise.dtype, device=noise.device) - (kernel_size - 1) / 2.0
            g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g1d = g1d / g1d.sum().clamp_min(torch.finfo(g1d.dtype).eps)
            self._cached_kernel = g1d.contiguous()

        g1d = self._cached_kernel
        padding = kernel_size // 2

        # 3D Gaussian Blur is separable. We apply 3 sequential 1D convolutions 
        # instead of 1 massive 3D convolution to reduce MACs by a factor of ~2000x!
        k_d = g1d.view(1, 1, kernel_size, 1, 1).expand(channels, 1, kernel_size, 1, 1).contiguous()
        k_h = g1d.view(1, 1, 1, kernel_size, 1).expand(channels, 1, 1, kernel_size, 1).contiguous()
        k_w = g1d.view(1, 1, 1, 1, kernel_size).expand(channels, 1, 1, 1, kernel_size).contiguous()

        smoothed = F.conv3d(noise, k_d, padding=(padding, 0, 0), groups=channels)
        smoothed = F.conv3d(smoothed, k_h, padding=(0, padding, 0), groups=channels)
        smoothed = F.conv3d(smoothed, k_w, padding=(0, 0, padding), groups=channels)
        
        return smoothed

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.p <= 0.0 or x.ndim != 5:
            return x if mask is None else (x, mask)

        batch_size, _, depth, height, width = x.shape
        device = x.device
        dtype = x.dtype

        import random
        # No host-device sync: use python random for batch-level probability
        masks = [random.random() < self.p for _ in range(batch_size)]
        if not any(masks):
            return x if mask is None else (x, mask)
            
        apply_mask = torch.tensor(masks, device=device, dtype=torch.bool)

        sigma_low, sigma_high = self.sigma_range
        sigma_val = sigma_low if sigma_high <= sigma_low else random.uniform(sigma_low, sigma_high)
        
        # 2. High-Frequency Waste in Low-Frequency Fields: Scale down for noise generation
        scale_factor = 4
        lr_depth = max(1, depth // scale_factor)
        lr_height = max(1, height // scale_factor)
        lr_width = max(1, width // scale_factor)
        lr_sigma = sigma_val / scale_factor
        
        kernel_size = self._gaussian_kernel_size(lr_sigma)

        # Generate noise at lower resolution
        noise = (torch.rand((batch_size, 3, lr_depth, lr_height, lr_width), device=device, dtype=dtype) * 2.0) - 1.0
        smoothed_noise_lr = self._smooth_noise(noise, kernel_size=kernel_size, sigma=lr_sigma)

        # Upsample the smoothed noise back to full resolution
        smoothed_noise = F.interpolate(smoothed_noise_lr, size=(depth, height, width), mode='trilinear', align_corners=False)

        import random
        v_low, v_high = self.magnitude_range
        v_mags = [v_low if v_high <= v_low else random.uniform(v_low, v_high) for _ in range(batch_size)]
        voxel_magnitude = torch.tensor(v_mags, device=device, dtype=dtype).view(batch_size, 1, 1, 1, 1)

        norm_scale = torch.tensor(
            [
                2.0 / max(width - 1, 1),
                2.0 / max(height - 1, 1),
                2.0 / max(depth - 1, 1),
            ],
            device=device,
            dtype=dtype,
        ).view(1, 3, 1, 1, 1)
        displacement = smoothed_noise * voxel_magnitude * norm_scale

        if self._cached_grid is None or self._cached_grid.shape[1:4] != (depth, height, width) or self._cached_grid.device != device:
            self._cached_grid = create_meshgrid3d(depth, height, width, normalized_coordinates=True, device=device, dtype=dtype)
        
        base_grid = self._cached_grid.expand(batch_size, -1, -1, -1, -1)
        warp_grid = (base_grid + displacement.permute(0, 2, 3, 4, 1)).clamp(-1.0, 1.0)
        
        # Fix memory_format thrashing by keeping grid_sample in contiguous format or channels_last
        warp_grid = warp_grid.contiguous()
        x_contig = x.contiguous()


        warped = F.grid_sample(
            x_contig,
            warp_grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

        keep_mask = (~apply_mask).view(batch_size, 1, 1, 1, 1)
        res_x = torch.where(keep_mask, x, warped)
        
        if mask is not None:
            mask_contig = mask.contiguous()
            warped_mask = F.grid_sample(
                mask_contig,
                warp_grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=self.align_corners,
            )
            res_mask = torch.where(keep_mask, mask, warped_mask)
            return res_x, res_mask
            
        return res_x



class RandomLowResolution3D(nn.Module):
    def __init__(self, p: float = 0.3, zoom_range: tuple[float, float] = (0.5, 1.0)):
        super().__init__()
        self.p = p
        self.zoom_range = zoom_range
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.p <= 0.0 or x.ndim != 5:
            return x if mask is None else (x, mask)
            
        b, c, d, h, w = x.shape
        device = x.device
        dtype = x.dtype
        
        import random
        # Optimize host-device sync
        masks = [random.random() < self.p for _ in range(b)]
        if not any(masks):
            return x if mask is None else (x, mask)
            
        output = x.clone()
        for i in range(b):
            if masks[i]:
                zoom = random.uniform(self.zoom_range[0], self.zoom_range[1])
                down_size = (max(1, int(d * zoom)), max(1, int(h * zoom)), max(1, int(w * zoom)))
                down = F.interpolate(x[i:i+1], size=down_size, mode='nearest')
                output[i:i+1] = F.interpolate(down, size=(d, h, w), mode='trilinear', align_corners=False)

        return output if mask is None else (output, mask)

class RandomGaussianNoise3D(nn.Module):
    def __init__(self, p: float = 0.2, mean: float = 0.0, std: float = 0.02):
        super().__init__()
        self.p = p
        self.mean = mean
        self.std = std
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.p <= 0.0 or x.ndim != 5:
            return x if mask is None else (x, mask)
            
        b = x.shape[0]
        import random
        masks = [random.random() < self.p for _ in range(b)]
        if not any(masks):
            return x if mask is None else (x, mask)
            
        apply_mask = torch.tensor(masks, device=x.device, dtype=torch.bool).view(b, 1, 1, 1, 1)
        noise = torch.randn_like(x) * self.std + self.mean
        output = torch.where(apply_mask, x + noise, x)
        return output if mask is None else (output, mask)

class RandomGaussianSmooth3D(nn.Module):
    def __init__(self, p: float = 0.2, sigma_range: tuple[float, float] = (0.5, 1.0)):
        super().__init__()
        self.p = p
        self.sigma_range = sigma_range
        
    def _get_1d_kernel(self, sigma: float, device: torch.device, dtype: torch.dtype):
        kernel_size = int(max(3, round(sigma * 6)))
        if kernel_size % 2 == 0: kernel_size += 1
        
        coords = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2.0
        g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g1d = g1d / g1d.sum().clamp_min(torch.finfo(dtype).eps)
        return g1d.contiguous(), kernel_size

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.p <= 0.0 or x.ndim != 5:
            return x if mask is None else (x, mask)
        b, c, d, h, w = x.shape
        device = x.device
        dtype = x.dtype
        
        apply_mask = torch.rand((b,), device=device) < self.p
        if not bool(apply_mask.any()):
            return x if mask is None else (x, mask)
            
        output = x.clone()
        for i in range(b):
            if apply_mask[i]:
                sigma = float(torch.empty(1, device=device).uniform_(self.sigma_range[0], self.sigma_range[1]).item())
                g1d, k = self._get_1d_kernel(sigma, device, dtype)

                k_d = g1d.view(1, 1, k, 1, 1).expand(c, 1, k, 1, 1).contiguous()
                k_h = g1d.view(1, 1, 1, k, 1).expand(c, 1, 1, k, 1).contiguous()
                k_w = g1d.view(1, 1, 1, 1, k).expand(c, 1, 1, 1, k).contiguous()

                padding = k // 2
                v = x[i:i+1]
                v = F.conv3d(v, k_d, padding=(padding, 0, 0), groups=c)
                v = F.conv3d(v, k_h, padding=(0, padding, 0), groups=c)
                v = F.conv3d(v, k_w, padding=(0, 0, padding), groups=c)
                output[i:i+1] = v

        return output if mask is None else (output, mask)

class KorniaMRIAugmentation3D(nn.Module):
    """Kornia-only 3D augmentation pipeline for MRI volumes."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        aug_cfg = cfg.training.generator.gpu_aug
        scale_delta = tuple(float(v) for v in aug_cfg.affine_scale_range)
        scale = tuple((1.0 - delta, 1.0 + delta) for delta in scale_delta)
        self.affine = RandomAffine3D(
            p=float(aug_cfg.affine_prob),
            degrees=tuple(float(v) for v in aug_cfg.affine_rotate_range),
            scale=scale,
            resample="BILINEAR",
            same_on_batch=False,
        )
        self.elastic = RandomElasticTransform3D(
            p=float(aug_cfg.elastic_prob),
            sigma_range=tuple(float(v) for v in aug_cfg.elastic_sigma_range),
            magnitude_range=tuple(float(v) for v in aug_cfg.elastic_magnitude_range),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        self.low_res = RandomLowResolution3D(p=0.3, zoom_range=(0.5, 1.0))
        self.noise = RandomGaussianNoise3D(p=0.2, mean=0.0, std=0.02)
        self.smooth = RandomGaussianSmooth3D(p=0.2, sigma_range=(0.5, 1.0))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Prevent gradients for data augmentation
        with torch.no_grad():
            if mask is not None:
                # apply affine on concatenated tensors, and then threshold the mask to emulate 'nearest'
                b, cx, d, h, w = x.shape
                cm = mask.shape[1]
                cat_xm = torch.cat([x, mask], dim=1)
                cat_xm = self.affine(cat_xm)
                x = cat_xm[:, :cx]
                mask = (cat_xm[:, cx:] > 0.5).float()

                x, mask = self.elastic(x, mask)
                x, mask = self.low_res(x, mask)
                x, mask = self.noise(x, mask)
                x, mask = self.smooth(x, mask)
                return x.clamp(0.0, 1.0), mask
            else:
                x = self.affine(x)
                x = self.elastic(x)
                x = self.low_res(x)
                x = self.noise(x)
                x = self.smooth(x)
                return x.clamp(0.0, 1.0)


def build_kornia_augmentation(cfg: DictConfig) -> nn.Module:
    return KorniaMRIAugmentation3D(cfg)
