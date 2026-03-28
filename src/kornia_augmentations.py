from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import RandomAffine3D
from kornia.utils import create_meshgrid3d
from omegaconf import DictConfig


def _resolve_aug_cfg(cfg: DictConfig, *, task: str) -> DictConfig:
    if task == "segmenter" and hasattr(cfg.training, "segmenter") and hasattr(cfg.training.segmenter, "gpu_aug"):
        return cfg.training.segmenter.gpu_aug
    return cfg.training.generator.gpu_aug


class RandomFourierAmplitude3D(nn.Module):
    """Randomly perturbs high-frequency FFT amplitudes while preserving phase."""

    def __init__(
        self,
        p: float = 0.5,
        low_freq_ratio: float = 0.15,
        scale_range: tuple[float, float] = (0.5, 1.5),
    ) -> None:
        super().__init__()
        self.p = float(p)
        self.low_freq_ratio = float(low_freq_ratio)
        self.scale_range = tuple(float(v) for v in scale_range)

    def _build_low_frequency_mask(self, x: torch.Tensor) -> torch.Tensor:
        _, _, depth, height, width = x.shape
        device = x.device

        dz = max(1, int(round(depth * self.low_freq_ratio)))
        dy = max(1, int(round(height * self.low_freq_ratio)))
        dx = max(1, int(round(width * self.low_freq_ratio)))

        cz = depth // 2
        cy = height // 2
        cx = width // 2

        z0 = max(0, cz - dz // 2)
        z1 = min(depth, z0 + dz)
        y0 = max(0, cy - dy // 2)
        y1 = min(height, y0 + dy)
        x0 = max(0, cx - dx // 2)
        x1 = min(width, x0 + dx)

        low_freq_mask = torch.zeros((1, 1, depth, height, width), device=device, dtype=torch.bool)
        low_freq_mask[:, :, z0:z1, y0:y1, x0:x1] = True
        return low_freq_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or x.ndim != 5:
            return x

        batch_size = x.shape[0]
        apply_mask = (torch.rand((batch_size,), device=x.device) < self.p).view(batch_size, 1, 1, 1, 1)
        if not bool(apply_mask.any()):
            return x

        fft = torch.fft.fftn(x, dim=(-3, -2, -1))
        amp = torch.abs(fft)
        phase = torch.angle(fft)

        shifted_amp = torch.fft.fftshift(amp, dim=(-3, -2, -1))
        low_freq_mask = self._build_low_frequency_mask(x)

        scale_min, scale_max = self.scale_range
        if scale_max <= scale_min:
            random_scales = torch.full_like(shifted_amp, scale_min)
        else:
            random_scales = torch.empty_like(shifted_amp).uniform_(scale_min, scale_max)

        high_freq_scale = torch.where(low_freq_mask, torch.ones_like(random_scales), random_scales)
        perturbed_shifted_amp = shifted_amp * high_freq_scale
        perturbed_amp = torch.fft.ifftshift(perturbed_shifted_amp, dim=(-3, -2, -1))

        complex_tensor = perturbed_amp * torch.exp(1j * phase)
        transformed = torch.fft.ifftn(complex_tensor, dim=(-3, -2, -1)).real

        output = transformed.clamp(0.0, 1.0)
        return torch.where(apply_mask, output, x)


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

        apply_mask = torch.rand((batch_size,), device=device) < self.p
        if not bool(apply_mask.any()):
            return x if mask is None else (x, mask)

        sigma_low, sigma_high = self.sigma_range
        if sigma_high <= sigma_low:
            sigma_val = sigma_low
        else:
            sigma_val = float(torch.empty(1, device=device).uniform_(sigma_low, sigma_high).item())
        
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

        v_low, v_high = self.magnitude_range
        if v_high <= v_low:
            voxel_magnitude = torch.full((batch_size, 1, 1, 1, 1), v_low, device=device, dtype=dtype)
        else:
            voxel_magnitude = torch.empty((batch_size, 1, 1, 1, 1), device=device, dtype=dtype).uniform_(v_low, v_high)

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
            
        b, _, d, h, w = x.shape
        device = x.device

        apply_mask = torch.rand((b,), device=device) < self.p
        if not bool(apply_mask.any()):
            return x if mask is None else (x, mask)

        zoom_low, zoom_high = self.zoom_range
        if zoom_high <= zoom_low:
            zooms = torch.full((b,), zoom_low, device=device)
        else:
            zooms = torch.empty((b,), device=device).uniform_(zoom_low, zoom_high)
            
        output = x.clone()
        for i in range(b):
            if bool(apply_mask[i]):
                zoom = float(zooms[i].item())
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
        apply_mask = (torch.rand((b,), device=x.device) < self.p).view(b, 1, 1, 1, 1)
        if not bool(apply_mask.any()):
            return x if mask is None else (x, mask)

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

    def __init__(self, cfg: DictConfig, *, task: str = "generator"):
        super().__init__()
        aug_cfg = _resolve_aug_cfg(cfg, task=task)
        scale_delta = tuple(float(v) for v in aug_cfg.affine_scale_range)
        scale = tuple((1.0 - delta, 1.0 + delta) for delta in scale_delta)
        affine_prob = float(aug_cfg.affine_prob)
        if task == "generator" and str(cfg.version) == "v8":
            # v8 robustness hotfix: avoid Kornia affine path that can trigger cuSOLVER failures
            # in warp_affine3d homography inversion on some driver/runtime combinations.
            affine_prob = 0.0

        self.affine_image = None
        self.affine_mask = None
        if affine_prob > 0.0:
            self.affine_image = RandomAffine3D(
                p=affine_prob,
                degrees=tuple(float(v) for v in aug_cfg.affine_rotate_range),
                scale=scale,
                resample="BILINEAR",
                same_on_batch=False,
            )
            self.affine_mask = RandomAffine3D(
                p=1.0,
                degrees=tuple(float(v) for v in aug_cfg.affine_rotate_range),
                scale=scale,
                resample="NEAREST",
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
        self.low_res = RandomLowResolution3D(
            p=float(aug_cfg.low_res_prob),
            zoom_range=tuple(float(v) for v in aug_cfg.low_res_zoom_range),
        )
        self.noise = RandomGaussianNoise3D(
            p=float(aug_cfg.noise_prob),
            mean=float(aug_cfg.noise_mean),
            std=float(aug_cfg.noise_std),
        )
        self.smooth = RandomGaussianSmooth3D(
            p=float(aug_cfg.smooth_prob),
            sigma_range=tuple(float(v) for v in aug_cfg.smooth_sigma_range),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Prevent gradients for data augmentation
        with torch.no_grad():
            if mask is not None:
                if self.affine_image is not None and self.affine_mask is not None:
                    x = self.affine_image(x)
                    affine_params = self.affine_image._params
                    mask = self.affine_mask(mask, params=affine_params)

                x, mask = self.elastic(x, mask)
                x, mask = self.low_res(x, mask)
                x, mask = self.noise(x, mask)
                x, mask = self.smooth(x, mask)
                return x.clamp(0.0, 1.0), mask
            else:
                if self.affine_image is not None:
                    x = self.affine_image(x)
                x = self.elastic(x)
                x = self.low_res(x)
                x = self.noise(x)
                x = self.smooth(x)
                return x.clamp(0.0, 1.0)


def build_kornia_augmentation(cfg: DictConfig, *, task: str = "generator") -> nn.Module:
    return KorniaMRIAugmentation3D(cfg, task=task)
