from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sample_uniform(
    shape: tuple[int, ...],
    low: float,
    high: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if high <= low:
        return torch.full(shape, low, device=device, dtype=dtype)
    return torch.empty(shape, device=device, dtype=dtype).uniform_(low, high)


def _prob_mask(batch_size: int, p: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return (torch.rand((batch_size, 1, 1, 1, 1), device=device, dtype=dtype) < p)


def _gaussian_kernel_1d(sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigma_clamped = max(float(sigma), 1e-3)
    kernel_size = int(max(3, round(sigma_clamped * 6.0)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) * 0.5
    g = torch.exp(-(coords * coords) / (2.0 * sigma_clamped * sigma_clamped))
    g = g / g.sum().clamp_min(torch.finfo(dtype).eps)
    return g.contiguous()


def _separable_gaussian_blur_3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    b, c, _, _, _ = x.shape
    del b
    g1d = _gaussian_kernel_1d(sigma, device=x.device, dtype=x.dtype)
    k = int(g1d.shape[0])
    pad = k // 2

    kd = g1d.view(1, 1, k, 1, 1).expand(c, 1, k, 1, 1).contiguous()
    kh = g1d.view(1, 1, 1, k, 1).expand(c, 1, 1, k, 1).contiguous()
    kw = g1d.view(1, 1, 1, 1, k).expand(c, 1, 1, 1, k).contiguous()

    out = F.conv3d(x, kd, padding=(pad, 0, 0), groups=c)
    out = F.conv3d(out, kh, padding=(0, pad, 0), groups=c)
    out = F.conv3d(out, kw, padding=(0, 0, pad), groups=c)
    return out


class BigAugmentation3D(nn.Module):
    """Zhang et al. style stacked augmentation pipeline for 3D MRI.

    Quality transforms (1-6) apply to image only.
    Spatial transforms (7-9) apply to image and label with one fused resampling.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = float(p)

    def _apply_intensity_stack(self, x: torch.Tensor) -> torch.Tensor:
        b, _, d, h, w = x.shape
        device = x.device
        dtype = x.dtype

        # Process appearance transforms at half resolution, then upsample.
        # This keeps all nine transforms active while reducing blur/noise hotspot cost.
        x_lr = F.interpolate(
            x,
            size=(max(1, d // 2), max(1, h // 2), max(1, w // 2)),
            mode="trilinear",
            align_corners=False,
        )

        # 1) Sharpness (unsharp masking variant from requested formula)
        sharp_mask = _prob_mask(b, self.p, device=device, dtype=dtype)
        sigma_sharp = float(_sample_uniform((1,), 0.25, 1.5, device=device, dtype=dtype).detach().cpu())
        alpha = _sample_uniform((b, 1, 1, 1, 1), 10.0, 30.0, device=device, dtype=dtype)
        blurred = _separable_gaussian_blur_3d(x_lr, sigma=sigma_sharp)
        filtered_blurred = _separable_gaussian_blur_3d(blurred, sigma=sigma_sharp)
        sharpened = blurred + (blurred - filtered_blurred) * alpha
        x_lr = torch.where(sharp_mask, sharpened, x_lr)

        # 2) Gaussian blur
        blur_mask = _prob_mask(b, self.p, device=device, dtype=dtype)
        sigma_blur = float(_sample_uniform((1,), 0.25, 1.5, device=device, dtype=dtype).detach().cpu())
        blurred2 = _separable_gaussian_blur_3d(x_lr, sigma=sigma_blur)
        x_lr = torch.where(blur_mask, blurred2, x_lr)

        # 3) Gaussian noise
        noise_mask = _prob_mask(b, self.p, device=device, dtype=dtype)
        noise_sigma = _sample_uniform((b, 1, 1, 1, 1), 0.1, 1.0, device=device, dtype=dtype)
        x_noise = x_lr + torch.randn_like(x_lr) * noise_sigma
        x_lr = torch.where(noise_mask, x_noise, x_lr)

        # 4) Brightness shift
        bright_mask = _prob_mask(b, self.p, device=device, dtype=dtype)
        bright_shift = _sample_uniform((b, 1, 1, 1, 1), -0.1, 0.1, device=device, dtype=dtype)
        x_lr = torch.where(bright_mask, x_lr + bright_shift, x_lr)

        # 5) Contrast via gamma correction
        gamma_mask = _prob_mask(b, self.p, device=device, dtype=dtype)
        gamma_branch = torch.rand((b, 1, 1, 1, 1), device=device, dtype=dtype) < 0.5
        gamma_low = _sample_uniform((b, 1, 1, 1, 1), 0.5, 1.0, device=device, dtype=dtype)
        gamma_high = _sample_uniform((b, 1, 1, 1, 1), 1.0, 4.5, device=device, dtype=dtype)
        gamma = torch.where(gamma_branch, gamma_low, gamma_high)
        x_gamma = torch.pow(x_lr.clamp(0.0, 1.0), gamma)
        x_lr = torch.where(gamma_mask, x_gamma, x_lr)

        # 6) Perturb (scale and shift)
        perturb_mask = _prob_mask(b, self.p, device=device, dtype=dtype)
        scale = 1.0 + _sample_uniform((b, 1, 1, 1, 1), -0.1, 0.1, device=device, dtype=dtype)
        shift = _sample_uniform((b, 1, 1, 1, 1), -0.1, 0.1, device=device, dtype=dtype)
        x_perturb = x_lr * scale + shift
        x_lr = torch.where(perturb_mask, x_perturb, x_lr)

        x_out = F.interpolate(x_lr, size=(d, h, w), mode="trilinear", align_corners=False)
        return x_out.clamp(0.0, 1.0)

    def _fused_spatial_transform(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, d, h, w = x.shape
        device = x.device
        dtype = x.dtype

        rot_on = (torch.rand((b,), device=device) < self.p)
        scale_on = (torch.rand((b,), device=device) < self.p)
        deform_on = (torch.rand((b,), device=device) < self.p)
        any_spatial = rot_on | scale_on | deform_on
        if not bool(any_spatial.any()):
            return x, y

        active_idx = torch.nonzero(any_spatial, as_tuple=False).squeeze(1)
        x_active = x[active_idx]
        y_active = y[active_idx]
        n = int(active_idx.numel())
        rot_on_a = rot_on[active_idx]
        scale_on_a = scale_on[active_idx]
        deform_on_a = deform_on[active_idx]

        # Rotation (degrees -> radians), independent gate.
        ang_deg = _sample_uniform((n, 3), -20.0, 20.0, device=device, dtype=dtype)
        ang = ang_deg * (math.pi / 180.0)
        ang = torch.where(rot_on_a.view(n, 1), ang, torch.zeros_like(ang))

        cx = torch.cos(ang[:, 0])
        sx = torch.sin(ang[:, 0])
        cy = torch.cos(ang[:, 1])
        sy = torch.sin(ang[:, 1])
        cz = torch.cos(ang[:, 2])
        sz = torch.sin(ang[:, 2])

        rx = torch.zeros((n, 3, 3), device=device, dtype=dtype)
        ry = torch.zeros((n, 3, 3), device=device, dtype=dtype)
        rz = torch.zeros((n, 3, 3), device=device, dtype=dtype)

        rx[:, 0, 0] = 1.0
        rx[:, 1, 1] = cx
        rx[:, 1, 2] = -sx
        rx[:, 2, 1] = sx
        rx[:, 2, 2] = cx

        ry[:, 0, 0] = cy
        ry[:, 0, 2] = sy
        ry[:, 1, 1] = 1.0
        ry[:, 2, 0] = -sy
        ry[:, 2, 2] = cy

        rz[:, 0, 0] = cz
        rz[:, 0, 1] = -sz
        rz[:, 1, 0] = sz
        rz[:, 1, 1] = cz
        rz[:, 2, 2] = 1.0

        rot = torch.bmm(rz, torch.bmm(ry, rx))

        # Scaling with independent gate.
        scales = _sample_uniform((n, 3), 0.4, 1.6, device=device, dtype=dtype)
        scales = torch.where(scale_on_a.view(n, 1), scales, torch.ones_like(scales))
        scale_mat = torch.diag_embed(scales)

        # Compose affine and convert to theta for affine_grid.
        affine = torch.bmm(rot, scale_mat)
        theta = torch.zeros((n, 3, 4), device=device, dtype=dtype)
        theta[:, :, :3] = affine
        grid = F.affine_grid(theta, size=x_active.shape, align_corners=False)

        # Elastic deformation with independent gate, fused into same grid.
        lr_factor = 8
        ld = max(1, d // lr_factor)
        lh = max(1, h // lr_factor)
        lw = max(1, w // lr_factor)

        final_grid = grid
        if bool(deform_on_a.any()):
            sigma_lr = float(
                _sample_uniform((1,), 10.0 / lr_factor, 13.0 / lr_factor, device=device, dtype=dtype)
                .detach()
                .cpu()
            )
            noise = (torch.rand((n, 3, ld, lh, lw), device=device, dtype=dtype) * 2.0) - 1.0
            smooth_noise = _separable_gaussian_blur_3d(noise, sigma=sigma_lr)
            smooth_noise = F.interpolate(smooth_noise, size=(d, h, w), mode="trilinear", align_corners=False)

            mag = _sample_uniform((n, 1, 1, 1, 1), 0.0, 1000.0, device=device, dtype=dtype)
            mag = torch.where(deform_on_a.view(n, 1, 1, 1, 1), mag, torch.zeros_like(mag))

            norm = torch.tensor(
                [
                    2.0 / max(w - 1, 1),
                    2.0 / max(h - 1, 1),
                    2.0 / max(d - 1, 1),
                ],
                device=device,
                dtype=dtype,
            ).view(1, 3, 1, 1, 1)
            disp = (smooth_noise * mag * norm).permute(0, 2, 3, 4, 1)
            final_grid = final_grid + disp

        final_grid = final_grid.clamp(-1.0, 1.0)

        x_warp = F.grid_sample(
            x_active,
            final_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        y_warp = F.grid_sample(
            y_active,
            final_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )

        x_out = x.clone()
        y_out = y.clone()
        x_out[active_idx] = x_warp
        y_out[active_idx] = y_warp
        return x_out, y_out

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x_aug = self._apply_intensity_stack(x)
            x_aug, y_aug = self._fused_spatial_transform(x_aug, y)
            return x_aug.clamp(0.0, 1.0), y_aug


def build_bigaug_augmentation() -> nn.Module:
    return BigAugmentation3D(p=0.5)
