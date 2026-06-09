from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.synthesis.histogram_ops import apply_gaussian_blur_3d

class GuidanceLoss3D(nn.Module):
    """
    Applies a 3D Gaussian blur to both the prediction and the target guidance map,
    then computes the L1 loss. This acts as a low-frequency spatial guide.
    """
    def __init__(self, kernel_size: int = 5, sigma: float = 2.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the 3D Gaussian blur."""
        return apply_gaussian_blur_3d(x, kernel_size=self.kernel_size, sigma=self.sigma)

    def forward(self, prediction: torch.Tensor, target_guidance: torch.Tensor) -> torch.Tensor:
        # Instead of blurring both and subtracting, we subtract then blur once
        # because convolution (blur) is a linear operation!
        diff = prediction - target_guidance
        blurred_diff = self._blur(diff)
        return torch.abs(blurred_diff).mean()
    
    
class DiceEdgeLoss3D(nn.Module):
    def __init__(self, sigmoid_scale: float = 12.0, edge_threshold: float = 0.05, dice_weight: float = 1.0, l1_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.sigmoid_scale = sigmoid_scale
        self.edge_threshold = edge_threshold
        self.dice_weight = dice_weight
        self.l1_weight = l1_weight
        self.eps = eps
        
        # Combine Sobel kernels into one kernel so we only do a single 3x3x3 conv
        sobel_x = self._build_kernel("x")
        sobel_y = self._build_kernel("y")
        sobel_z = self._build_kernel("z")
        self.register_buffer("sobel_kernel", torch.cat([sobel_x, sobel_y, sobel_z], dim=0), persistent=False)

    @staticmethod
    def _build_kernel(axis: str) -> torch.Tensor:
        smoothing = torch.tensor([1.0, 2.0, 1.0])
        derivative = torch.tensor([-1.0, 0.0, 1.0])

        if axis == "x":
            kernel = torch.einsum("i,j,k->ijk", smoothing, smoothing, derivative)
        elif axis == "y":
            kernel = torch.einsum("i,j,k->ijk", smoothing, derivative, smoothing)
        elif axis == "z":
            kernel = torch.einsum("i,j,k->ijk", derivative, smoothing, smoothing)
        else:
            raise ValueError(f"Unsupported axis: {axis}")

        return kernel.view(1, 1, 3, 3, 3)

    def _grad_mag(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1, keepdim=True)
        grad = F.conv3d(x, self.sobel_kernel, padding=1)
        return torch.sqrt(grad.pow(2).sum(dim=1, keepdim=True) + self.eps)

    def _edge_map(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.sigmoid_scale * (self._grad_mag(x) - self.edge_threshold))

    def _edge_map_with_threshold(self, x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """Per-sample threshold, shape (B, 1, 1, 1, 1), broadcast over spatial dims."""
        return torch.sigmoid(self.sigmoid_scale * (self._grad_mag(x) - threshold))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor,
                threshold: torch.Tensor | None = None) -> torch.Tensor:
        if threshold is not None:
            pred_edges = self._edge_map_with_threshold(prediction, threshold)
            target_edges = self._edge_map_with_threshold(target, threshold)
        else:
            pred_edges = self._edge_map(prediction)
            target_edges = self._edge_map(target)

        l1_loss = F.l1_loss(pred_edges, target_edges)
        intersection = (pred_edges * target_edges).sum(dim=(1, 2, 3, 4))
        denominator = pred_edges.sum(dim=(1, 2, 3, 4)) + target_edges.sum(dim=(1, 2, 3, 4))
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        dice_loss = 1.0 - dice.mean()

        return self.l1_weight * l1_loss + self.dice_weight * dice_loss


class TotalVariationLoss3D(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        depth_tv = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).mean()
        height_tv = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).mean()
        width_tv = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).mean()
        return depth_tv + height_tv + width_tv


class DifferentiableWassersteinLoss(nn.Module):
    def __init__(self, dark_threshold: float = 0.05, pdf_weight: float = 1.0, cdf_weight: float = 1.0, bright_weight: float = 1.0):
        super().__init__()
        self.dark_threshold = dark_threshold
        self.pdf_weight = pdf_weight
        self.cdf_weight = cdf_weight
        self.bright_weight = bright_weight

    def forward(self, generated_hist: torch.Tensor, target_hist: torch.Tensor) -> torch.Tensor:
        if generated_hist.shape != target_hist.shape:
            raise ValueError("Histogram shapes must match.")

        # Convert raw voxel counts to Probability Density Functions (PDFs)
        gen_pdf = generated_hist / generated_hist.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        tar_pdf = target_hist / target_hist.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        pdf_loss = F.l1_loss(gen_pdf, tar_pdf) * 200.0  # Scaled up slightly based on your 2D code

        # Calculate Cumulative Distribution Functions (CDFs)
        gen_cdf = torch.cumsum(gen_pdf, dim=-1)
        tar_cdf = torch.cumsum(tar_pdf, dim=-1)
        cdf_loss = F.l1_loss(gen_cdf, tar_cdf)

        # Calculate isolated Bright Loss
        num_bins = generated_hist.shape[-1]
        bright_start = min(int(self.dark_threshold * num_bins), num_bins - 1)
        
        bright_generated = generated_hist[..., bright_start:]
        bright_target = target_hist[..., bright_start:]
        
        # Normalize the bright parts separately
        bright_gen_pdf = bright_generated / bright_generated.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        bright_tar_pdf = bright_target / bright_target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        
        bright_gen_cdf = torch.cumsum(bright_gen_pdf, dim=-1)
        bright_tar_cdf = torch.cumsum(bright_tar_pdf, dim=-1)
        
        bright_loss = F.l1_loss(bright_gen_pdf, bright_tar_pdf) * 200.0 + F.l1_loss(bright_gen_cdf, bright_tar_cdf)

        return self.pdf_weight * pdf_loss + self.cdf_weight * cdf_loss + self.bright_weight * bright_loss


class RangeLoss(nn.Module):
    def __init__(self, min_value: float = -1.0, max_value: float = 1.0):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        below = F.relu(self.min_value - x)
        above = F.relu(x - self.max_value)
        return (below + above).mean()


class LocalVarianceLoss3D(nn.Module):
    """L1 loss between local standard deviation maps of prediction and target.

    Compares texture magnitude (local intensity variation) at each spatial location
    without constraining specific edge positions — directly targets GLCM homogeneity.
    """

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def local_std(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1, keepdim=True)
        k, p = self.kernel_size, self.padding
        mu = F.avg_pool3d(x, kernel_size=k, stride=1, padding=p)
        mu2 = F.avg_pool3d(x ** 2, kernel_size=k, stride=1, padding=p)
        # eps avoids sqrt(0) → infinite gradient in background/smooth regions
        return (mu2 - mu ** 2).clamp(min=0.0).add(1e-8).sqrt()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.local_std(prediction), self.local_std(target))


class MultiScaleGradientLoss3D(nn.Module):
    """Multi-scale gradient magnitude loss.

    Computes L1 loss between prediction and target gradient magnitudes at
    multiple Gaussian-smoothed scales. Designed as a continuous replacement
    for DiceEdgeLoss3D that captures texture at multiple frequencies.
    """

    def __init__(
        self,
        sigmas: tuple[float, ...] = (1.0, 2.0, 4.0),
        scale_weights: tuple[float, ...] = (1.0, 1.0, 0.5),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.sigmas = sigmas
        self.scale_weights = scale_weights
        self.eps = eps

        sobel_x = self._build_sobel("x")
        sobel_y = self._build_sobel("y")
        sobel_z = self._build_sobel("z")
        self.register_buffer(
            "sobel_kernel", torch.cat([sobel_x, sobel_y, sobel_z], dim=0), persistent=False
        )

    @staticmethod
    def _build_sobel(axis: str) -> torch.Tensor:
        smoothing = torch.tensor([1.0, 2.0, 1.0])
        derivative = torch.tensor([-1.0, 0.0, 1.0])
        if axis == "x":
            kernel = torch.einsum("i,j,k->ijk", smoothing, smoothing, derivative)
        elif axis == "y":
            kernel = torch.einsum("i,j,k->ijk", smoothing, derivative, smoothing)
        else:
            kernel = torch.einsum("i,j,k->ijk", derivative, smoothing, smoothing)
        return kernel.view(1, 1, 3, 3, 3)

    def _grad_mag(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        x = x.mean(dim=1, keepdim=True)
        if sigma > 0.5:
            k = max(3, int(sigma * 4) | 1)
            x = apply_gaussian_blur_3d(x, kernel_size=k, sigma=sigma)
        grad = F.conv3d(x, self.sobel_kernel, padding=1)
        return torch.sqrt(grad.pow(2).sum(dim=1, keepdim=True) + self.eps)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = torch.tensor(0.0, device=prediction.device)
        for sigma, w in zip(self.sigmas, self.scale_weights):
            total = total + w * F.l1_loss(
                self._grad_mag(prediction, sigma), self._grad_mag(target, sigma)
            )
        return total