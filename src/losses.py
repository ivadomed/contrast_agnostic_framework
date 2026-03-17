from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.histogram_ops import apply_gaussian_blur_3d

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
        blurred_pred = self._blur(prediction)
        blurred_target = self._blur(target_guidance)
        return F.l1_loss(blurred_pred, blurred_target)
    
    
class DiceEdgeLoss3D(nn.Module):
    def __init__(self, sigmoid_scale: float = 12.0, edge_threshold: float = 0.05, dice_weight: float = 1.0, l1_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.sigmoid_scale = sigmoid_scale
        self.edge_threshold = edge_threshold
        self.dice_weight = dice_weight
        self.l1_weight = l1_weight
        self.eps = eps
        self.register_buffer("sobel_x", self._build_kernel("x"), persistent=False)
        self.register_buffer("sobel_y", self._build_kernel("y"), persistent=False)
        self.register_buffer("sobel_z", self._build_kernel("z"), persistent=False)

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

    def _edge_map(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1, keepdim=True)
        grad_x = F.conv3d(x, self.sobel_x, padding=1)
        grad_y = F.conv3d(x, self.sobel_y, padding=1)
        grad_z = F.conv3d(x, self.sobel_z, padding=1)
        grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + grad_z.pow(2) + self.eps)
        return torch.sigmoid(self.sigmoid_scale * (grad_mag - self.edge_threshold))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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