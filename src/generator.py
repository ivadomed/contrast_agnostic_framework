from __future__ import annotations

import torch
from torch import nn


class ConvNormAct3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvNormAct3D(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
        )
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + residual)


class EncoderBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            ConvNormAct3D(in_channels, out_channels),
            ResidualBlock3D(out_channels),
        )
        self.downsample = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(x)
        down = self.downsample(features)
        return features, down


class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.fuse = nn.Sequential(
            ConvNormAct3D(out_channels + skip_channels, out_channels),
            ResidualBlock3D(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class MRI_Synthesis_Net(nn.Module):
    """Moderate-size 3D residual U-Net for pseudo-contrast synthesis."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_filters: int = 32,
    ):
        super().__init__()
        widths = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8]

        self.stem = ConvNormAct3D(in_channels, widths[0])
        self.enc1 = EncoderBlock3D(widths[0], widths[0])
        self.enc2 = EncoderBlock3D(widths[0], widths[1])
        self.enc3 = EncoderBlock3D(widths[1], widths[2])

        self.bottleneck = nn.Sequential(
            ConvNormAct3D(widths[2], widths[3]),
            ResidualBlock3D(widths[3]),
            ResidualBlock3D(widths[3]),
        )

        self.dec3 = DecoderBlock3D(widths[3], widths[2], widths[2])
        self.dec2 = DecoderBlock3D(widths[2], widths[1], widths[1])
        self.dec1 = DecoderBlock3D(widths[1], widths[0], widths[0])

        self.head = nn.Sequential(
            ConvNormAct3D(widths[0], widths[0]),
            nn.Conv3d(widths[0], out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x[:, :1]

        x0 = self.stem(x)
        skip1, x1 = self.enc1(x0)
        skip2, x2 = self.enc2(x1)
        skip3, x3 = self.enc3(x2)

        x4 = self.bottleneck(x3)
        x = self.dec3(x4, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        residual = self.head(x)
        return torch.tanh(original_x + residual)
