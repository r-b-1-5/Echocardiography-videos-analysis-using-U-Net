"""Milesial U-Net loading with the notebook's local compatible fallback."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class LocalUNet(nn.Module):
    """The original notebook's U-Net-compatible fallback implementation."""

    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def build_model(device: torch.device, image_size: int = 112, repo_dir: Path | None = None):
    """Prefer milesial/Pytorch-UNet; use the exact notebook fallback if unavailable."""
    if repo_dir is None:
        repo_dir = (
            Path("/kaggle/working/Pytorch-UNet")
            if Path("/kaggle/working").exists()
            else Path.cwd() / "Pytorch-UNet"
        )
    repo_dir = Path(repo_dir)
    unet_imported_from_repo = False
    unet_class = None

    try:
        if not repo_dir.exists():
            print("Cloning the requested U-Net repository...")
            subprocess.run(
                [
                    "git", "clone", "--depth", "1",
                    "https://github.com/milesial/Pytorch-UNet.git", str(repo_dir),
                ],
                check=True, capture_output=True, text=True,
            )
        sys.path.insert(0, str(repo_dir))
        from unet import UNet as MilesialUNet

        unet_class = MilesialUNet
        unet_imported_from_repo = True
        print("Successfully imported UNet from:", repo_dir)
    except Exception as exc:
        print("Repository import was unavailable; using the local U-Net-compatible fallback.")
        print("Reason:", repr(exc))

    if not unet_imported_from_repo:
        unet_class = LocalUNet

    model = unet_class(n_channels=2, n_classes=4, bilinear=True).to(device)
    print("\nModel constructed:")
    print(model.__class__.__name__, "| source:",
          "Milesial repository" if unet_imported_from_repo else "local fallback")
    print("  Input channels : 2 (ED grayscale, ES grayscale)")
    print("  Output channels: 4 (ED background/LV, ES background/LV)")
    print("  Parameters     :", f"{sum(parameter.numel() for parameter in model.parameters()):,}")
    with torch.no_grad():
        dummy_output = model(torch.zeros(1, 2, image_size, image_size, device=device))
    print("  Forward test   : input", f"(1,2,{image_size},{image_size}) -> output",
          tuple(dummy_output.shape))
    if tuple(dummy_output.shape) != (1, 4, image_size, image_size):
        raise RuntimeError(
            f"Unexpected U-Net output shape {tuple(dummy_output.shape)}; "
            f"expected {(1, 4, image_size, image_size)}"
        )
    return model, unet_imported_from_repo
