"""Small shared utilities used by training and evaluation."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def seed_everything(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds (as in the notebook)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_environment(config, device: torch.device) -> None:
    import cv2

    print("Environment initialized")
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  OpenCV version  : {cv2.__version__}")
    print(f"  Device          : {device}")
    if torch.cuda.is_available():
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
    print(f"  AMP enabled     : {config.use_amp}")
    print(f"  Output folder   : {config.resolved_output_dir.resolve()}")
    print("\nSplit customization:")
    print(f"  Train studies requested: {config.train_limit if config.train_limit is not None else 'ALL'}")
    print(f"  Val studies requested  : {config.val_limit if config.val_limit is not None else 'ALL'}")
    print(f"  Test studies requested : {config.test_limit if config.test_limit is not None else 'ALL'}")


def save_training_pair_example(dataset, output_dir: Path) -> None:
    """Save the notebook's example ED/ES image and reference-mask visualization."""
    example_x, example_y, example_ef = dataset[0]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    for col, (phase_idx, phase_name) in enumerate([(0, "ED"), (1, "ES")]):
        axes[2 * col].imshow(example_x[phase_idx], cmap="gray")
        axes[2 * col].set_title(f"{phase_name} frame")
        axes[2 * col + 1].imshow(example_x[phase_idx], cmap="gray")
        axes[2 * col + 1].contour(
            example_y[phase_idx].numpy(), levels=[0.5], colors=["lime"], linewidths=1.5
        )
        axes[2 * col + 1].set_title(f"{phase_name} + reconstructed LV mask")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Example paired study | clinical EF = {example_ef.item():.1f}%")
    plt.tight_layout()
    fig.savefig(output_dir / "example_training_pair.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_pipeline_flowchart(output_dir: Path) -> None:
    """Save the same end-to-end process schematic shown in the notebook."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    nodes = [
        (0.5, 7.1, 3.0, 1.0, "EchoNet-Dynamic\nAVI + FileList + VolumeTracings"),
        (4.0, 7.1, 2.8, 1.0, "Official split selection\nTRAIN / VAL / TEST"),
        (7.3, 7.1, 2.8, 1.0, "ED / ES frame extraction\nLargest / smallest traced LV area"),
        (10.6, 7.1, 2.8, 1.0, "Rasterize line tracings\nReference LV masks"),
        (4.0, 4.9, 3.0, 1.1, "Two-channel Milesial U-Net\nInput: ED + ES frames"),
        (8.0, 4.9, 2.7, 1.1, "Two-phase LV segmentation\nED mask + ES mask"),
        (10.0, 2.7, 3.2, 1.1, "20-disc single-view Simpson\nEDV / ESV (pixel³)"),
        (5.8, 2.7, 3.2, 1.1, "EF = 100 × (EDV−ESV)/EDV\nMAE vs clinical EF label"),
        (1.2, 2.7, 3.4, 1.1, "Combined training objective\nCross-Entropy + Soft Dice + EF MAE"),
        (4.3, 0.5, 3.2, 1.1, "Test metrics + plots\nDice, IoU, EF MAE, ROC, Bland–Altman"),
        (8.2, 0.5, 4.4, 1.1, "Task classification + examples\nEF < 50% → HFrEF_task; 5 visual cases"),
    ]
    for x0, y0, width, height, label in nodes:
        box = FancyBboxPatch(
            (x0, y0), width, height,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.3, edgecolor="#334155", facecolor="#e8f0fb",
        )
        ax.add_patch(box)
        ax.text(x0 + width / 2, y0 + height / 2, label, ha="center", va="center", fontsize=9)
    arrows = [
        ((3.5, 7.6), (4.0, 7.6)), ((6.8, 7.6), (7.3, 7.6)), ((10.1, 7.6), (10.6, 7.6)),
        ((12.0, 7.1), (9.4, 6.0)), ((7.0, 5.45), (8.0, 5.45)),
        ((10.7, 4.9), (11.5, 3.8)), ((10.0, 3.25), (9.0, 3.25)),
        ((5.8, 3.25), (4.6, 3.25)), ((2.9, 2.7), (5.7, 1.6)),
        ((7.5, 1.05), (8.2, 1.05)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=13,
            linewidth=1.2, color="#475569", connectionstyle="arc3,rad=0.0",
        ))
    ax.text(
        7, 8.65,
        "EchoNet-Dynamic LV Segmentation → EF Estimation → Reduced-EF Task Classification",
        ha="center", va="center", fontsize=14, fontweight="bold",
    )
    ax.text(
        7, 0.08,
        "Note: A4C single-view approximation; not biplane clinical EF. Volumes are relative pixel³ units.",
        ha="center", va="center", fontsize=8, color="#7f1d1d",
    )
    plt.tight_layout()
    fig.savefig(output_dir / "pipeline_flowchart.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
