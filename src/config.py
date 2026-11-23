"""Configuration and command-line arguments for EchoNet U-Net."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class RunConfig:
    """Runtime settings. Defaults match the listed original notebook."""

    data_root: Optional[str] = None
    output_dir: Optional[str] = None
    train_limit: Optional[int] = None
    val_limit: Optional[int] = None
    test_limit: Optional[int] = None
    seed: int = 42
    image_size: int = 112
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_workers: int = 0
    use_amp: Optional[bool] = None
    simpson_discs: int = 20
    ef_loss_weight: float = 1.0
    task_ef_threshold: float = 50.0

    def __post_init__(self) -> None:
        if self.use_amp is None:
            self.use_amp = torch.cuda.is_available()
        if self.output_dir is None:
            default_output = (
                Path("/kaggle/working/echonet_results")
                if Path("/kaggle/working").exists()
                else Path.cwd() / "echonet_results"
            )
            self.output_dir = str(default_output)
        self.validate()

    def validate(self) -> None:
        for name in ("train_limit", "val_limit", "test_limit"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0 or None, got {value}")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if self.simpson_discs <= 0:
            raise ValueError("simpson_discs must be positive")

    @property
    def resolved_output_dir(self) -> Path:
        path = Path(self.output_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the EchoNet-Dynamic U-Net model for paired ED/ES "
            "LV segmentation and Simpson EF estimation."
        )
    )
    parser.add_argument("--data-root", type=str, default=None,
                        help="Folder containing FileList.csv, VolumeTracings.csv, and Videos/. "
                             "When omitted, Kaggle/Colab/current-directory discovery is used.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where checkpoints, metrics, CSV files, and plots are saved.")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="Maximum training studies sampled from the official train split; default: all.")
    parser.add_argument("--val-limit", type=int, default=None,
                        help="Maximum validation studies sampled from the official validation split; default: all.")
    parser.add_argument("--test-limit", type=int, default=None,
                        help="Maximum test studies sampled from the official test split; default: all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers; 0 is the notebook/Kaggle default.")
    parser.add_argument("--disable-amp", action="store_true",
                        help="Disable CUDA automatic mixed precision.")
    parser.add_argument("--simpson-discs", type=int, default=20)
    parser.add_argument("--ef-loss-weight", type=float, default=1.0)
    parser.add_argument("--task-ef-threshold", type=float, default=50.0)
    return parser


def config_from_args(args: Optional[list[str]] = None) -> RunConfig:
    parsed = build_arg_parser().parse_args(args=args)
    return RunConfig(
        data_root=parsed.data_root,
        output_dir=parsed.output_dir,
        train_limit=parsed.train_limit,
        val_limit=parsed.val_limit,
        test_limit=parsed.test_limit,
        seed=parsed.seed,
        image_size=parsed.image_size,
        batch_size=parsed.batch_size,
        epochs=parsed.epochs,
        learning_rate=parsed.learning_rate,
        weight_decay=parsed.weight_decay,
        num_workers=parsed.num_workers,
        use_amp=(torch.cuda.is_available() and not parsed.disable_amp),
        simpson_discs=parsed.simpson_discs,
        ef_loss_weight=parsed.ef_loss_weight,
        task_ef_threshold=parsed.task_ef_threshold,
    )
