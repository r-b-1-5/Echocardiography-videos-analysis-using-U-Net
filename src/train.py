"""Command-line entry point for training and evaluating the EchoNet U-Net model."""

from __future__ import annotations

import torch

from .config import config_from_args
from .dataset import build_dataloaders, prepare_splits
from .evaluate import save_all_evaluation_artifacts, save_run_summary
from .losses import soft_simpson_volume
from .models import build_model
from .training import restore_best_checkpoint, train_model
from .utils import get_device, print_environment, save_training_pair_example, seed_everything


def run(config):
    output_dir = config.resolved_output_dir
    seed_everything(config.seed)
    device = get_device()
    print_environment(config, device)

    prepared = prepare_splits(config)
    datasets, loaders = build_dataloaders(prepared, config, device)
    train_ds, val_ds, test_ds = datasets
    train_loader, val_loader, test_loader = loaders
    save_training_pair_example(train_ds, output_dir)

    model, model_source = build_model(device, image_size=config.image_size)

    # Differentiable Simpson sanity check from the notebook.
    probe = torch.zeros(
        2, config.image_size, config.image_size, device=device, requires_grad=True
    )
    with torch.no_grad():
        probe[:, 35:85, 42:70] = 0.8
    probe.requires_grad_(True)
    probe_vol = soft_simpson_volume(probe, n_discs=config.simpson_discs)
    probe_vol.sum().backward()
    print("Differentiable Simpson sanity check:")
    print("  Volumes (pixel³ relative units):", probe_vol.detach().cpu().numpy().round(2).tolist())
    print("  Gradient finite:", bool(torch.isfinite(probe.grad).all()) if probe.grad is not None else
          "gradient is non-leaf in this probe")
    print("  Discs per phase:", config.simpson_discs)
    print("  Training objective: CE + soft Dice + (EF MAE in percentage points / 100)")

    history, best_val, best_checkpoint = train_model(
        model, train_loader, val_loader, device, config
    )
    restore_best_checkpoint(model, best_checkpoint, device)
    metrics = save_all_evaluation_artifacts(
        model, history, prepared, test_loader, test_ds, device, config
    )
    save_run_summary(model, model_source, prepared, best_val, metrics, device, config)
    return model, metrics


def main(args=None):
    config = config_from_args(args)
    run(config)


if __name__ == "__main__":
    main()
