"""Training loop, validation objective, and checkpoint management."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .losses import combined_objective


def run_one_epoch(
    model,
    loader,
    device,
    optimizer,
    scaler,
    config,
    training: bool = True,
):
    model.train(training)
    sums = defaultdict(float)
    n_samples = 0
    all_ef_pred, all_ef_true = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_idx, (images, masks, clinical_ef) in enumerate(loader, start=1):
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            masks = masks.to(device, dtype=torch.long, non_blocking=True)
            clinical_ef = clinical_ef.to(device, dtype=torch.float32, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=config.use_amp):
                logits = model(images)
            # Geometry/loss calculations remain float32 for stable eigendecomposition/grid sampling.
            losses = combined_objective(
                logits.float(), masks, clinical_ef,
                ef_loss_weight=config.ef_loss_weight,
                n_discs=config.simpson_discs,
            )

            if training:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            batch_size_now = images.size(0)
            for key in ["total", "ce", "dice_loss", "dice", "ef_mae_pct"]:
                sums[key] += float(losses[key].detach().item()) * batch_size_now
            n_samples += batch_size_now
            all_ef_pred.extend(losses["ef_pred_soft"].detach().cpu().numpy().tolist())
            all_ef_true.extend(clinical_ef.detach().cpu().numpy().tolist())

            if training and batch_idx % 100 == 0:
                print(
                    f"    batch {batch_idx:4d}/{len(loader):4d} | "
                    f"total={losses['total'].item():.4f} | CE={losses['ce'].item():.4f} | "
                    f"DiceLoss={losses['dice_loss'].item():.4f} | "
                    f"EF-MAE={losses['ef_mae_pct'].item():.2f} pp"
                )

    return (
        {key: value / max(n_samples, 1) for key, value in sums.items()},
        np.asarray(all_ef_pred),
        np.asarray(all_ef_true),
    )


def train_model(model, train_loader, val_loader, device, config):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
    history = {
        "train_total": [], "val_total": [],
        "train_ce": [], "val_ce": [],
        "train_dice_loss": [], "val_dice_loss": [],
        "train_dice": [], "val_dice": [],
        "train_ef_mae": [], "val_ef_mae": [],
        "lr": [],
    }
    best_val = float("inf")
    best_checkpoint = config.resolved_output_dir / "best_unet_echonet.pth"

    print("Starting training")
    print(f"  Train studies : {len(train_loader.dataset):,}")
    print(f"  Val studies   : {len(val_loader.dataset):,}")
    print(f"  Epochs        : {config.epochs}")
    print(f"  Batch size    : {config.batch_size}")
    print(f"  Learning rate : {config.learning_rate}")
    print("  Loss          : CrossEntropy + SoftDice + EF_MAE/100")

    for epoch in range(1, config.epochs + 1):
        train_metrics, _, _ = run_one_epoch(
            model, train_loader, device, optimizer, scaler, config, training=True
        )
        val_metrics, _, _ = run_one_epoch(
            model, val_loader, device, optimizer, scaler, config, training=False
        )
        scheduler.step(val_metrics["total"])
        current_lr = optimizer.param_groups[0]["lr"]

        for key in ["total", "ce", "dice_loss", "dice", "ef_mae_pct"]:
            history_key = "ef_mae" if key == "ef_mae_pct" else key
            history[f"train_{history_key}"].append(train_metrics[key])
            history[f"val_{history_key}"].append(val_metrics[key])
        history["lr"].append(current_lr)

        print(f"\nEpoch {epoch:02d}/{config.epochs}")
        print(
            f"  TRAIN | total {train_metrics['total']:.4f} | CE {train_metrics['ce']:.4f} | "
            f"DiceLoss {train_metrics['dice_loss']:.4f} | Dice {train_metrics['dice']:.4f} | "
            f"EF-MAE {train_metrics['ef_mae_pct']:.2f} percentage points"
        )
        print(
            f"  VAL   | total {val_metrics['total']:.4f} | CE {val_metrics['ce']:.4f} | "
            f"DiceLoss {val_metrics['dice_loss']:.4f} | Dice {val_metrics['dice']:.4f} | "
            f"EF-MAE {val_metrics['ef_mae_pct']:.2f} percentage points"
        )
        print(f"  Learning rate: {current_lr:.2e}")

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_total": best_val,
                "history": history,
            }, best_checkpoint)
            print(
                f"  >>> Best checkpoint saved: {best_checkpoint.name} "
                f"(val total={best_val:.4f})"
            )

    print("\nTraining completed.")
    print("Best validation combined loss:", f"{best_val:.4f}")
    print("Checkpoint:", best_checkpoint)
    return history, best_val, best_checkpoint


def restore_best_checkpoint(model, checkpoint_path: Path, device):
    """Restore the checkpoint selected by the combined validation objective."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        f"Loaded best checkpoint from epoch {checkpoint['epoch']} "
        f"| val total={checkpoint['best_val_total']:.4f}"
    )
    return checkpoint
