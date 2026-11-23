"""Held-out evaluation, Simpson EF estimation, plots, and run artifacts."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .utils import save_pipeline_flowchart


def simpson_volume_from_binary_mask(mask, n_discs=20):
    """Hard-mask single-view Simpson volume: (volume_px3, long_axis_px, disc widths)."""
    mask = np.asarray(mask).astype(bool)
    ys, xs = np.where(mask)
    if len(xs) < 30:
        return np.nan, np.nan, np.full(n_discs, np.nan)
    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, np.argmax(eigenvalues)]
    minor = eigenvectors[:, np.argmin(eigenvalues)]
    s = centered @ major
    t = centered @ minor
    s_min, s_max = np.percentile(s, [1, 99])
    long_axis = float(s_max - s_min)
    if not np.isfinite(long_axis) or long_axis < 3:
        return np.nan, np.nan, np.full(n_discs, np.nan)

    edges = np.linspace(s_min, s_max, n_discs + 1)
    diameters = np.full(n_discs, np.nan, dtype=np.float64)
    for index in range(n_discs):
        in_disc = (s >= edges[index]) & (
            s < edges[index + 1] if index < n_discs - 1 else s <= edges[index + 1]
        )
        transverse = t[in_disc]
        if len(transverse) >= 3:
            q_low, q_high = np.percentile(transverse, [5, 95])
            diameters[index] = max(float(q_high - q_low), 0.0)

    valid = np.isfinite(diameters)
    if valid.sum() < max(8, n_discs // 2):
        return np.nan, long_axis, diameters
    # Interpolate occasional thin apical bins with neighboring valid measurements.
    indices = np.arange(n_discs)
    diameters[~valid] = np.interp(indices[~valid], indices[valid], diameters[valid])
    volume = (math.pi / 4.0) * (long_axis / n_discs) * np.square(diameters).sum()
    return float(volume), long_axis, diameters


@torch.no_grad()
def predict_split(model, loader, dataset_obj, device, n_discs=20, split_name="test"):
    model.eval()
    all_images, all_masks, all_probs = [], [], []
    all_clinical_ef, all_soft_ef = [], []
    for images, masks, clinical_ef in loader:
        images = images.to(device, dtype=torch.float32)
        logits = model(images).float()
        batch_size, _, height, width = logits.shape
        logits_pair = logits.view(batch_size, 2, 2, height, width)
        foreground = torch.softmax(logits_pair, dim=2)[:, :, 1]
        from .losses import soft_simpson_volume
        edv_soft = soft_simpson_volume(foreground[:, 0], n_discs=n_discs)
        esv_soft = soft_simpson_volume(foreground[:, 1], n_discs=n_discs)
        ef_soft = 100.0 * (edv_soft - esv_soft) / (edv_soft + 1e-6)

        all_images.append(images.cpu().numpy())
        all_masks.append(masks.numpy())
        all_probs.append(foreground.cpu().numpy())
        all_clinical_ef.extend(clinical_ef.numpy().tolist())
        all_soft_ef.extend(ef_soft.cpu().numpy().tolist())

    images_np = np.concatenate(all_images, axis=0)
    masks_np = np.concatenate(all_masks, axis=0)
    probs_np = np.concatenate(all_probs, axis=0)
    clinical_ef_np = np.asarray(all_clinical_ef, dtype=np.float32)
    soft_ef_np = np.asarray(all_soft_ef, dtype=np.float32)
    predicted_masks = (probs_np >= 0.5).astype(np.uint8)

    hard_edv, hard_esv, hard_ef, ed_lengths, es_lengths = [], [], [], [], []
    ed_widths_all, es_widths_all = [], []
    for index in range(len(predicted_masks)):
        v_ed, l_ed, w_ed = simpson_volume_from_binary_mask(
            predicted_masks[index, 0], n_discs=n_discs
        )
        v_es, l_es, w_es = simpson_volume_from_binary_mask(
            predicted_masks[index, 1], n_discs=n_discs
        )
        if np.isfinite(v_ed) and np.isfinite(v_es) and v_ed > 0:
            ef_hard = 100.0 * (v_ed - v_es) / v_ed
        else:
            ef_hard = np.nan
        hard_edv.append(v_ed)
        hard_esv.append(v_es)
        hard_ef.append(ef_hard)
        ed_lengths.append(l_ed)
        es_lengths.append(l_es)
        ed_widths_all.append(w_ed)
        es_widths_all.append(w_es)

    hard_ef_np = np.asarray(hard_ef, dtype=np.float32)
    # Keep invalid hard-mask EF values explicit; do not hide them in the availability count.
    return {
        "images": images_np,
        "masks_true": masks_np,
        "masks_pred": predicted_masks,
        "probs": probs_np,
        "clinical_ef": clinical_ef_np,
        "soft_ef": soft_ef_np,
        "hard_edv_px3": np.asarray(hard_edv),
        "hard_esv_px3": np.asarray(hard_esv),
        "hard_ef": hard_ef_np,
        "ed_lengths": np.asarray(ed_lengths),
        "es_lengths": np.asarray(es_lengths),
        "ed_widths": np.asarray(ed_widths_all),
        "es_widths": np.asarray(es_widths_all),
        "meta": dataset_obj.meta.copy(),
    }


def save_training_plots(history: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    epoch_axis = np.arange(1, len(history["train_total"]) + 1)
    axes[0, 0].plot(epoch_axis, history["train_total"], label="Train total")
    axes[0, 0].plot(epoch_axis, history["val_total"], label="Val total")
    axes[0, 0].set_title("Combined objective")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(epoch_axis, history["train_ce"], label="Train CE")
    axes[0, 1].plot(epoch_axis, history["val_ce"], label="Val CE")
    axes[0, 1].plot(epoch_axis, history["train_dice_loss"], "--", label="Train soft Dice loss")
    axes[0, 1].plot(epoch_axis, history["val_dice_loss"], "--", label="Val soft Dice loss")
    axes[0, 1].set_title("Segmentation loss components")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(epoch_axis, history["train_dice"], label="Train Dice")
    axes[1, 0].plot(epoch_axis, history["val_dice"], label="Val Dice")
    axes[1, 0].set_title("Foreground LV Dice")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Dice")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(epoch_axis, history["train_ef_mae"], label="Train EF MAE")
    axes[1, 1].plot(epoch_axis, history["val_ef_mae"], label="Val EF MAE")
    axes[1, 1].set_title("EF MAE (percentage points)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("MAE (%)")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(epoch_axis, history["lr"], marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning-rate schedule")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_dir / "learning_rate.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_reference_simpson_demo(prepared: dict, n_discs: int, output_dir: Path) -> None:
    example_mask = prepared["train"]["masks"][0, 0]
    example_volume, example_length, example_widths = simpson_volume_from_binary_mask(
        example_mask, n_discs=n_discs
    )
    print("Hard-mask Simpson geometry check (training reference ED mask):")
    print("  Long axis (pixels) :", round(example_length, 2))
    print("  20 widths (pixels) :", np.round(example_widths, 2).tolist())
    print("  ED volume (pixel³) :", round(example_volume, 2))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(np.arange(1, n_discs + 1), example_widths)
    ax.set_xlabel("Short-axis disc number (apex → base)")
    ax.set_ylabel("Diameter (pixels)")
    ax.set_title("20 short-axis diameters from one reference LV mask")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_dir / "reference_simpson_geometry.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_test_split(model, test_loader, test_ds, device, config):
    output_dir = config.resolved_output_dir
    test_results = predict_split(
        model, test_loader, test_ds, device,
        n_discs=config.simpson_discs, split_name="test",
    )
    print("Test predictions generated:")
    for key in [
        "images", "masks_true", "masks_pred", "probs", "clinical_ef", "soft_ef",
        "hard_edv_px3", "hard_esv_px3", "hard_ef",
    ]:
        print(f"  {key:16s}: {np.asarray(test_results[key]).shape}")
    print("  Hard-mask EF available:", int(np.isfinite(test_results["hard_ef"]).sum()), "/", len(test_ds))

    # Pixel-wise segmentation metrics over both ED and ES phases.
    y_true = test_results["masks_true"].reshape(-1).astype(np.uint8)
    y_pred = test_results["masks_pred"].reshape(-1).astype(np.uint8)
    intersection = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    union = int(np.logical_or(y_true == 1, y_pred == 1).sum())
    seg_dice = (2 * intersection) / (int((y_true == 1).sum()) + int((y_pred == 1).sum()) + 1e-8)
    seg_iou = intersection / (union + 1e-8)

    per_image_dice, per_image_iou = [], []
    for index in range(len(test_ds)):
        for phase in [0, 1]:
            gt = test_results["masks_true"][index, phase].astype(bool)
            pr = test_results["masks_pred"][index, phase].astype(bool)
            inter = np.logical_and(gt, pr).sum()
            denom = gt.sum() + pr.sum()
            uni = np.logical_or(gt, pr).sum()
            per_image_dice.append((2 * inter) / (denom + 1e-8))
            per_image_iou.append(inter / (uni + 1e-8))

    print("\nSegmentation metrics on test split:")
    print(f"  Global foreground Dice : {seg_dice:.4f}")
    print(f"  Global foreground IoU  : {seg_iou:.4f}")
    print(f"  Mean per-frame Dice     : {np.mean(per_image_dice):.4f} ± {np.std(per_image_dice):.4f}")
    print(f"  Mean per-frame IoU      : {np.mean(per_image_iou):.4f} ± {np.std(per_image_iou):.4f}")

    clinical = test_results["clinical_ef"]
    soft_pred = np.clip(test_results["soft_ef"], 0, 100)
    hard_pred = test_results["hard_ef"]
    valid_hard = np.isfinite(hard_pred)
    soft_ef_mae = mean_absolute_error(clinical, soft_pred)
    hard_ef_mae = mean_absolute_error(clinical[valid_hard], hard_pred[valid_hard]) if valid_hard.any() else np.nan
    hard_ef_rmse = (
        mean_squared_error(clinical[valid_hard], hard_pred[valid_hard]) ** 0.5
        if valid_hard.any() else np.nan
    )
    hard_ef_r2 = r2_score(clinical[valid_hard], hard_pred[valid_hard]) if valid_hard.sum() > 1 else np.nan
    print("\nEF comparison against released clinical EF label:")
    print(f"  Soft-mask Simpson EF MAE: {soft_ef_mae:.2f} percentage points")
    if valid_hard.sum() > 0:
        print(f"  Hard-mask Simpson EF MAE: {hard_ef_mae:.2f} percentage points")
        print(f"  Hard-mask EF RMSE       : {hard_ef_rmse:.2f} percentage points")
        print(f"  Hard-mask EF R²         : {hard_ef_r2:.3f}")
    else:
        print("  No valid hard-mask EF estimates; inspect predicted masks and adjust the mask threshold / training.")

    # Save row-level test results for reproducibility.
    results_table = test_results["meta"].copy()
    results_table["EF_clinical_pct"] = clinical
    results_table["EF_soft_simpson_pct"] = soft_pred
    results_table["EDV_hard_pixel3"] = test_results["hard_edv_px3"]
    results_table["ESV_hard_pixel3"] = test_results["hard_esv_px3"]
    results_table["EF_hard_simpson_pct"] = hard_pred
    results_table["HFrEF_task_true_EF_lt_50"] = clinical < config.task_ef_threshold
    results_table["HFrEF_task_pred_EF_lt_50"] = hard_pred < config.task_ef_threshold
    results_csv = output_dir / "test_predictions.csv"
    results_table.to_csv(results_csv, index=False)
    print("\nSaved study-level test predictions to:", results_csv)

    # EF agreement and Bland–Altman analysis using valid hard-mask EF estimates.
    if valid_hard.sum() > 1:
        true_valid = clinical[valid_hard]
        pred_valid = hard_pred[valid_hard]
        mean_pair = (true_valid + pred_valid) / 2.0
        diff_pair = pred_valid - true_valid
        bias = float(np.mean(diff_pair))
        sd_diff = float(np.std(diff_pair, ddof=1))
        loa_low, loa_high = bias - 1.96 * sd_diff, bias + 1.96 * sd_diff
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(true_valid, pred_valid, alpha=0.55, s=22)
        axes[0].plot([0, 100], [0, 100], "k--", linewidth=1)
        axes[0].set_xlim(0, 100)
        axes[0].set_ylim(0, 100)
        axes[0].set_xlabel("Clinical EF (%)")
        axes[0].set_ylabel("Hard-mask Simpson EF (%)")
        axes[0].set_title("EF agreement: predicted vs clinical")
        axes[0].grid(alpha=0.25)
        axes[1].scatter(mean_pair, diff_pair, alpha=0.55, s=22)
        axes[1].axhline(bias, color="black", linestyle="--", label=f"Bias = {bias:.1f}")
        axes[1].axhline(loa_low, color="red", linestyle=":", label=f"Lower 95% LoA = {loa_low:.1f}")
        axes[1].axhline(loa_high, color="red", linestyle=":", label=f"Upper 95% LoA = {loa_high:.1f}")
        axes[1].set_xlabel("Mean of clinical and predicted EF (%)")
        axes[1].set_ylabel("Predicted − clinical EF (%)")
        axes[1].set_title("Bland–Altman plot")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.25)
        plt.tight_layout()
        fig.savefig(output_dir / "ef_agreement_bland_altman.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Bland–Altman bias: {bias:.2f} percentage points")
        print(f"95% limits of agreement: [{loa_low:.2f}, {loa_high:.2f}]")

    # Task-specific binary classification: EF < 50% is positive by user request.
    valid_cls = np.isfinite(hard_pred)
    y_true_cls = (clinical[valid_cls] < config.task_ef_threshold).astype(int)
    y_pred_cls = (hard_pred[valid_cls] < config.task_ef_threshold).astype(int)
    print(f"\nTask-defined reduced-EF classification (threshold EF < {config.task_ef_threshold:.0f}%):")
    classification_metrics = {}
    if len(y_true_cls) > 0:
        classification_metrics = {
            "accuracy": accuracy_score(y_true_cls, y_pred_cls),
            "precision": precision_score(y_true_cls, y_pred_cls, zero_division=0),
            "recall": recall_score(y_true_cls, y_pred_cls, zero_division=0),
            "f1": f1_score(y_true_cls, y_pred_cls, zero_division=0),
        }
        print("  Samples evaluated:", len(y_true_cls))
        print("  Accuracy :", f"{classification_metrics['accuracy']:.4f}")
        print("  Precision:", f"{classification_metrics['precision']:.4f}")
        print("  Recall   :", f"{classification_metrics['recall']:.4f}")
        print("  F1       :", f"{classification_metrics['f1']:.4f}")
        print("\nClassification report:")
        print(classification_report(
            y_true_cls, y_pred_cls, labels=[0, 1],
            target_names=["EF ≥ 50%", "EF < 50%"], zero_division=0,
        ))
        cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["Pred ≥50%", "Pred <50%"])
        ax.set_yticks([0, 1], ["True ≥50%", "True <50%"])
        ax.set_title("Confusion matrix — task EF < 50%")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig.savefig(output_dir / "hfre_task_confusion_matrix.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        if len(np.unique(y_true_cls)) == 2:
            reduced_ef_score = config.task_ef_threshold - hard_pred[valid_cls]
            auc = roc_auc_score(y_true_cls, reduced_ef_score)
            fpr, tpr, _ = roc_curve(y_true_cls, reduced_ef_score)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("ROC — task-defined EF < 50%")
            ax.legend()
            ax.grid(alpha=0.25)
            plt.tight_layout()
            fig.savefig(output_dir / "hfre_task_roc.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
            classification_metrics["roc_auc"] = auc
            print("  ROC-AUC  :", f"{auc:.4f}")
        else:
            print("  ROC-AUC not calculated: the test subset contains only one target class.")
    else:
        print("  No valid hard-mask EF estimates to classify.")

    # Distribution plots for volumes and EF.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(test_results["hard_edv_px3"][np.isfinite(test_results["hard_edv_px3"])], bins=25)
    axes[0].set_title("Estimated EDV")
    axes[0].set_xlabel("pixel³ (relative)")
    axes[1].hist(test_results["hard_esv_px3"][np.isfinite(test_results["hard_esv_px3"])], bins=25)
    axes[1].set_title("Estimated ESV")
    axes[1].set_xlabel("pixel³ (relative)")
    axes[2].hist(hard_pred[np.isfinite(hard_pred)], bins=25)
    axes[2].axvline(config.task_ef_threshold, color="red", linestyle="--", label="50% task cutoff")
    axes[2].set_title("Hard-mask Simpson EF")
    axes[2].set_xlabel("EF (%)")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(output_dir / "volume_ef_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Show/save the same five held-out examples with reference and predicted contours.
    n_show = min(5, len(test_ds))
    fig, axes = plt.subplots(n_show, 4, figsize=(14, 3.2 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    for index in range(n_show):
        ed_img = test_results["images"][index, 0]
        es_img = test_results["images"][index, 1]
        ed_gt = test_results["masks_true"][index, 0]
        es_gt = test_results["masks_true"][index, 1]
        ed_pr = test_results["masks_pred"][index, 0]
        es_pr = test_results["masks_pred"][index, 1]
        axes[index, 0].imshow(ed_img, cmap="gray")
        axes[index, 0].contour(ed_gt, levels=[0.5], colors=["lime"], linewidths=1.4)
        axes[index, 0].set_title("ED frame + reference contour")
        axes[index, 1].imshow(ed_img, cmap="gray")
        axes[index, 1].contour(ed_pr, levels=[0.5], colors=["red"], linewidths=1.4)
        axes[index, 1].set_title("ED frame + predicted contour")
        axes[index, 2].imshow(es_img, cmap="gray")
        axes[index, 2].contour(es_gt, levels=[0.5], colors=["lime"], linewidths=1.4)
        axes[index, 2].set_title("ES frame + reference contour")
        axes[index, 3].imshow(es_img, cmap="gray")
        axes[index, 3].contour(es_pr, levels=[0.5], colors=["red"], linewidths=1.4)
        axes[index, 3].set_title("ES frame + predicted contour")
        for ax in axes[index]:
            ax.axis("off")
        ef_h = test_results["hard_ef"][index]
        ef_h_txt = f"{ef_h:.1f}%" if np.isfinite(ef_h) else "N/A"
        axes[index, 0].set_ylabel(
            f"Study {index + 1}\nClinical EF {test_results['clinical_ef'][index]:.1f}%\n"
            f"Soft EF {test_results['soft_ef'][index]:.1f}% | Hard EF {ef_h_txt}",
            rotation=0, labelpad=95, va="center", fontsize=9,
        )
    fig.suptitle("Five held-out test examples: LV segmentation at ED and ES", y=1.002, fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "five_test_segmentation_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Predicted 20-disc geometry for one held-out case.
    case_idx = 0
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(test_results["images"][case_idx, 0], cmap="gray")
    axes[0].contour(test_results["masks_pred"][case_idx, 0], levels=[0.5], colors=["red"], linewidths=1.5)
    axes[0].set_title("Predicted ED LV mask")
    axes[1].imshow(test_results["images"][case_idx, 1], cmap="gray")
    axes[1].contour(test_results["masks_pred"][case_idx, 1], levels=[0.5], colors=["red"], linewidths=1.5)
    axes[1].set_title("Predicted ES LV mask")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_dir / "predicted_ed_es_masks.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(1, config.simpson_discs + 1)
    ax.plot(x, test_results["ed_widths"][case_idx], marker="o", label="ED short-axis widths")
    ax.plot(x, test_results["es_widths"][case_idx], marker="s", label="ES short-axis widths")
    ax.set_xlabel("Disc index (apex → base)")
    ax.set_ylabel("Width (pixels)")
    ax.set_title(f"{config.simpson_discs} short-axis diameters used in Simpson volume estimation")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_dir / "simpson_disc_widths.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Displayed", n_show, "test studies and plotted their 20-disc ED/ES diameter profiles.")

    return {
        "test_results": test_results,
        "seg_dice": seg_dice,
        "seg_iou": seg_iou,
        "mean_per_frame_dice": float(np.mean(per_image_dice)),
        "std_per_frame_dice": float(np.std(per_image_dice)),
        "mean_per_frame_iou": float(np.mean(per_image_iou)),
        "std_per_frame_iou": float(np.std(per_image_iou)),
        "clinical": clinical,
        "soft_pred": soft_pred,
        "hard_pred": hard_pred,
        "valid_hard": valid_hard,
        "soft_ef_mae": soft_ef_mae,
        "hard_ef_mae": hard_ef_mae,
        "hard_ef_rmse": hard_ef_rmse,
        "hard_ef_r2": hard_ef_r2,
        "classification_metrics": classification_metrics,
    }


def save_run_summary(model, model_source, prepared, best_val, metrics, device, config) -> Path:
    output_dir = config.resolved_output_dir
    final_model_path = output_dir / "best_unet_echonet_state_dict.pth"
    torch.save(model.state_dict(), final_model_path)

    test_results = metrics["test_results"]
    valid_hard = metrics["valid_hard"]
    summary_lines = [
        "EchoNet-Dynamic LV segmentation / EF experiment summary",
        f"Device: {device}",
        f"Model source: {'milesial/Pytorch-UNet' if model_source else 'local U-Net-compatible fallback'}",
        "Usable studies: "
        f"train={len(prepared['train']['ef'])}, val={len(prepared['val']['ef'])}, test={len(prepared['test']['ef'])}",
        f"Best validation combined loss: {best_val:.6f}",
        f"Test global Dice: {metrics['seg_dice']:.6f}",
        f"Test global IoU: {metrics['seg_iou']:.6f}",
        f"Test soft-mask EF MAE (pp): {metrics['soft_ef_mae']:.4f}",
        f"Test hard-mask EF valid count: {int(valid_hard.sum())}/{len(prepared['test']['ef'])}",
        f"Task EF threshold: EF < {config.task_ef_threshold:.1f}%",
        "EF method: single-view 20-disc Simpson approximation from predicted binary LV masks",
        "EDV/ESV units: pixel^3 relative units (no physical pixel spacing in this dataset release)",
    ]
    if valid_hard.sum() > 0:
        summary_lines.append(f"Test hard-mask EF MAE (pp): {metrics['hard_ef_mae']:.4f}")
    summary_path = output_dir / "run_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n" + "\n".join(summary_lines))
    print("\nSaved output files:")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name:45s} {path.stat().st_size / (1024 ** 2):8.2f} MB")
    print("\nPython run complete.")
    return summary_path


def save_all_evaluation_artifacts(model, history, prepared, test_loader, test_ds, device, config):
    save_training_plots(history, config.resolved_output_dir)
    save_reference_simpson_demo(prepared, config.simpson_discs, config.resolved_output_dir)
    metrics = evaluate_test_split(model, test_loader, test_ds, device, config)
    save_pipeline_flowchart(config.resolved_output_dir)
    return metrics
