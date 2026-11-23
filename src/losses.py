"""Segmentation and differentiable 20-disc Simpson EF objective."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def soft_simpson_volume(prob, n_discs=20, n_samples=160, temperature_px=2.5):
    """Differentiable single-view 20-disc volume estimate from [B,H,W] probabilities."""
    if prob.ndim != 3:
        raise ValueError(f"Expected [B,H,W] probabilities; got {tuple(prob.shape)}")
    batch_size, height, width = prob.shape
    dtype, device = prob.dtype, prob.device
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    p = prob.reshape(batch_size, -1).clamp(1e-6, 1.0)
    weights = p.pow(2.0)
    mass = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    centroid = (weights.unsqueeze(-1) * coords.unsqueeze(0)).sum(dim=1) / mass
    centered = coords.unsqueeze(0) - centroid.unsqueeze(1)
    covariance = torch.einsum("bn,bni,bnj->bij", weights, centered, centered) / mass.unsqueeze(-1)

    # Detach PCA eigenvectors for numerical stability near nearly circular early predictions.
    # Gradients still flow through the probability-weighted center, projected extents,
    # line samples, and disc diameters.
    _, eigenvectors = torch.linalg.eigh(covariance)
    major = eigenvectors[:, :, 1].detach()
    minor = eigenvectors[:, :, 0].detach()
    s = (centered * major.unsqueeze(1)).sum(dim=-1)
    t = (centered * minor.unsqueeze(1)).sum(dim=-1)

    logw = 2.0 * torch.log(p.clamp_min(1e-6))
    tau = float(temperature_px)
    soft_s_min = (torch.softmax(logw - s / tau, dim=1) * s).sum(dim=1)
    soft_s_max = (torch.softmax(logw + s / tau, dim=1) * s).sum(dim=1)
    soft_t_min = (torch.softmax(logw - t / tau, dim=1) * t).sum(dim=1)
    soft_t_max = (torch.softmax(logw + t / tau, dim=1) * t).sum(dim=1)

    long_axis = (soft_s_max - soft_s_min).clamp(min=4.0, max=float(2 * max(height, width)))
    transverse_center = 0.5 * (soft_t_max + soft_t_min)
    transverse_span = (soft_t_max - soft_t_min).clamp(min=4.0, max=float(2 * max(height, width)))
    transverse_radius = 0.60 * transverse_span

    disc_fraction = (torch.arange(n_discs, device=device, dtype=dtype) + 0.5) / n_discs
    s_centers = soft_s_min.unsqueeze(1) + disc_fraction.unsqueeze(0) * long_axis.unsqueeze(1)
    t_offsets = torch.linspace(-1.0, 1.0, n_samples, device=device, dtype=dtype).view(1, 1, -1)
    t_values = transverse_center.view(batch_size, 1, 1) + transverse_radius.view(batch_size, 1, 1) * t_offsets

    sample_x = (
        centroid[:, 0].view(batch_size, 1, 1)
        + s_centers.unsqueeze(-1) * major[:, 0].view(batch_size, 1, 1)
        + t_values * minor[:, 0].view(batch_size, 1, 1)
    )
    sample_y = (
        centroid[:, 1].view(batch_size, 1, 1)
        + s_centers.unsqueeze(-1) * major[:, 1].view(batch_size, 1, 1)
        + t_values * minor[:, 1].view(batch_size, 1, 1)
    )
    grid_x = 2.0 * sample_x / max(width - 1, 1) - 1.0
    grid_y = 2.0 * sample_y / max(height - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    sampled = F.grid_sample(
        prob.unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ).squeeze(1)
    step = (2.0 * transverse_radius / max(n_samples - 1, 1)).unsqueeze(1)
    diameters = sampled.sum(dim=-1) * step
    volume = (math.pi / 4.0) * (long_axis / n_discs) * diameters.square().sum(dim=1)
    return volume.clamp_min(1e-5)


def segmentation_losses(logits, target_masks):
    """Cross-Entropy + soft Dice for a pair of binary ED/ES masks."""
    batch_size, _, height, width = logits.shape
    logits_pair = logits.view(batch_size, 2, 2, height, width)
    flat_logits = logits_pair.reshape(batch_size * 2, 2, height, width)
    flat_target = target_masks.reshape(batch_size * 2, height, width).long()
    ce = F.cross_entropy(flat_logits, flat_target)
    foreground_prob = torch.softmax(flat_logits, dim=1)[:, 1]
    target_fg = (flat_target == 1).float()
    dims = (1, 2)
    intersection = (foreground_prob * target_fg).sum(dim=dims)
    denominator = foreground_prob.sum(dim=dims) + target_fg.sum(dim=dims)
    dice_per_frame = (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice_loss = 1.0 - dice_per_frame.mean()
    return ce, dice_loss, dice_per_frame.mean()


def combined_objective(
    logits, target_masks, clinical_ef, ef_loss_weight=1.0, n_discs=20
):
    batch_size, _, height, width = logits.shape
    ce, dice_loss, dice_score = segmentation_losses(logits, target_masks)
    logits_pair = logits.float().view(batch_size, 2, 2, height, width)
    foreground = torch.softmax(logits_pair, dim=2)[:, :, 1]
    edv_soft = soft_simpson_volume(foreground[:, 0], n_discs=n_discs)
    esv_soft = soft_simpson_volume(foreground[:, 1], n_discs=n_discs)
    ef_soft = 100.0 * (edv_soft - esv_soft) / (edv_soft + 1e-6)
    ef_mae_pct = torch.mean(torch.abs(ef_soft - clinical_ef.float()))
    ef_mae_fraction = ef_mae_pct / 100.0
    total = ce + dice_loss + ef_loss_weight * ef_mae_fraction
    return {
        "total": total,
        "ce": ce,
        "dice_loss": dice_loss,
        "dice": dice_score,
        "ef_mae_pct": ef_mae_pct,
        "ef_pred_soft": ef_soft,
        "edv_soft": edv_soft,
        "esv_soft": esv_soft,
    }
