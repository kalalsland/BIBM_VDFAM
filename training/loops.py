"""Per-epoch training and evaluation loops.

These functions are deliberately stateless (no Trainer class) so they can
be reused directly in the Optuna objective and in the final training run.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# -------------------------------------------------------------- train loop --


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    cfg: dict,
    scheduler=None,
    scheduler_batch_step: bool = False,
) -> Tuple[float, float, float, float]:
    """Train the model for one epoch.

    Args:
        model:                 The multimodal model.
        train_loader:          Training DataLoader.
        criterion:             Classification loss (e.g. CrossEntropyLoss).
        optimizer:             Parameter optimizer.
        device:                Compute device.
        epoch:                 Current epoch index (0-based, used for CL weight decay).
        cfg:                   Config dict.
        scheduler:             LR scheduler (or ``None``).
        scheduler_batch_step:  If ``True``, ``scheduler.step()`` is called each batch.

    Returns:
        ``(total_loss, cls_loss, cl_loss, align_loss)`` – all epoch-averaged.
    """
    model.train()
    total_loss = total_cls = total_cl = total_align = 0.0

    dynamic_cl_weight = cfg.get("dynamic_cl_weight", True)
    lambda_cl = cfg["lambda_cl"] * (0.95 ** epoch) if dynamic_cl_weight else cfg["lambda_cl"]
    lambda_align = cfg["lambda_align"]

    for tabular, image, text, labels in train_loader:
        tabular = tabular.to(device)
        image = image.to(device)
        text = text.to(device)
        labels = labels.to(device)

        # MixUp augmentation
        if cfg["use_mixup"] and random.random() > 0.5:
            alpha = cfg["mixup_alpha"]
            lam = float(np.random.beta(alpha, alpha))
            idx = torch.randperm(tabular.size(0))
            tabular_m = lam * tabular + (1 - lam) * tabular[idx]
            image_m = lam * image + (1 - lam) * image[idx]
            text_m = lam * text + (1 - lam) * text[idx]
            labels_a, labels_b = labels, labels[idx]
            outputs, cl_loss, align_loss = model(
                tabular_m, image_m, text_m,
                compute_contrastive=False, compute_alignment=False,
            )
            cls_loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            loss = cls_loss
            cl_loss = torch.tensor(0.0, device=device)
            align_loss = torch.tensor(0.0, device=device)
        else:
            outputs, cl_loss, align_loss = model(tabular, image, text)
            cls_loss = criterion(outputs, labels)
            loss = cls_loss + lambda_cl * cl_loss + lambda_align * align_loss

        optimizer.zero_grad()
        loss.backward()
        if cfg["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        if scheduler is not None and scheduler_batch_step:
            scheduler.step()

        n = len(labels)
        total_loss += loss.item() * n
        total_cls += cls_loss.item() * n
        total_cl += cl_loss.item() * n
        total_align += align_loss.item() * n

    N = len(train_loader.dataset)
    return total_loss / N, total_cls / N, total_cl / N, total_align / N


# ------------------------------------------------------------- eval loop ---


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], List[int], List[float]]:
    """Evaluate the model on a given dataloader.

    Args:
        model:      The multimodal model (eval mode set internally).
        dataloader: DataLoader to evaluate on.
        criterion:  Loss function.
        device:     Compute device.
        threshold:  Decision threshold for converting probabilities to labels.

    Returns:
        ``(metrics_dict, all_labels, all_probs)``

        metrics_dict keys: ``loss``, ``auc``, ``accuracy``, ``f1``,
        ``precision``, ``recall``, ``specificity``.
    """
    model.eval()
    total_loss = 0.0
    all_labels: List[int] = []
    all_probs: List[float] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for tabular, image, text, labels in dataloader:
            tabular = tabular.to(device)
            image = image.to(device)
            text = text.to(device)
            labels = labels.to(device)

            outputs, _, _ = model(tabular, image, text, compute_contrastive=False, compute_alignment=False)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)

            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= threshold).astype(int)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            all_preds.extend(preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(all_labels, all_preds)
    if cm.size == 1:
        specificity = 1.0 if all_labels[0] == 0 else 0.0
    else:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics: Dict[str, float] = {
        "loss":        total_loss / len(dataloader.dataset),
        "auc":         auc,
        "accuracy":    accuracy_score(all_labels, all_preds),
        "f1":          f1_score(all_labels, all_preds, zero_division=0),
        "precision":   precision_score(all_labels, all_preds, zero_division=0),
        "recall":      recall_score(all_labels, all_preds, zero_division=0),
        "specificity": specificity,
    }
    return metrics, list(all_labels), list(all_probs)


# ------------------------------------------------- threshold tuning helper --


def find_optimal_threshold(labels: List[int], probs: List[float]) -> float:
    """Sweep thresholds in [0.3, 0.7] and pick the one maximising F1 on val set."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.3, 0.7, 50):
        preds = (np.array(probs) >= t).astype(int)
        f = f1_score(labels, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = t
    print(f"Optimal threshold: {best_t:.4f}  F1: {best_f1:.4f}")
    return float(best_t)
