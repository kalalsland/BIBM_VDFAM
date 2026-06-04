"""High-level training pipeline: Optuna HPO + final multi-seed evaluation.

Contains:
  - objective_model_params  – Optuna trial objective (CV or single split)
  - train_and_plot_final_model – Train a final model and save training curves
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..data.dataset import MultimodalDataset
from ..data.augmentation import apply_smote
from ..models.ours_fusion import OursFusion
from .scheduler import RAdam, build_scheduler
from .loops import evaluate, train_one_epoch


def _init_weights(m: nn.Module) -> None:
    """Xavier-uniform init for Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def _make_loader(dataset: MultimodalDataset, cfg: dict, *, shuffle: bool = True, sampler=None):
    if sampler is not None:
        return DataLoader(dataset, batch_size=cfg["batch_size"], sampler=sampler)
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=shuffle)


# -------------------------------------------------------- Optuna objective --


def objective_model_params(
    trial: optuna.Trial,
    cfg: dict,
    X_tab: np.ndarray,
    X_img: np.ndarray,
    X_txt: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> float:
    """Optuna trial: search over lambda_cl, lambda_align (and optionally others).

    Returns mean validation AUC across folds / splits.
    """
    # -- hyper-parameter search space (narrow by default, expand as needed) --
    cfg = cfg.copy()
    cfg["lambda_cl"]    = trial.suggest_float("lambda_cl",    0.02, 0.02, log=True)
    cfg["lambda_align"] = trial.suggest_float("lambda_align", 0.20, 0.20, log=True)
    cfg["hidden_dim"]   = trial.suggest_categorical("hidden_dim",    [cfg["hidden_dim"]])
    cfg["projection_dim"] = trial.suggest_categorical("projection_dim", [cfg["projection_dim"]])
    cfg["num_heads"]    = trial.suggest_int("num_heads", cfg["num_heads"], cfg["num_heads"])
    cfg["dropout_rate"] = trial.suggest_float("dropout_rate", cfg["dropout_rate"], cfg["dropout_rate"])

    tabular_dim, image_dim, text_dim = X_tab.shape[1], X_img.shape[1], X_txt.shape[1]

    # build splits
    if cfg["use_cross_validation"]:
        skf = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=42)
        splits = list(skf.split(X_tab, y))
    else:
        tr_idx, va_idx = train_test_split(
            np.arange(len(y)), test_size=cfg["val_size"], random_state=42, stratify=y
        )
        splits = [(tr_idx, va_idx)]

    fold_aucs: List[float] = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        label = "Fold" if cfg["use_cross_validation"] else "Split"
        print(f"\n{label} {fold + 1}/{len(splits)}")

        Xt, Xi, Xte = X_tab[train_idx], X_img[train_idx], X_txt[train_idx]
        Xv_t, Xv_i, Xv_te = X_tab[val_idx], X_img[val_idx], X_txt[val_idx]
        yt, yv = y[train_idx], y[val_idx]

        if cfg["use_smote"]:
            Xt, Xi, Xte, yt, sampler = apply_smote(Xt, Xi, Xte, yt, cfg["smote_method"])
        else:
            sampler = None

        train_ds = MultimodalDataset(Xt, Xi, Xte, yt, is_train=True)
        val_ds = MultimodalDataset(
            Xv_t, Xv_i, Xv_te, yv,
            tabular_scaler=train_ds.tabular_scaler,
            image_scaler=train_ds.image_scaler,
            text_scaler=train_ds.text_scaler,
            is_train=False,
        )
        train_loader = _make_loader(train_ds, cfg, shuffle=True, sampler=sampler)
        val_loader = _make_loader(val_ds, cfg, shuffle=False)

        model = OursFusion(tabular_dim, image_dim, text_dim, cfg).to(device)
        model.apply(_init_weights)
        optimizer = RAdam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        scheduler, scheduler_batch_step = build_scheduler(optimizer, cfg, len(train_loader))
        criterion = nn.CrossEntropyLoss()

        best_val_auc = 0.0
        no_improve = 0

        for epoch in range(cfg["epochs"]):
            train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, cfg,
                            scheduler=scheduler, scheduler_batch_step=scheduler_batch_step)
            val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
            val_auc = val_metrics["auc"]

            if scheduler is not None and not scheduler_batch_step:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_auc)
                else:
                    scheduler.step()

            trial.report(val_auc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg["patience"]:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            print(f"Epoch {epoch + 1}/{cfg['epochs']} | Val AUC: {val_auc:.4f} (best {best_val_auc:.4f})")

        fold_aucs.append(best_val_auc)
        print(f"{label} {fold + 1} best AUC: {best_val_auc:.4f}")

    mean_auc = float(np.mean(fold_aucs))
    print("Mean AUC:", mean_auc)
    return mean_auc


# ------------------------------------------------ final training run -------


def train_and_plot_final_model(
    cfg: dict,
    X_tab_train: np.ndarray,
    X_img_train: np.ndarray,
    X_txt_train: np.ndarray,
    y_train: np.ndarray,
    X_tab_val: np.ndarray,
    X_img_val: np.ndarray,
    X_txt_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: str,
    device: torch.device,
) -> Tuple[nn.Module, MultimodalDataset, float]:
    """Train the final model with the best config, save training curves + best weights.

    Returns:
        ``(model, train_dataset, optimal_threshold)``
        where *train_dataset* carries the fitted scalers needed to transform
        val/test data.
    """
    if cfg["use_smote"]:
        X_tab_train, X_img_train, X_txt_train, y_train, sampler = apply_smote(
            X_tab_train, X_img_train, X_txt_train, y_train, cfg["smote_method"]
        )
    else:
        sampler = None

    train_ds = MultimodalDataset(X_tab_train, X_img_train, X_txt_train, y_train, is_train=True)
    val_ds = MultimodalDataset(
        X_tab_val, X_img_val, X_txt_val, y_val,
        tabular_scaler=train_ds.tabular_scaler,
        image_scaler=train_ds.image_scaler,
        text_scaler=train_ds.text_scaler,
        is_train=False,
    )
    train_loader = _make_loader(train_ds, cfg, shuffle=True, sampler=sampler)
    val_loader = _make_loader(val_ds, cfg, shuffle=False)

    tabular_dim = X_tab_train.shape[1]
    image_dim = X_img_train.shape[1]
    text_dim = X_txt_train.shape[1]
    model = OursFusion(tabular_dim, image_dim, text_dim, cfg).to(device)
    model.apply(_init_weights)

    optimizer = RAdam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler, scheduler_batch_step = build_scheduler(optimizer, cfg, len(train_loader))
    criterion = nn.CrossEntropyLoss()

    # history buffers
    train_losses, val_losses = [], []
    train_cls_losses, train_cl_losses, train_align_losses = [], [], []
    val_aucs, learning_rates = [], []

    best_val_auc = 0.0
    best_model_path = os.path.join(output_dir, "best_model.pth")
    best_val_labels: Optional[List] = None
    best_val_probs: Optional[List] = None

    for epoch in range(cfg["epochs"]):
        tr_loss, tr_cls, tr_cl, tr_align = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, cfg,
            scheduler=scheduler, scheduler_batch_step=scheduler_batch_step,
        )
        val_metrics, val_labels, val_probs = evaluate(model, val_loader, criterion, device)
        val_auc = val_metrics["auc"]

        if scheduler is not None and not scheduler_batch_step:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_metrics["loss"])
        train_cls_losses.append(tr_cls)
        train_cl_losses.append(tr_cl)
        train_align_losses.append(tr_align)
        val_aucs.append(val_auc)
        learning_rates.append(optimizer.param_groups[0]["lr"])

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            best_val_labels = val_labels
            best_val_probs = val_probs
            print(f"[Epoch {epoch + 1}] New best – Val AUC: {val_auc:.4f}")

        print(
            f"Epoch {epoch + 1}/{cfg['epochs']} | "
            f"Train {tr_loss:.4f} | Val {val_metrics['loss']:.4f} | "
            f"Val AUC {val_auc:.4f} (best {best_val_auc:.4f})"
        )

    # ---------------------------------------------------------------- plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(train_losses, label="Train"); axes[0, 0].plot(val_losses, label="Val")
    axes[0, 0].legend(); axes[0, 0].set_title("Loss")
    axes[0, 1].plot(train_cls_losses, label="Cls")
    axes[0, 1].plot(train_cl_losses, label="CL")
    axes[0, 1].plot(train_align_losses, label="Align")
    axes[0, 1].legend(); axes[0, 1].set_title("Loss components")
    axes[1, 0].plot(val_aucs, label="Val AUC"); axes[1, 0].set_title("Val AUC")
    axes[1, 1].plot(learning_rates); axes[1, 1].set_yscale("log"); axes[1, 1].set_title("LR")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_metrics.png")
    plt.savefig(plot_path); plt.close()
    print("Saved plot:", plot_path)

    # ---------------------------------------------------------- reload best
    model.load_state_dict(torch.load(best_model_path))

    optimal_threshold = 0.5
    if cfg.get("use_threshold_tuning", False) and best_val_labels and best_val_probs:
        from .loops import find_optimal_threshold
        optimal_threshold = find_optimal_threshold(best_val_labels, best_val_probs)

    return model, train_ds, optimal_threshold
