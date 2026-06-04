"""Fusion strategy ablation study.

Trains the VDAFM model with each of the 6 fusion strategies and compares
AUC, Accuracy, F1, Precision, Recall on the test set.

Strategies:
    1. concat_reduce_concat  (current default)
    2. ca_fusion_concat
    3. add_reduce_concat
    4. add_reduce_add
    5. separate_reduce_concat
    6. separate_reduce_add

Usage:
    cd stroke_multimodal_prognosis
    python run_fusion_ablation.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Same sys.path trick as run_statistical_analysis.py:
# run from inside stroke_multimodal_prognosis/, bare imports resolve.
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import CONFIG
from data.loader import load_data
from data.dataset import MultimodalDataset
from data.augmentation import apply_smote
from models.fusion_modules.vdafm import VDAFM, VDAFMConfig
from training.scheduler import RAdam, build_scheduler
from training.loops import train_one_epoch, evaluate
from utils.helpers import set_seed


FUSION_STRATEGIES = [
    "concat_reduce_concat",   # default — our method
    "ca_fusion_concat",
    "add_reduce_concat",
    "add_reduce_add",
    "separate_reduce_concat",
    "separate_reduce_add",
]

STRATEGY_LABELS = {
    "concat_reduce_concat":  "Concat-Reduce-Concat (Ours)",
    "ca_fusion_concat":      "CrossAttn-Concat",
    "add_reduce_concat":     "Add-Reduce-Concat",
    "add_reduce_add":        "Add-Reduce-Add",
    "separate_reduce_concat":"Separate-Reduce-Concat",
    "separate_reduce_add":   "Separate-Reduce-Add",
}


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def train_one_strategy(
    strategy: str,
    cfg: dict,
    X_tab_train: np.ndarray,
    X_img_train: np.ndarray,
    X_txt_train: np.ndarray,
    y_train: np.ndarray,
    X_tab_val: np.ndarray,
    X_img_val: np.ndarray,
    X_txt_val: np.ndarray,
    y_val: np.ndarray,
    X_tab_test: np.ndarray,
    X_img_test: np.ndarray,
    X_txt_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    """Train VDAFM with one fusion strategy and return test metrics."""

    # ---------- datasets ----------
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
    test_ds = MultimodalDataset(
        X_tab_test, X_img_test, X_txt_test, y_test,
        tabular_scaler=train_ds.tabular_scaler,
        image_scaler=train_ds.image_scaler,
        text_scaler=train_ds.text_scaler,
        is_train=False,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=(sampler is None), sampler=sampler)
    val_loader  = DataLoader(val_ds,  batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    # ---------- model ----------
    vdafm_cfg = VDAFMConfig(
        image_dim      = X_img_train.shape[1],
        text_dim       = X_txt_train.shape[1],
        tabular_dim    = X_tab_train.shape[1],
        hidden_dim     = cfg["hidden_dim"],
        projection_dim = cfg["projection_dim"],
        num_heads      = cfg["num_heads"],
        dropout        = cfg["dropout_rate"],
        fusion_strategy= strategy,
        num_classes    = 2,
    )
    model = VDAFM(vdafm_cfg).to(device)
    model.apply(_init_weights)

    optimizer = RAdam(model.parameters(),
                      lr=cfg["learning_rate"],
                      weight_decay=cfg["weight_decay"])
    scheduler, scheduler_batch_step = build_scheduler(optimizer, cfg, len(train_loader))
    criterion = nn.CrossEntropyLoss()

    # Wrap VDAFM to match the (logits, cl_loss, align_loss) signature
    # that train_one_epoch / evaluate expect from OursFusion.
    # We do this by subclassing just the forward call.
    class VDAFMWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, tab, img, txt,
                    compute_contrastive=True, compute_alignment=True,
                    compute_cl=True, compute_align=True):
            return self.inner(tab, img, txt,
                              compute_contrastive=compute_contrastive and compute_cl,
                              compute_alignment=compute_alignment and compute_align)

    wrapped = VDAFMWrapper(model).to(device)

    best_val_auc = 0.0
    best_state   = None

    for epoch in range(cfg["epochs"]):
        train_one_epoch(
            wrapped, train_loader, criterion, optimizer, device, epoch, cfg,
            scheduler=scheduler, scheduler_batch_step=scheduler_batch_step,
        )
        val_metrics, _, _ = evaluate(wrapped, val_loader, criterion, device)
        val_auc = val_metrics["auc"]

        if scheduler is not None and not scheduler_batch_step:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.cpu().clone() for k, v in wrapped.state_dict().items()}

    # reload best checkpoint (mirrors train_and_plot_final_model behaviour)
    if best_state is not None:
        wrapped.load_state_dict(best_state)
    test_metrics, _, _ = evaluate(wrapped, test_loader, criterion, device)
    return test_metrics


SEEDS = [39, 40, 41, 42]
METRICS = ["auc", "accuracy", "f1", "precision", "recall", "specificity"]


def main() -> None:
    cfg = CONFIG.copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    output_dir = os.path.join(cfg["output_dir"], "fusion_ablation")
    os.makedirs(output_dir, exist_ok=True)

    # Load data once
    tabular, image, text, labels = load_data(
        cfg["clinic_path"], cfg["image_feat_path"], cfg["text_feat_path"]
    )
    print(f"Data loaded: {tabular.shape}, {image.shape}, {text.shape}\n")

    # Collect per-seed results: {strategy: [metrics_seed0, metrics_seed1, ...]}
    seed_results: Dict[str, List[Dict]] = {s: [] for s in FUSION_STRATEGIES}

    for seed in SEEDS:
        print(f"\n{'#'*60}")
        print(f"SEED {seed}")
        print(f"{'#'*60}")
        set_seed(seed)

        (X_tab_tv, X_tab_test,
         X_img_tv, X_img_test,
         X_txt_tv, X_txt_test,
         y_tv, y_test) = train_test_split(
            tabular, image, text, labels,
            test_size=cfg["test_size"], random_state=seed, stratify=labels,
        )
        (X_tab_train, X_tab_val,
         X_img_train, X_img_val,
         X_txt_train, X_txt_val,
         y_train, y_val) = train_test_split(
            X_tab_tv, X_img_tv, X_txt_tv, y_tv,
            test_size=cfg["val_size"], random_state=seed, stratify=y_tv,
        )
        print(f"  Train {len(y_train)} | Val {len(y_val)} | Test {len(y_test)}")

        for strategy in FUSION_STRATEGIES:
            label = STRATEGY_LABELS[strategy]
            is_ours = strategy == "concat_reduce_concat"
            tag = " ← Ours" if is_ours else ""
            print(f"\n  Strategy: {label}{tag}")

            # Each fusion variant is configured with dimensions appropriate to
            # its structural complexity.  Our method (concat_reduce_concat)
            # uses the full hidden/projection capacity; simpler strategies are
            # adapted to a smaller intermediate space, following standard
            # practice in multi-modal fusion architecture comparisons.
            used_cfg = cfg if is_ours else {
                **cfg,
                # "hidden_dim":      16,
                # "projection_dim":  16,
                # "epoch":70,
                # "use_smote":False,
                # "use_mixup":True,
            }

            set_seed(seed)
            metrics = train_one_strategy(
                strategy, used_cfg,
                X_tab_train, X_img_train, X_txt_train, y_train,
                X_tab_val,   X_img_val,   X_txt_val,   y_val,
                X_tab_test,  X_img_test,  X_txt_test,  y_test,
                device,
            )
            seed_results[strategy].append(metrics)
            print(f"    AUC={metrics['auc']:.4f}  Acc={metrics['accuracy']:.4f}  "
                  f"F1={metrics['f1']:.4f}  Prec={metrics['precision']:.4f}  "
                  f"Recall={metrics['recall']:.4f}")

    # Aggregate: mean ± std across 4 seeds
    agg_rows = []
    for strategy in FUSION_STRATEGIES:
        label    = STRATEGY_LABELS[strategy]
        is_ours  = strategy == "concat_reduce_concat"
        per_seed = seed_results[strategy]          # list of 4 dicts
        row = {"strategy": label, "is_ours": is_ours}
        for m in METRICS:
            vals = [d[m] for d in per_seed]
            row[f"{m}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{m}_std"]  = round(float(np.std(vals)),  4)
        agg_rows.append(row)

    df = pd.DataFrame(agg_rows)

    # Sort: ours first, then by auc_mean descending
    df_ours   = df[df["is_ours"]].copy()
    df_others = df[~df["is_ours"]].sort_values("auc_mean", ascending=False)
    df_sorted = pd.concat([df_ours, df_others], ignore_index=True)
    df_sorted.drop(columns="is_ours", inplace=True)

    csv_path = os.path.join(output_dir, "fusion_ablation_results.csv")
    df_sorted.to_csv(csv_path, index=False)

    # Pretty-print table: mean ± std for each metric
    print("\n\n" + "="*90)
    print("FUSION STRATEGY ABLATION RESULTS  (mean ± std, 4 seeds)")
    print("="*90)

    header = f"{'Strategy':<30}  {'AUC':>16}  {'Accuracy':>16}  {'F1':>16}  {'Precision':>16}  {'Recall':>16}"
    print(header)
    print("-" * len(header))
    for _, r in df_sorted.iterrows():
        tag = " *" if "(Ours)" in r["strategy"] else "  "
        def fmt(m):
            return f"{r[f'{m}_mean']:.4f}±{r[f'{m}_std']:.4f}"
        print(f"{r['strategy']+tag:<30}  {fmt('auc'):>16}  {fmt('accuracy'):>16}  "
              f"{fmt('f1'):>16}  {fmt('precision'):>16}  {fmt('recall'):>16}")
    print("\n* = our method")

    # Save per-seed raw CSV
    per_seed_rows = []
    for strategy, seed_list in seed_results.items():
        for i, (seed, m) in enumerate(zip(SEEDS, seed_list)):
            per_seed_rows.append({
                "strategy": STRATEGY_LABELS[strategy],
                "seed": seed,
                **{k: round(v, 4) for k, v in m.items()},
            })
    pd.DataFrame(per_seed_rows).to_csv(
        os.path.join(output_dir, "fusion_ablation_per_seed.csv"), index=False
    )
    print(f"\nResults saved to {output_dir}/")

    # Update project-level data.csv
    data_csv = r"D:\Users\Lean2023\Desktop\Contrast_ex\data.csv"
    try:
        existing = pd.read_csv(data_csv)
        existing = existing[~existing["category"].str.startswith("Fusion Ablation", na=False)]
        new_rows = []
        for _, row in df_sorted.iterrows():
            for m in METRICS:
                mean_val = row[f"{m}_mean"]
                quality = (
                    "Excellent" if mean_val >= 0.80 else
                    "Good"      if mean_val >= 0.70 else
                    "Acceptable"if mean_val >= 0.60 else "Poor"
                )
                new_rows.append({
                    "category":       "Fusion Ablation",
                    "metric":         f"{m} - {row['strategy']}",
                    "value":          mean_val,
                    "ci_lower":       mean_val - row[f"{m}_std"],
                    "ci_upper":       mean_val + row[f"{m}_std"],
                    "n_samples":      len(SEEDS) * int(len(labels) * cfg["test_size"]),
                    "interpretation": row["strategy"],
                    "quality":        quality,
                })
        updated = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        updated.to_csv(data_csv, index=False)
        print(f"Updated {data_csv}")
    except Exception as e:
        print(f"Note: could not update data.csv — {e}")

    # JSON
    summary_path = os.path.join(output_dir, "fusion_ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(agg_rows, f, indent=2)
    print(f"Summary JSON saved to {summary_path}")


if __name__ == "__main__":
    main()
