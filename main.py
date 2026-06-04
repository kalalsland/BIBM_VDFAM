"""Entry point for stroke multimodal prognosis training.

Usage — run directly from the project directory:

    cd stroke_multimodal_prognosis
    python main.py

The script:
  1. Loads pre-encoded features (tabular, image, text) from Excel files.
  2. Runs Optuna HPO to find the best hyper-parameters.
  3. Trains a final model for each of N random seeds and evaluates on test sets.
  4. Saves per-seed results + an aggregate summary JSON.
"""

import json
import os
import sys

# Allow running as `python main.py` directly from the project directory.
# Insert the *parent* of this file into sys.path so that relative imports
# (from .config, from .data, …) resolve correctly in both execution modes.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "stroke_multimodal_prognosis"  # noqa: F841

import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .config import CONFIG
from .data.loader import load_data
from .data.dataset import MultimodalDataset
from .training.loops import evaluate
from .training.pipeline import objective_model_params, train_and_plot_final_model
from .utils.helpers import set_seed


def main() -> None:
    cfg = CONFIG.copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    print("Config:", json.dumps(cfg, indent=2, default=str))

    # ------------------------------------------------------------------ data
    tabular, image, text, labels = load_data(
        cfg["clinic_path"], cfg["image_feat_path"], cfg["text_feat_path"]
    )
    print("Data shapes:", tabular.shape, image.shape, text.shape, labels.shape)

    class_dist = np.bincount(labels) / len(labels)
    print("Class distribution:", class_dist)
    if min(class_dist) < 0.3:
        cfg["use_smote"] = True
        print("Detected class imbalance; smote_method:", cfg.get("smote_method"))

    # ---------------------------------------------------------------- optuna
    study = optuna.create_study(
        direction=cfg["optuna"]["direction"],
        sampler=TPESampler(seed=cfg["initial_seed"]),
    )
    if cfg["optuna"]["pruning"]:
        study.pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10, interval_steps=1
        )

    print("\nStarting Optuna HPO …")
    study.optimize(
        lambda trial: objective_model_params(trial, cfg, tabular, image, text, labels, device),
        n_trials=cfg["optuna"]["n_trials"],
        timeout=cfg["optuna"]["timeout"],
    )
    print(
        "Optuna done.  Best trial:", study.best_trial.number,
        "value:", study.best_trial.value,
    )

    best_params = study.best_trial.params
    cfg.update(best_params)
    with open(os.path.join(cfg["output_dir"], "best_config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    # ------------------------------------------------ multi-seed evaluation
    all_test_metrics = []

    for seed_idx in range(cfg["num_random_seeds"]):
        seed = cfg["initial_seed"] + seed_idx
        print(f"\n{'='*60}\nSeed trial {seed}\n{'='*60}")
        set_seed(seed)

        # split off test set first
        (X_tab_tv, X_tab_test,
         X_img_tv, X_img_test,
         X_txt_tv, X_txt_test,
         y_tv, y_test) = train_test_split(
            tabular, image, text, labels,
            test_size=cfg["test_size"], random_state=seed, stratify=labels,
        )

        # split remaining into train + val
        (X_tab_train, X_tab_val,
         X_img_train, X_img_val,
         X_txt_train, X_txt_val,
         y_train, y_val) = train_test_split(
            X_tab_tv, X_img_tv, X_txt_tv, y_tv,
            test_size=cfg["val_size"], random_state=seed, stratify=y_tv,
        )
        print(f"Sizes – train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")

        trial_dir = os.path.join(cfg["output_dir"], f"seed_{seed}")
        os.makedirs(trial_dir, exist_ok=True)

        final_model, train_ds, optimal_threshold = train_and_plot_final_model(
            cfg,
            X_tab_train, X_img_train, X_txt_train, y_train,
            X_tab_val, X_img_val, X_txt_val, y_val,
            trial_dir, device,
        )

        # evaluate on held-out test set using scalers from the training split
        test_ds = MultimodalDataset(
            X_tab_test, X_img_test, X_txt_test, y_test,
            tabular_scaler=train_ds.tabular_scaler,
            image_scaler=train_ds.image_scaler,
            text_scaler=train_ds.text_scaler,
            is_train=False,
        )
        test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
        test_metrics, _, _ = evaluate(
            final_model, test_loader, nn.CrossEntropyLoss(), device,
            threshold=optimal_threshold,
        )

        print("Test metrics:", test_metrics)
        all_test_metrics.append(test_metrics)
        with open(os.path.join(trial_dir, "test_results.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)

    # ---------------------------------------------------------------- summary
    if all_test_metrics:
        keys = all_test_metrics[0].keys()
        summary = {}
        for k in keys:
            vals = [m[k] for m in all_test_metrics]
            summary[f"avg_{k}"] = float(np.mean(vals))
            summary[f"std_{k}"] = float(np.std(vals))
        with open(os.path.join(cfg["output_dir"], "summary_all_seeds.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("\nSummary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
