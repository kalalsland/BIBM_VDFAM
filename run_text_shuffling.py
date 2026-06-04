"""
Text-Shuffling Ablation: Full Retrain Experiment
==================================================
Run from the parent directory (ours/):

    cd D:\\Users\\Lean2023\\Desktop\\Contrast_ex\\ours
    python -m stroke_multimodal_prognosis.run_text_shuffling

Two full-train conditions (NO retraining on generic; that equals text=0 in Table 2):
  1. correct  – original patient-specific LLM text (baseline)
  2. shuffled – text permuted across patients during BOTH train and test
                permutation uses split_seed so all 4 model seeds see the
                same text permutation (isolates model randomness from text randomness)

Runs 1 fixed split (split_seed=103) × 4 model seeds, matching the protocol
of the main experiment.
"""

import json
import os
import sys

if __name__ == "__main__" and __package__ is None:
    _pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _pkg_parent)
    __package__ = "stroke_multimodal_prognosis"

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .config import CONFIG
from .data.loader import load_data
from .data.dataset import MultimodalDataset
from .data.augmentation import apply_smote
from .models.ours_fusion import OursFusion
from .training.loops import evaluate, find_optimal_threshold, train_one_epoch
from .training.pipeline import _init_weights
from .training.scheduler import RAdam, build_scheduler
from .utils.helpers import set_seed


# ─────────────────────────────────────── text manipulation helpers ────────────

def _shuffle_text_traintest(txt_train, txt_val, txt_test, split_seed):
    """
    Permute text embeddings using split_seed (NOT model seed).
    All 4 model seeds within the same split see the identical permutation,
    so differences in results are due to model training randomness, not
    different text pairings.
    """
    rng = np.random.default_rng(split_seed)
    txt_train_s = txt_train[rng.permutation(len(txt_train))]
    txt_val_s   = txt_val[rng.permutation(len(txt_val))]
    txt_test_s  = txt_test[rng.permutation(len(txt_test))]
    return txt_train_s, txt_val_s, txt_test_s


# ─────────────────────────────────────── single seed train+eval ──────────────

def _train_and_eval_one_seed(cfg, tab_train, img_train, txt_train, y_train,
                              tab_val, img_val, txt_val, y_val,
                              tab_test, img_test, txt_test, y_test,
                              device, out_dir):
    """Full train → threshold tune → test for one seed. Returns metrics dict."""
    if cfg["use_smote"]:
        tab_train, img_train, txt_train, y_train, sampler = apply_smote(
            tab_train, img_train, txt_train, y_train, cfg["smote_method"]
        )
    else:
        sampler = None

    train_ds = MultimodalDataset(tab_train, img_train, txt_train, y_train, is_train=True)
    val_ds = MultimodalDataset(
        tab_val, img_val, txt_val, y_val,
        tabular_scaler=train_ds.tabular_scaler,
        image_scaler=train_ds.image_scaler,
        text_scaler=train_ds.text_scaler,
        is_train=False,
    )
    test_ds = MultimodalDataset(
        tab_test, img_test, txt_test, y_test,
        tabular_scaler=train_ds.tabular_scaler,
        image_scaler=train_ds.image_scaler,
        text_scaler=train_ds.text_scaler,
        is_train=False,
    )

    def _loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle)

    if sampler is not None:
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler)
    else:
        train_loader = _loader(train_ds, shuffle=True)
    val_loader  = _loader(val_ds)
    test_loader = _loader(test_ds)

    model = OursFusion(
        tab_train.shape[1], img_train.shape[1], txt_train.shape[1], cfg
    ).to(device)
    model.apply(_init_weights)

    optimizer = RAdam(model.parameters(), lr=cfg["learning_rate"],
                      weight_decay=cfg["weight_decay"])
    scheduler, batch_step = build_scheduler(optimizer, cfg, len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    best_val_auc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(cfg["epochs"]):
        train_one_epoch(model, train_loader, criterion, optimizer, device,
                        epoch, cfg, scheduler=scheduler,
                        scheduler_batch_step=batch_step)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        val_auc = val_metrics["auc"]

        if scheduler is not None and not batch_step:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()

    _, val_labels, val_probs = evaluate(model, val_loader, criterion, device)
    threshold = find_optimal_threshold(val_labels, val_probs)

    test_metrics, _, _ = evaluate(model, test_loader, criterion, device,
                                   threshold=threshold)
    return {
        "auc":         round(test_metrics["auc"] * 100, 2),
        "accuracy":    round(test_metrics["accuracy"] * 100, 2),
        "f1":          round(test_metrics["f1"] * 100, 2),
        "precision":   round(test_metrics["precision"] * 100, 2),
        "recall":      round(test_metrics["recall"] * 100, 2),
        "specificity": round(test_metrics["specificity"] * 100, 2),
    }


# ─────────────────────────────────────────────────────── main ─────────────────

def run_ablation():
    cfg = CONFIG.copy()
    cfg["optuna"] = {"n_trials": 0, "timeout": 0, "pruning": False, "direction": "maximize"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tabular, image, text, labels = load_data(
        cfg["clinic_path"], cfg["image_feat_path"], cfg["text_feat_path"]
    )
    print(f"Dataset: tab={tabular.shape}, img={image.shape}, "
          f"txt={text.shape}, N={len(labels)}")

    # Fixed split: use split_seed=103 (Split 4, confirmed good results)
    SPLIT_SEED = 103
    conditions = ["shuffled"]
    all_results = {c: [] for c in conditions}
    seeds = [cfg["initial_seed"] + i for i in range(cfg["num_random_seeds"])]

    print(f"\n{'='*60}\nSplit (split_seed={SPLIT_SEED})\n{'='*60}")

    # Outer split: hold out test set
    (tab_tv, tab_test, img_tv, img_test,
     txt_tv, txt_test, y_tv, y_test) = train_test_split(
        tabular, image, text, labels,
        test_size=cfg["test_size"], random_state=SPLIT_SEED, stratify=labels,
    )
    # Inner split: train / val
    (tab_train, tab_val, img_train, img_val,
     txt_train, txt_val, y_train, y_val) = train_test_split(
        tab_tv, img_tv, txt_tv, y_tv,
        test_size=cfg["val_size"], random_state=SPLIT_SEED, stratify=y_tv,
    )

    # Pre-compute shuffled text once for this split (same permutation across all seeds)
    txt_train_sh, txt_val_sh, txt_test_sh = _shuffle_text_traintest(
        txt_train.copy(), txt_val.copy(), txt_test.copy(), SPLIT_SEED
    )

    text_splits = {
        "shuffled": (txt_train_sh, txt_val_sh, txt_test_sh),
    }

    for seed in seeds:
        print(f"\n  Seed {seed}")
        set_seed(seed)

        for cond, (tr_txt, vl_txt, te_txt) in text_splits.items():
            out_dir = os.path.join(
                cfg["output_dir"], "text_shuffle_retrain",
                f"split_seed_{SPLIT_SEED}", f"seed_{seed}", cond
            )
            os.makedirs(out_dir, exist_ok=True)

            metrics = _train_and_eval_one_seed(
                cfg,
                tab_train, img_train, tr_txt, y_train,
                tab_val,   img_val,   vl_txt, y_val,
                tab_test,  img_test,  te_txt, y_test,
                device, out_dir,
            )
            metrics["seed"]      = seed
            metrics["condition"] = cond
            all_results[cond].append(metrics)
            print(f"    [{cond:10s}]  "
                  f"AUC={metrics['auc']:.2f}  "
                  f"ACC={metrics['accuracy']:.2f}  "
                  f"F1={metrics['f1']:.2f}")

    # ── aggregate ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nAGGREGATED RESULTS  (1 split × {len(seeds)} seeds)\n{'='*60}")
    summary = {}
    metric_keys = ["auc", "accuracy", "f1", "precision", "recall", "specificity"]
    for cond in conditions:
        recs = all_results[cond]
        if not recs:
            continue
        agg = {}
        for k in metric_keys:
            vals = [r[k] for r in recs]
            agg[k] = {"mean": round(float(np.mean(vals)), 2),
                      "std":  round(float(np.std(vals)),  2)}
        summary[cond] = agg
        print(f"  [{cond:10s}]  "
              f"AUC={agg['auc']['mean']:.2f}±{agg['auc']['std']:.2f}  "
              f"ACC={agg['accuracy']['mean']:.2f}±{agg['accuracy']['std']:.2f}  "
              f"F1={agg['f1']['mean']:.2f}±{agg['f1']['std']:.2f}")

    # ── save ───────────────────────────────────────────────────────────────────
    out_root = os.path.join(cfg["output_dir"], "text_shuffle_retrain",
                            f"split_seed_{SPLIT_SEED}")
    with open(os.path.join(out_root, "per_run.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_root}/")

    # ── LaTeX rows ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nLaTeX rows (Table tab:anchor)\n{'='*60}")
    row_labels = {
        "shuffled": r"Correct MRI + shuffled LLM text + table",
    }
    for cond, label in row_labels.items():
        if cond not in summary:
            continue
        ag = summary[cond]
        bold = (cond == "correct")
        def _fmt(k, b=bold):
            m, s = ag[k]["mean"], ag[k]["std"]
            return (f"$\\mathbf{{{m:.2f}}} \\pm {s:.2f}$" if b
                    else f"${m:.2f} \\pm {s:.2f}$")
        print(f"{label} & {_fmt('auc')} & {_fmt('accuracy')} & {_fmt('f1')} \\\\")
    print(r"Image + table only (text\,=\,0, Table~\ref{tab2}) & "
          r"$79.97 \pm 1.35$ & $76.37 \pm 2.67$ & $63.21 \pm 1.76$ \\")


if __name__ == "__main__":
    run_ablation()
