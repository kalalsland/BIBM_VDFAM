"""Run statistical significance tests, subgroup analysis, and calibration.

This script performs:
1. Bootstrap confidence intervals for all metrics (multi-seed aggregated)
2. DeLong's test comparing our model against baselines
3. Subgroup analysis by age and gender (multi-seed aggregated)
4. Calibration analysis with post-hoc calibration (Platt + Isotonic)

Usage:
    # Single seed
    python run_statistical_analysis.py --seed 39

    # All seeds aggregated (recommended for top-venue submission)
    python run_statistical_analysis.py --all_seeds

    # With baseline comparison
    python run_statistical_analysis.py --all_seeds --baseline_dir ../baseline_results
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Mirror the same sys.path trick used in main.py so bare imports resolve correctly
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import CONFIG
from data.loader import load_data
from data.dataset import MultimodalDataset
from models.ours_fusion import OursFusion
from utils.statistical_tests import (
    delong_test,
    bootstrap_ci_all_metrics,
    format_ci_result,
)
from utils.subgroup_analysis import generate_subgroup_report
from utils.calibration import generate_calibration_report, interpret_calibration, compute_calibration_metrics
from utils.post_calibration import calibrate_predictions


def load_model_and_predict(
    model_path: str,
    X_tab: np.ndarray,
    X_img: np.ndarray,
    X_txt: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    device: torch.device,
) -> tuple:
    """Load a trained model and get predictions.

    Returns:
        (y_true, y_pred_probs)
    """
    # Create dataset with proper scaling
    # Note: In real usage, you need the training set scalers
    # For now, we create a temporary dataset
    dataset = MultimodalDataset(X_tab, X_img, X_txt, y, is_train=False)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False)

    # Load model
    tabular_dim = X_tab.shape[1]
    image_dim = X_img.shape[1]
    text_dim = X_txt.shape[1]
    model = OursFusion(tabular_dim, image_dim, text_dim, cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get predictions
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for tabular, image, text, labels in loader:
            tabular = tabular.to(device)
            image = image.to(device)
            text = text.to(device)

            outputs, _, _ = model(tabular, image, text, compute_contrastive=False, compute_alignment=False)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_probs)


def run_delong_tests(
    y_true: np.ndarray,
    y_pred_ours: np.ndarray,
    baseline_preds: dict,
    output_dir: str,
) -> pd.DataFrame:
    """Run DeLong's test comparing our model against baselines.

    Args:
        y_true: Ground truth labels
        y_pred_ours: Our model's predictions
        baseline_preds: Dict mapping baseline names to their predictions
        output_dir: Directory to save results

    Returns:
        DataFrame with test results
    """
    print("\n" + "="*60)
    print("DELONG'S TEST FOR AUC COMPARISON")
    print("="*60)

    results = []

    for baseline_name, y_pred_baseline in baseline_preds.items():
        z_stat, p_value = delong_test(y_true, y_pred_ours, y_pred_baseline)

        from sklearn.metrics import roc_auc_score
        auc_ours = roc_auc_score(y_true, y_pred_ours)
        auc_baseline = roc_auc_score(y_true, y_pred_baseline)

        results.append({
            'baseline': baseline_name,
            'auc_ours': auc_ours,
            'auc_baseline': auc_baseline,
            'auc_diff': auc_ours - auc_baseline,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': 'Yes' if p_value < 0.05 else 'No',
        })

        print(f"\nOurs vs {baseline_name}:")
        print(f"  AUC (Ours):     {auc_ours:.4f}")
        print(f"  AUC (Baseline): {auc_baseline:.4f}")
        print(f"  Difference:     {auc_ours - auc_baseline:+.4f}")
        print(f"  Z-statistic:    {z_stat:.4f}")
        print(f"  P-value:        {p_value:.4f}")
        print(f"  Significant:    {'Yes (p < 0.05)' if p_value < 0.05 else 'No (p ≥ 0.05)'}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'delong_test_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    return df


def run_bootstrap_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    n_bootstraps: int = 1000,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Run bootstrap confidence interval analysis."""
    print("\n" + "="*60)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("="*60)
    print(f"Using threshold: {threshold:.4f}")

    ci_results = bootstrap_ci_all_metrics(
        y_true, y_pred, n_bootstraps=n_bootstraps,
        confidence_level=0.95, threshold=threshold
    )

    results = []
    for metric, (point, lower, upper) in ci_results.items():
        results.append({
            'metric': metric,
            'point_estimate': point,
            'ci_lower': lower,
            'ci_upper': upper,
            'ci_width': upper - lower,
        })
        print(f"{metric.upper():12s}: {format_ci_result(point, lower, upper)}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'bootstrap_ci_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Statistical analysis for stroke prognosis model')
    parser.add_argument('--seed', type=int, default=None,
                        help='Single random seed to analyze')
    parser.add_argument('--all_seeds', action='store_true',
                        help='Analyze all seeds (39, 40, 41, 42) and aggregate results')
    parser.add_argument('--baseline_dir', type=str, default=None,
                        help='Directory containing baseline model predictions')
    parser.add_argument('--n_bootstraps', type=int, default=1000,
                        help='Number of bootstrap samples (default: 1000)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')

    args = parser.parse_args()

    # Determine which seeds to analyze
    if args.all_seeds:
        seeds = [39, 40, 41, 42]
        print("Analyzing all seeds: 39, 40, 41, 42")
    elif args.seed is not None:
        seeds = [args.seed]
        print(f"Analyzing single seed: {args.seed}")
    else:
        seeds = [39]  # Default
        print("No seed specified, using default: 39")

    # Setup
    cfg = CONFIG.copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Determine output directory
    if args.output_dir is None:
        if len(seeds) == 1:
            seed_dir = os.path.join(cfg["output_dir"], f"seed_{seeds[0]}")
            output_dir = os.path.join(seed_dir, "statistical_analysis")
        else:
            output_dir = os.path.join(cfg["output_dir"], "aggregated_analysis")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load data once
    print("\nLoading data...")
    tabular, image, text, labels = load_data(
        cfg["clinic_path"], cfg["image_feat_path"], cfg["text_feat_path"]
    )

    # Collect predictions from all seeds
    all_y_true = []
    all_y_pred = []
    all_clinical_data = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Processing seed {seed}")
        print(f"{'='*60}")

        # Split data
        (X_tab_tv, X_tab_test,
         X_img_tv, X_img_test,
         X_txt_tv, X_txt_test,
         y_tv, y_test) = train_test_split(
            tabular, image, text, labels,
            test_size=cfg["test_size"], random_state=seed, stratify=labels,
        )

        # Further split train/val for calibration
        (X_tab_train, X_tab_val,
         X_img_train, X_img_val,
         X_txt_train, X_txt_val,
         y_train, y_val) = train_test_split(
            X_tab_tv, X_img_tv, X_txt_tv, y_tv,
            test_size=cfg["val_size"], random_state=seed, stratify=y_tv,
        )

        # Load model
        seed_dir = os.path.join(cfg["output_dir"], f"seed_{seed}")
        model_path = os.path.join(seed_dir, "best_model.pth")

        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}, skipping seed {seed}")
            continue

        # Get validation predictions (for calibration)
        print("Getting validation predictions...")
        y_val_true, y_val_pred = load_model_and_predict(
            model_path, X_tab_val, X_img_val, X_txt_val, y_val, cfg, device
        )

        # Get test predictions
        print("Getting test predictions...")
        y_test_true, y_test_pred = load_model_and_predict(
            model_path, X_tab_test, X_img_test, X_txt_test, y_test, cfg, device
        )

        # Apply post-hoc calibration
        print("Applying post-hoc calibration (Platt Scaling)...")
        y_val_pred_calibrated, calibrator = calibrate_predictions(
            y_val_true, y_val_pred, y_val_pred, method='platt'
        )
        y_test_pred_calibrated, _ = calibrate_predictions(
            y_val_true, y_val_pred, y_test_pred, method='platt'
        )

        # Find optimal threshold on calibrated val predictions (Youden's J)
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_val_true, y_val_pred_calibrated)
        j_scores = tpr - fpr
        optimal_threshold = float(thresholds[np.argmax(j_scores)])
        print(f"Optimal threshold (Youden's J on val): {optimal_threshold:.4f}")

        # Store calibrated predictions and optimal threshold
        all_y_true.append(y_test_true)
        all_y_pred.append(y_test_pred_calibrated)

        # Store threshold for later use (we'll use the average across seeds)
        if not hasattr(main, 'optimal_thresholds'):
            main.optimal_thresholds = []
        main.optimal_thresholds.append(optimal_threshold)

        # Get clinical data for this test set
        clinic_df = pd.read_excel(cfg["clinic_path"])
        test_indices = train_test_split(
            np.arange(len(labels)), labels,
            test_size=cfg["test_size"], random_state=seed, stratify=labels,
        )[1]
        clinic_test = clinic_df.iloc[test_indices].reset_index(drop=True)

        # Extract age and gender
        clinical_data = pd.DataFrame()
        col_names = clinic_test.columns.tolist()

        for col in col_names:
            if '年龄' in col.lower() or 'age' in col.lower():
                clinical_data['age'] = clinic_test[col]
            if '性别' in col.lower() or 'gender' in col.lower() or 'sex' in col.lower():
                clinical_data['gender'] = clinic_test[col]

        all_clinical_data.append(clinical_data)

    # Aggregate all seeds
    y_true_agg = np.concatenate(all_y_true)
    y_pred_agg = np.concatenate(all_y_pred)
    clinical_data_agg = pd.concat(all_clinical_data, ignore_index=True)

    # Use mean optimal threshold across seeds
    optimal_threshold = float(np.mean(main.optimal_thresholds)) if hasattr(main, 'optimal_thresholds') else 0.5
    print(f"\nMean optimal threshold across seeds: {optimal_threshold:.4f}")

    print(f"\n{'='*60}")
    print(f"AGGREGATED RESULTS ({len(seeds)} seeds)")
    print(f"Total test samples: {len(y_true_agg)}")
    print(f"{'='*60}")

    # ============================================================
    # 1. Bootstrap Confidence Intervals (on aggregated data)
    # ============================================================
    bootstrap_df = run_bootstrap_analysis(
        y_true_agg, y_pred_agg, output_dir,
        n_bootstraps=args.n_bootstraps, threshold=optimal_threshold
    )

    # ============================================================
    # 2. DeLong's Test (if baseline predictions available)
    # ============================================================
    if args.baseline_dir and os.path.exists(args.baseline_dir):
        print("\nLoading baseline predictions...")
        baseline_preds = {}

        for file in os.listdir(args.baseline_dir):
            if file.endswith('_predictions.npy'):
                baseline_name = file.replace('_predictions.npy', '')
                pred_path = os.path.join(args.baseline_dir, file)
                baseline_preds[baseline_name] = np.load(pred_path)
                print(f"  Loaded {baseline_name}")

        if baseline_preds:
            delong_df = run_delong_tests(y_true_agg, y_pred_agg, baseline_preds, output_dir)
        else:
            print("No baseline predictions found.")
    else:
        print("\nNo baseline directory provided. Skipping DeLong's test.")

    # ============================================================
    # 3. Subgroup Analysis (on aggregated data)
    # ============================================================
    if len(clinical_data_agg.columns) > 0:
        subgroup_dir = os.path.join(output_dir, "subgroup_analysis")
        print("\n" + "="*60)
        print("SUBGROUP ANALYSIS (AGGREGATED)")
        print("="*60)
        print(f"Using threshold: {optimal_threshold:.4f}")
        subgroup_results = generate_subgroup_report(
            y_true_agg, y_pred_agg, clinical_data_agg, subgroup_dir,
            threshold=optimal_threshold
        )
    else:
        print("No clinical variables found for subgroup analysis")

    # ============================================================
    # 4. Calibration Analysis (on aggregated data)
    # ============================================================
    calibration_dir = os.path.join(output_dir, "calibration")
    calibration_metrics = generate_calibration_report(
        y_true_agg, y_pred_agg, calibration_dir, n_bins=10
    )
    print(interpret_calibration(calibration_metrics['brier_score'], calibration_metrics['ece']))

    # ============================================================
    # Generate Summary Report
    # ============================================================
    summary_path = os.path.join(output_dir, "analysis_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("STATISTICAL ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")

        f.write(f"Seeds analyzed: {seeds}\n")
        f.write(f"Total test set size: {len(y_true_agg)}\n")
        f.write(f"Positive samples: {np.sum(y_true_agg)} ({100*np.mean(y_true_agg):.1f}%)\n")
        f.write(f"Post-hoc calibration: Platt Scaling\n\n")

        f.write("-"*60 + "\n")
        f.write("BOOTSTRAP CONFIDENCE INTERVALS (95% CI)\n")
        f.write("-"*60 + "\n")
        for _, row in bootstrap_df.iterrows():
            f.write(f"{row['metric'].upper():12s}: {format_ci_result(row['point_estimate'], row['ci_lower'], row['ci_upper'])}\n")

        if args.baseline_dir and 'delong_df' in locals():
            f.write("\n" + "-"*60 + "\n")
            f.write("DELONG'S TEST RESULTS\n")
            f.write("-"*60 + "\n")
            f.write(delong_df.to_string(index=False))

        f.write("\n" + "-"*60 + "\n")
        f.write("CALIBRATION (After Platt Scaling)\n")
        f.write("-"*60 + "\n")
        f.write(f"Brier Score : {calibration_metrics['brier_score']:.4f}\n")
        f.write(f"ECE         : {calibration_metrics['ece']:.4f}\n")
        f.write(interpret_calibration(calibration_metrics['brier_score'], calibration_metrics['ece']) + "\n")

        f.write("\n\nFor detailed subgroup analysis, see subgroup_analysis/ directory\n")
        f.write("For calibration plots, see calibration/ directory\n")

    print(f"\n{'='*60}")
    print(f"Analysis complete! Summary saved to {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
