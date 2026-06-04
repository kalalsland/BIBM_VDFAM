"""Calibration analysis for probabilistic predictions.

This module provides tools to assess and visualize model calibration:
1. Calibration curves (reliability diagrams)
2. Brier score
3. Expected Calibration Error (ECE)
"""

from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute calibration metrics.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted probabilities (N,)
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with:
            - brier_score: Brier score (lower is better, 0=perfect, 0.25=random)
            - ece: Expected Calibration Error
            - fraction_of_positives: Actual positive rate per bin
            - mean_predicted_value: Mean predicted probability per bin
    """
    # Brier score
    brier = brier_score_loss(y_true, y_pred)

    # Calibration curve with quantile-based binning (equal sample size per bin)
    # This ensures each bin has enough samples
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred, n_bins=n_bins, strategy='quantile'
    )

    # Expected Calibration Error (ECE)
    ece = compute_ece(y_true, y_pred, n_bins=n_bins)

    return {
        'brier_score': float(brier),
        'ece': float(ece),
        'fraction_of_positives': fraction_of_positives,
        'mean_predicted_value': mean_predicted_value,
    }


def compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the average difference between predicted probabilities
    and actual frequencies across bins.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        n_bins: Number of bins

    Returns:
        ECE value (0 = perfect calibration)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_pred[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> None:
    """Plot calibration curve (reliability diagram).

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        n_bins: Number of bins
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    # Compute calibration
    metrics = compute_calibration_metrics(y_true, y_pred, n_bins=n_bins)
    fraction_of_positives = metrics['fraction_of_positives']
    mean_predicted_value = metrics['mean_predicted_value']
    brier = metrics['brier_score']
    ece = metrics['ece']

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives, 's-',
            linewidth=2, markersize=8, label='Model', color='steelblue')

    # Plot perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')

    # Formatting
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)

    # Add metrics as text
    textstr = f'Brier Score: {brier:.4f}\nECE: {ece:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.65, 0.15, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved calibration curve to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_calibration_histogram(
    y_pred: np.ndarray,
    n_bins: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 4),
) -> None:
    """Plot histogram of predicted probabilities.

    This helps identify if the model is well-distributed across
    the probability range or concentrated at extremes.

    Args:
        y_pred: Predicted probabilities
        n_bins: Number of histogram bins
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(y_pred, bins=n_bins, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved probability histogram to {save_path}")
    else:
        plt.show()

    plt.close()


def generate_calibration_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    n_bins: int = 10,
) -> dict:
    """Generate comprehensive calibration analysis report.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        output_dir: Directory to save results
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary of calibration metrics
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("CALIBRATION ANALYSIS")
    print("="*60)

    # Compute metrics
    metrics = compute_calibration_metrics(y_true, y_pred, n_bins=n_bins)

    print(f"Brier Score: {metrics['brier_score']:.4f}")
    print(f"Expected Calibration Error (ECE): {metrics['ece']:.4f}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        'brier_score': metrics['brier_score'],
        'ece': metrics['ece'],
        'n_bins': n_bins,
    }])
    csv_path = os.path.join(output_dir, 'calibration_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")

    # Save calibration curve data
    curve_df = pd.DataFrame({
        'mean_predicted_probability': metrics['mean_predicted_value'],
        'fraction_of_positives': metrics['fraction_of_positives'],
    })
    curve_csv_path = os.path.join(output_dir, 'calibration_curve_data.csv')
    curve_df.to_csv(curve_csv_path, index=False)
    print(f"Saved calibration curve data to {curve_csv_path}")

    # Plot calibration curve
    plot_path = os.path.join(output_dir, 'calibration_curve.png')
    plot_calibration_curve(y_true, y_pred, n_bins=n_bins, save_path=plot_path)

    # Plot probability histogram
    hist_path = os.path.join(output_dir, 'probability_histogram.png')
    plot_calibration_histogram(y_pred, save_path=hist_path)

    print(f"\nCalibration analysis complete. Results saved to {output_dir}")

    return metrics


def interpret_calibration(brier_score: float, ece: float) -> str:
    """Provide interpretation of calibration metrics.

    Args:
        brier_score: Brier score value
        ece: Expected Calibration Error

    Returns:
        Interpretation string
    """
    interpretation = []

    # Brier score interpretation
    if brier_score < 0.1:
        interpretation.append("Excellent Brier score (< 0.1)")
    elif brier_score < 0.15:
        interpretation.append("Good Brier score (0.1-0.15)")
    elif brier_score < 0.25:
        interpretation.append("Acceptable Brier score (0.15-0.25)")
    else:
        interpretation.append("Poor Brier score (≥ 0.25, no better than random)")

    # ECE interpretation
    if ece < 0.05:
        interpretation.append("Excellent calibration (ECE < 0.05)")
    elif ece < 0.10:
        interpretation.append("Good calibration (ECE 0.05-0.10)")
    elif ece < 0.15:
        interpretation.append("Acceptable calibration (ECE 0.10-0.15)")
    else:
        interpretation.append("Poor calibration (ECE ≥ 0.15)")

    return " | ".join(interpretation)
