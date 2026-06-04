"""Statistical significance tests for model comparison.

This module provides:
1. DeLong's test for comparing AUC between two models
2. Bootstrap confidence intervals for performance metrics
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score


def delong_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
) -> Tuple[float, float]:
    """Perform DeLong's test to compare AUC of two models.

    DeLong's test is specifically designed for comparing ROC curves and is
    more appropriate than t-test for AUC comparison.

    Args:
        y_true: Ground truth labels (N,)
        y_pred_1: Predicted probabilities from model 1 (N,)
        y_pred_2: Predicted probabilities from model 2 (N,)

    Returns:
        Tuple of (z_statistic, p_value)
        - z_statistic: The test statistic
        - p_value: Two-tailed p-value (p < 0.05 indicates significant difference)

    Reference:
        DeLong et al. (1988). Comparing the areas under two or more correlated
        receiver operating characteristic curves: a nonparametric approach.
        Biometrics, 44(3), 837-845.
    """
    # Compute AUCs
    auc_1 = roc_auc_score(y_true, y_pred_1)
    auc_2 = roc_auc_score(y_true, y_pred_2)

    # Separate positive and negative samples
    pos_idx = y_true == 1
    neg_idx = y_true == 0

    y_pred_1_pos = y_pred_1[pos_idx]
    y_pred_1_neg = y_pred_1[neg_idx]
    y_pred_2_pos = y_pred_2[pos_idx]
    y_pred_2_neg = y_pred_2[neg_idx]

    n_pos = np.sum(pos_idx)
    n_neg = np.sum(neg_idx)

    # Compute structural components (V10, V01)
    # V10: variance component for positive samples
    V10_1 = _compute_v10(y_pred_1_pos, y_pred_1_neg)
    V10_2 = _compute_v10(y_pred_2_pos, y_pred_2_neg)

    # V01: variance component for negative samples
    V01_1 = _compute_v01(y_pred_1_pos, y_pred_1_neg)
    V01_2 = _compute_v01(y_pred_2_pos, y_pred_2_neg)

    # Covariance terms
    cov_10 = _compute_covariance_v10(y_pred_1_pos, y_pred_1_neg, y_pred_2_pos, y_pred_2_neg)
    cov_01 = _compute_covariance_v01(y_pred_1_pos, y_pred_1_neg, y_pred_2_pos, y_pred_2_neg)

    # Variance of AUC difference
    var_auc_diff = (V10_1 / n_pos + V01_1 / n_neg +
                    V10_2 / n_pos + V01_2 / n_neg -
                    2 * cov_10 / n_pos - 2 * cov_01 / n_neg)

    # Z-statistic
    z_stat = (auc_1 - auc_2) / np.sqrt(var_auc_diff)

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    return float(z_stat), float(p_value)


def _compute_v10(y_pos: np.ndarray, y_neg: np.ndarray) -> float:
    """Compute V10 component for DeLong's test."""
    n_pos = len(y_pos)
    n_neg = len(y_neg)

    # For each positive sample, count how many negative samples it's ranked above
    placements = np.zeros(n_pos)
    for i, pos_score in enumerate(y_pos):
        placements[i] = np.mean(pos_score > y_neg) + 0.5 * np.mean(pos_score == y_neg)

    return np.var(placements, ddof=1) if n_pos > 1 else 0.0


def _compute_v01(y_pos: np.ndarray, y_neg: np.ndarray) -> float:
    """Compute V01 component for DeLong's test."""
    n_pos = len(y_pos)
    n_neg = len(y_neg)

    # For each negative sample, count how many positive samples rank above it
    placements = np.zeros(n_neg)
    for i, neg_score in enumerate(y_neg):
        placements[i] = np.mean(y_pos > neg_score) + 0.5 * np.mean(y_pos == neg_score)

    return np.var(placements, ddof=1) if n_neg > 1 else 0.0


def _compute_covariance_v10(
    y_pred_1_pos: np.ndarray,
    y_pred_1_neg: np.ndarray,
    y_pred_2_pos: np.ndarray,
    y_pred_2_neg: np.ndarray,
) -> float:
    """Compute covariance term for V10."""
    n_pos = len(y_pred_1_pos)

    placements_1 = np.zeros(n_pos)
    placements_2 = np.zeros(n_pos)

    for i in range(n_pos):
        placements_1[i] = (np.mean(y_pred_1_pos[i] > y_pred_1_neg) +
                          0.5 * np.mean(y_pred_1_pos[i] == y_pred_1_neg))
        placements_2[i] = (np.mean(y_pred_2_pos[i] > y_pred_2_neg) +
                          0.5 * np.mean(y_pred_2_pos[i] == y_pred_2_neg))

    if n_pos > 1:
        return np.cov(placements_1, placements_2)[0, 1]
    return 0.0


def _compute_covariance_v01(
    y_pred_1_pos: np.ndarray,
    y_pred_1_neg: np.ndarray,
    y_pred_2_pos: np.ndarray,
    y_pred_2_neg: np.ndarray,
) -> float:
    """Compute covariance term for V01."""
    n_neg = len(y_pred_1_neg)

    placements_1 = np.zeros(n_neg)
    placements_2 = np.zeros(n_neg)

    for i in range(n_neg):
        placements_1[i] = (np.mean(y_pred_1_pos > y_pred_1_neg[i]) +
                          0.5 * np.mean(y_pred_1_pos == y_pred_1_neg[i]))
        placements_2[i] = (np.mean(y_pred_2_pos > y_pred_2_neg[i]) +
                          0.5 * np.mean(y_pred_2_pos == y_pred_2_neg[i]))

    if n_neg > 1:
        return np.cov(placements_1, placements_2)[0, 1]
    return 0.0


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str = "auc",
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    threshold: float = 0.5,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted probabilities (N,)
        metric_name: Metric to compute ("auc", "accuracy", "f1", "precision", "recall")
        n_bootstraps: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        threshold: Decision threshold for classification metrics
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0.2, 0.8, 0.9, 0.3, 0.7])
        >>> auc, lower, upper = bootstrap_ci(y_true, y_pred, "auc")
        >>> print(f"AUC: {auc:.3f} [{lower:.3f}, {upper:.3f}]")
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)

    # Select metric function
    if metric_name == "auc":
        metric_fn = lambda yt, yp: roc_auc_score(yt, yp)
    elif metric_name == "accuracy":
        metric_fn = lambda yt, yp: accuracy_score(yt, (yp >= threshold).astype(int))
    elif metric_name == "f1":
        metric_fn = lambda yt, yp: f1_score(yt, (yp >= threshold).astype(int), zero_division=0)
    elif metric_name == "precision":
        metric_fn = lambda yt, yp: precision_score(yt, (yp >= threshold).astype(int), zero_division=0)
    elif metric_name == "recall":
        metric_fn = lambda yt, yp: recall_score(yt, (yp >= threshold).astype(int), zero_division=0)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    # Compute point estimate
    point_estimate = metric_fn(y_true, y_pred)

    # Bootstrap sampling
    bootstrap_scores = []
    for _ in range(n_bootstraps):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Skip if bootstrap sample has only one class
        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            score = metric_fn(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        except Exception:
            continue

    bootstrap_scores = np.array(bootstrap_scores)

    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    lower_bound = np.percentile(bootstrap_scores, lower_percentile)
    upper_bound = np.percentile(bootstrap_scores, upper_percentile)

    return float(point_estimate), float(lower_bound), float(upper_bound)


def bootstrap_ci_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    threshold: float = 0.5,
    random_state: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """Compute bootstrap CIs for all common metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        n_bootstraps: Number of bootstrap samples
        confidence_level: Confidence level
        threshold: Decision threshold
        random_state: Random seed

    Returns:
        Dictionary mapping metric names to (point_estimate, lower, upper) tuples
    """
    metrics = ["auc", "accuracy", "f1", "precision", "recall"]
    results = {}

    for metric in metrics:
        results[metric] = bootstrap_ci(
            y_true, y_pred, metric, n_bootstraps, confidence_level, threshold, random_state
        )

    return results


def format_ci_result(point: float, lower: float, upper: float, decimals: int = 4) -> str:
    """Format confidence interval result as string.

    Args:
        point: Point estimate
        lower: Lower bound
        upper: Upper bound
        decimals: Number of decimal places

    Returns:
        Formatted string like "0.8500 [0.8200, 0.8800]"
    """
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(point)} [{fmt.format(lower)}, {fmt.format(upper)}]"
