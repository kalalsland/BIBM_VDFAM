"""Post-hoc calibration methods for improving probability calibration.

This module provides:
1. Temperature Scaling - learns a single temperature parameter
2. Platt Scaling - fits a logistic regression on validation set
3. Isotonic Regression - non-parametric calibration
"""

from typing import Tuple, Optional

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class TemperatureScaling:
    """Temperature scaling calibration.

    Learns a single scalar temperature T that scales logits:
        calibrated_prob = softmax(logits / T)

    Reference:
        Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> 'TemperatureScaling':
        """Fit temperature on validation set.

        Args:
            logits: Raw model logits (N, 2) for binary classification
            y_true: Ground truth labels (N,)

        Returns:
            self
        """
        def nll_loss(T):
            """Negative log-likelihood loss."""
            scaled_logits = logits / T
            # Softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            # NLL
            log_probs = np.log(probs[np.arange(len(y_true)), y_true] + 1e-12)
            return -np.mean(log_probs)

        # Optimize temperature
        result = minimize(nll_loss, x0=1.0, method='L-BFGS-B', bounds=[(0.01, 10.0)])
        self.temperature = float(result.x[0])

        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits.

        Args:
            logits: Raw model logits (N, 2)

        Returns:
            Calibrated probabilities for positive class (N,)
        """
        scaled_logits = logits / self.temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs[:, 1]

    def fit_transform(self, logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(logits, y_true)
        return self.transform(logits)


class PlattScaling:
    """Platt scaling calibration.

    Fits a logistic regression: P(y=1|s) = 1 / (1 + exp(A*s + B))
    where s is the uncalibrated score.

    Reference:
        Platt (1999). Probabilistic Outputs for Support Vector Machines.
    """

    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> 'PlattScaling':
        """Fit Platt scaling on validation set.

        Args:
            scores: Uncalibrated scores (N,) - can be logits or probabilities
            y_true: Ground truth labels (N,)

        Returns:
            self
        """
        self.model.fit(scores.reshape(-1, 1), y_true)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling.

        Args:
            scores: Uncalibrated scores (N,)

        Returns:
            Calibrated probabilities (N,)
        """
        return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]

    def fit_transform(self, scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(scores, y_true)
        return self.transform(scores)


class IsotonicCalibration:
    """Isotonic regression calibration.

    Non-parametric calibration that fits a monotonic function.
    More flexible than Platt scaling but requires more data.
    """

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> 'IsotonicCalibration':
        """Fit isotonic regression on validation set.

        Args:
            scores: Uncalibrated scores (N,)
            y_true: Ground truth labels (N,)

        Returns:
            self
        """
        self.model.fit(scores, y_true)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration.

        Args:
            scores: Uncalibrated scores (N,)

        Returns:
            Calibrated probabilities (N,)
        """
        return self.model.predict(scores)

    def fit_transform(self, scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(scores, y_true)
        return self.transform(scores)


def calibrate_predictions(
    y_val: np.ndarray,
    y_pred_val: np.ndarray,
    y_pred_test: np.ndarray,
    method: str = 'temperature',
    logits_val: Optional[np.ndarray] = None,
    logits_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, object]:
    """Calibrate test predictions using validation set.

    Args:
        y_val: Validation labels (N_val,)
        y_pred_val: Validation predictions (N_val,) - probabilities
        y_pred_test: Test predictions (N_test,) - probabilities
        method: Calibration method ('temperature', 'platt', 'isotonic')
        logits_val: Validation logits (N_val, 2) - required for temperature scaling
        logits_test: Test logits (N_test, 2) - required for temperature scaling

    Returns:
        Tuple of (calibrated_test_predictions, calibrator_object)

    Example:
        >>> y_pred_calibrated, calibrator = calibrate_predictions(
        ...     y_val, y_pred_val, y_pred_test, method='platt'
        ... )
    """
    if method == 'temperature':
        if logits_val is None or logits_test is None:
            raise ValueError("Temperature scaling requires logits")
        calibrator = TemperatureScaling()
        calibrator.fit(logits_val, y_val)
        y_pred_calibrated = calibrator.transform(logits_test)

    elif method == 'platt':
        calibrator = PlattScaling()
        calibrator.fit(y_pred_val, y_val)
        y_pred_calibrated = calibrator.transform(y_pred_test)

    elif method == 'isotonic':
        calibrator = IsotonicCalibration()
        calibrator.fit(y_pred_val, y_val)
        y_pred_calibrated = calibrator.transform(y_pred_test)

    else:
        raise ValueError(f"Unknown calibration method: {method}")

    return y_pred_calibrated, calibrator


def compare_calibration_methods(
    y_val: np.ndarray,
    y_pred_val: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    logits_val: Optional[np.ndarray] = None,
    logits_test: Optional[np.ndarray] = None,
) -> dict:
    """Compare different calibration methods.

    Args:
        y_val: Validation labels
        y_pred_val: Validation predictions (probabilities)
        y_test: Test labels
        y_pred_test: Test predictions (probabilities)
        logits_val: Validation logits (for temperature scaling)
        logits_test: Test logits (for temperature scaling)

    Returns:
        Dictionary mapping method names to calibrated predictions
    """
    from .calibration import compute_calibration_metrics

    results = {
        'uncalibrated': y_pred_test
    }

    # Platt scaling
    y_pred_platt, _ = calibrate_predictions(
        y_val, y_pred_val, y_pred_test, method='platt'
    )
    results['platt'] = y_pred_platt

    # Isotonic regression
    y_pred_isotonic, _ = calibrate_predictions(
        y_val, y_pred_val, y_pred_test, method='isotonic'
    )
    results['isotonic'] = y_pred_isotonic

    # Temperature scaling (if logits available)
    if logits_val is not None and logits_test is not None:
        y_pred_temp, _ = calibrate_predictions(
            y_val, y_pred_val, y_pred_test, method='temperature',
            logits_val=logits_val, logits_test=logits_test
        )
        results['temperature'] = y_pred_temp

    # Compute metrics for each
    metrics_comparison = {}
    for method, y_pred in results.items():
        metrics = compute_calibration_metrics(y_test, y_pred, n_bins=10)
        metrics_comparison[method] = {
            'brier_score': metrics['brier_score'],
            'ece': metrics['ece'],
        }

    return results, metrics_comparison
