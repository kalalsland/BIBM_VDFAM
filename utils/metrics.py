"""Evaluation metrics for stroke prognosis model.

This module provides comprehensive metrics for binary classification tasks,
including AUC-ROC, accuracy, precision, recall, F1-score, and confusion matrix.
"""

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)


def compute_metrics(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics for binary classification.
    
    This function calculates multiple performance metrics including:
    - AUC-ROC: Area under the receiver operating characteristic curve
    - Accuracy: Overall classification accuracy
    - Precision: Positive predictive value
    - Recall: Sensitivity or true positive rate
    - F1: Harmonic mean of precision and recall
    - Specificity: True negative rate
    
    Args:
        outputs: Model predictions (logits or probabilities), shape (N,) or (N, 1)
        labels: Ground truth labels (0 or 1), shape (N,) or (N, 1)
        threshold: Decision threshold for binary classification (default: 0.5)
    
    Returns:
        Dictionary containing computed metrics:
            - 'auc': Area under ROC curve [0, 1]
            - 'accuracy': Classification accuracy [0, 1]
            - 'precision': Precision score [0, 1]
            - 'recall': Recall/sensitivity score [0, 1]
            - 'f1': F1 score [0, 1]
            - 'specificity': Specificity score [0, 1]
    
    Example:
        >>> outputs = torch.tensor([0.8, 0.3, 0.6, 0.9])
        >>> labels = torch.tensor([1, 0, 1, 1])
        >>> metrics = compute_metrics(outputs, labels)
        >>> print(f"AUC: {metrics['auc']:.4f}")
        AUC: 1.0000
    
    Note:
        - If outputs are logits, they will be converted to probabilities using sigmoid
        - Handles both 1D and 2D tensor inputs
        - Gracefully handles edge cases (e.g., all same class)
    """
    # Convert to numpy arrays
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Reshape if necessary
    if outputs.ndim == 2 and outputs.shape[1] == 1:
        outputs = outputs.squeeze(1)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    
    # Convert logits to probabilities if needed
    # Detect if outputs are logits (values outside [0, 1])
    if np.any(outputs < 0) or np.any(outputs > 1):
        outputs = 1 / (1 + np.exp(-outputs))  # Sigmoid activation
    
    # Convert probabilities to binary predictions
    predictions = (outputs >= threshold).astype(int)
    
    # Compute metrics
    metrics = {}
    
    try:
        # AUC-ROC (requires probabilities)
        if len(np.unique(labels)) > 1:
            metrics['auc'] = roc_auc_score(labels, outputs)
        else:
            metrics['auc'] = 0.0  # Undefined for single class
    except ValueError:
        metrics['auc'] = 0.0
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(labels, predictions)
    metrics['precision'] = precision_score(labels, predictions, zero_division=0)
    metrics['recall'] = recall_score(labels, predictions, zero_division=0)
    metrics['f1'] = f1_score(labels, predictions, zero_division=0)
    
    # Specificity (True Negative Rate)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return metrics


def compute_confusion_matrix(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Compute confusion matrix and related counts.
    
    Args:
        outputs: Model predictions (logits or probabilities)
        labels: Ground truth labels
        threshold: Decision threshold for binary classification
    
    Returns:
        Tuple containing:
            - confusion_matrix: 2x2 numpy array [[TN, FP], [FN, TP]]
            - counts: Dictionary with keys 'tn', 'fp', 'fn', 'tp'
    
    Example:
        >>> outputs = torch.tensor([0.8, 0.3, 0.6, 0.2])
        >>> labels = torch.tensor([1, 0, 1, 0])
        >>> cm, counts = compute_confusion_matrix(outputs, labels)
        >>> print(f"True Positives: {counts['tp']}")
        True Positives: 2
    """
    # Convert to numpy
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Reshape if necessary
    if outputs.ndim == 2:
        outputs = outputs.squeeze(1)
    if labels.ndim == 2:
        labels = labels.squeeze(1)
    
    # Convert logits to probabilities if needed
    if np.any(outputs < 0) or np.any(outputs > 1):
        outputs = 1 / (1 + np.exp(-outputs))
    
    # Binary predictions
    predictions = (outputs >= threshold).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    
    # Extract counts
    tn, fp, fn, tp = cm.ravel()
    counts = {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }
    
    return cm, counts


def compute_roc_curve(
    outputs: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve coordinates.
    
    Args:
        outputs: Model predictions (logits or probabilities)
        labels: Ground truth labels
    
    Returns:
        Tuple containing:
            - fpr: False positive rates
            - tpr: True positive rates
            - thresholds: Decision thresholds
    
    Example:
        >>> outputs = torch.randn(100)
        >>> labels = torch.randint(0, 2, (100,))
        >>> fpr, tpr, thresholds = compute_roc_curve(outputs, labels)
        >>> # Can be used for plotting ROC curve
    """
    # Convert to numpy
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Reshape if necessary
    if outputs.ndim == 2:
        outputs = outputs.squeeze(1)
    if labels.ndim == 2:
        labels = labels.squeeze(1)
    
    # Convert logits to probabilities if needed
    if np.any(outputs < 0) or np.any(outputs > 1):
        outputs = 1 / (1 + np.exp(-outputs))
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, outputs)
    
    return fpr, tpr, thresholds


def compute_optimal_threshold(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    method: str = 'youden',
) -> float:
    """Find optimal classification threshold.
    
    Args:
        outputs: Model predictions (logits or probabilities)
        labels: Ground truth labels
        method: Method for finding optimal threshold:
            - 'youden': Maximize Youden's J statistic (TPR - FPR)
            - 'f1': Maximize F1 score
            - 'balanced': Balance sensitivity and specificity
    
    Returns:
        Optimal threshold value
    
    Example:
        >>> outputs = torch.randn(100)
        >>> labels = torch.randint(0, 2, (100,))
        >>> optimal_thresh = compute_optimal_threshold(outputs, labels)
        >>> print(f"Optimal threshold: {optimal_thresh:.4f}")
    
    Mathematical Formulation:
        Youden's J statistic:
            J = sensitivity + specificity - 1
            J = TPR - FPR
        
        Optimal threshold maximizes J across all possible thresholds.
    """
    # Get ROC curve
    fpr, tpr, thresholds = compute_roc_curve(outputs, labels)
    
    if method == 'youden':
        # Youden's J statistic: J = TPR - FPR
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
    
    elif method == 'f1':
        # Find threshold that maximizes F1 score
        outputs_np = outputs.detach().cpu().numpy() if isinstance(outputs, torch.Tensor) else outputs
        labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels
        
        # Convert logits to probabilities if needed
        if np.any(outputs_np < 0) or np.any(outputs_np > 1):
            outputs_np = 1 / (1 + np.exp(-outputs_np))
        
        best_f1 = 0
        optimal_threshold = 0.5
        
        for thresh in np.linspace(0, 1, 100):
            preds = (outputs_np >= thresh).astype(int)
            f1 = f1_score(labels_np, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                optimal_threshold = thresh
    
    elif method == 'balanced':
        # Find threshold closest to equal sensitivity and specificity
        diff = np.abs(tpr - (1 - fpr))
        optimal_idx = np.argmin(diff)
        optimal_threshold = thresholds[optimal_idx]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(optimal_threshold)


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        prefix: Optional prefix for output (e.g., "Test", "Validation")
    
    Example:
        >>> metrics = {'auc': 0.85, 'accuracy': 0.80, 'f1': 0.78}
        >>> print_metrics(metrics, prefix="Test")
        Test Metrics:
          AUC: 0.8500
          Accuracy: 0.8000
          F1: 0.7800
    """
    if prefix:
        print(f"{prefix} Metrics:")
    else:
        print("Metrics:")
    
    for name, value in metrics.items():
        print(f"  {name.capitalize()}: {value:.4f}")
