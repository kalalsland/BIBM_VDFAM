"""Subgroup analysis for evaluating model performance across patient subgroups.

This module provides tools to:
1. Stratify patients by clinical characteristics (age, gender, etc.)
2. Compute metrics for each subgroup
3. Visualize subgroup performance
"""

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score


def stratify_by_age(
    ages: np.ndarray,
    bins: Optional[List[int]] = None,
    labels: Optional[List[str]] = None,
) -> np.ndarray:
    """Stratify patients into age groups.

    Args:
        ages: Array of patient ages
        bins: Age bin edges (default: [0, 70, 120] — simple binary split)
        labels: Labels for age groups (default: ["<70", "≥70"])

    Returns:
        Array of group labels for each patient
    """
    if bins is None:
        bins = [0, 70, 120]
    if labels is None:
        labels = ["<70", "≥70"]

    return pd.cut(ages, bins=bins, labels=labels, right=False)


def stratify_by_gender(
    genders: np.ndarray,
    male_value: int = 0,
) -> np.ndarray:
    """Stratify patients by gender.

    Args:
        genders: Array of gender values
        male_value: Value representing male (default: 0, female is 1)

    Returns:
        Array of gender labels ("Male" or "Female")
    """
    return np.where(genders == male_value, "Male", "Female")


def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_labels: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute performance metrics for each subgroup.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted probabilities (N,)
        subgroup_labels: Subgroup assignment for each sample (N,)
        threshold: Decision threshold for classification

    Returns:
        DataFrame with columns: subgroup, n_samples, auc, accuracy, f1, precision, recall

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1, 0])
        >>> y_pred = np.array([0.2, 0.8, 0.9, 0.3, 0.7, 0.1])
        >>> groups = np.array(["A", "A", "B", "B", "B", "A"])
        >>> df = compute_subgroup_metrics(y_true, y_pred, groups)
        >>> print(df)
    """
    # Convert to array and handle pandas Categorical
    if isinstance(subgroup_labels, pd.Categorical):
        subgroup_labels = subgroup_labels.astype(str)
    elif hasattr(subgroup_labels, 'values'):
        subgroup_labels = subgroup_labels.values

    # Remove NaN values
    valid_mask = pd.notna(subgroup_labels)
    subgroup_labels = subgroup_labels[valid_mask]
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    unique_groups = np.unique(subgroup_labels)
    results = []

    for group in unique_groups:
        mask = subgroup_labels == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        n_samples = len(y_true_group)

        # Skip if too few samples or only one class
        if n_samples < 5 or len(np.unique(y_true_group)) < 2:
            results.append({
                "subgroup": str(group),
                "n_samples": n_samples,
                "n_positive": int(np.sum(y_true_group)),
                "auc": np.nan,
                "accuracy": np.nan,
                "f1": np.nan,
                "precision": np.nan,
                "recall": np.nan,
            })
            continue

        # Compute metrics
        y_pred_binary = (y_pred_group >= threshold).astype(int)

        try:
            auc = roc_auc_score(y_true_group, y_pred_group)
        except Exception:
            auc = np.nan

        results.append({
            "subgroup": str(group),
            "n_samples": n_samples,
            "n_positive": int(np.sum(y_true_group)),
            "auc": auc,
            "accuracy": accuracy_score(y_true_group, y_pred_binary),
            "f1": f1_score(y_true_group, y_pred_binary, zero_division=0),
            "precision": precision_score(y_true_group, y_pred_binary, zero_division=0),
            "recall": recall_score(y_true_group, y_pred_binary, zero_division=0),
        })

    return pd.DataFrame(results)


def analyze_by_age(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ages: np.ndarray,
    bins: Optional[List[int]] = None,
    labels: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Perform subgroup analysis stratified by age.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        ages: Patient ages
        bins: Age bin edges
        labels: Age group labels
        threshold: Decision threshold

    Returns:
        DataFrame with metrics for each age group
    """
    age_groups = stratify_by_age(ages, bins, labels)
    return compute_subgroup_metrics(y_true, y_pred, age_groups, threshold)


def analyze_by_gender(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    genders: np.ndarray,
    male_value: int = 0,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Perform subgroup analysis stratified by gender.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        genders: Gender values
        male_value: Value representing male
        threshold: Decision threshold

    Returns:
        DataFrame with metrics for each gender
    """
    gender_labels = stratify_by_gender(genders, male_value)
    return compute_subgroup_metrics(y_true, y_pred, gender_labels, threshold)


def analyze_by_severity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    severity_scores: np.ndarray,
    bins: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Perform subgroup analysis stratified by severity score.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        severity_scores: Clinical severity scores (e.g., NIHSS)
        bins: Severity bin edges
        labels: Severity group labels
        threshold: Decision threshold

    Returns:
        DataFrame with metrics for each severity group
    """
    if bins is None:
        # Default bins for NIHSS-like scores
        bins = [0, 5, 15, 25, 50]
    if labels is None:
        labels = ["Mild", "Moderate", "Severe", "Very Severe"]

    severity_groups = pd.cut(severity_scores, bins=bins, labels=labels, right=False)
    return compute_subgroup_metrics(y_true, y_pred, severity_groups, threshold)


def plot_subgroup_comparison(
    subgroup_df: pd.DataFrame,
    metric: str = "auc",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """Plot subgroup performance comparison.

    Args:
        subgroup_df: DataFrame from compute_subgroup_metrics
        metric: Metric to plot ("auc", "accuracy", "f1", etc.)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (if None, displays instead)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Filter out NaN values
    plot_df = subgroup_df[~subgroup_df[metric].isna()].copy()

    if len(plot_df) == 0:
        print(f"No valid data to plot for metric: {metric}")
        return

    # Create bar plot
    x_pos = np.arange(len(plot_df))
    bars = ax.bar(x_pos, plot_df[metric], alpha=0.7, color='steelblue')

    # Add value labels on bars
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax.text(i, row[metric] + 0.01, f"{row[metric]:.3f}",
                ha='center', va='bottom', fontsize=10)
        # Add sample size below
        ax.text(i, -0.05, f"n={row['n_samples']}",
                ha='center', va='top', fontsize=9, color='gray')

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_df['subgroup'], rotation=0)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_xlabel('Subgroup', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'{metric.upper()} by Subgroup', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_subgroup_heatmap(
    subgroup_dfs: Dict[str, pd.DataFrame],
    metrics: List[str] = ["auc", "accuracy", "f1"],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> None:
    """Plot heatmap comparing multiple subgroup analyses.

    Args:
        subgroup_dfs: Dictionary mapping analysis names to DataFrames
        metrics: List of metrics to include
        figsize: Figure size
        save_path: Path to save figure
    """
    # Prepare data for heatmap
    rows = []
    for analysis_name, df in subgroup_dfs.items():
        for _, row in df.iterrows():
            for metric in metrics:
                if not pd.isna(row[metric]):
                    rows.append({
                        'Analysis': analysis_name,
                        'Subgroup': row['subgroup'],
                        'Metric': metric.upper(),
                        'Value': row[metric],
                    })

    if not rows:
        print("No valid data to plot")
        return

    plot_df = pd.DataFrame(rows)

    # Create pivot table
    pivot = plot_df.pivot_table(
        index=['Analysis', 'Subgroup'],
        columns='Metric',
        values='Value',
    )

    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, cbar_kws={'label': 'Score'},
                linewidths=0.5, ax=ax)

    ax.set_title('Subgroup Performance Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    else:
        plt.show()

    plt.close()


def generate_subgroup_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clinical_data: pd.DataFrame,
    output_dir: str,
    threshold: float = 0.5,
) -> Dict[str, pd.DataFrame]:
    """Generate comprehensive subgroup analysis report.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        clinical_data: DataFrame with columns like 'age', 'gender', 'severity'
        output_dir: Directory to save results
        threshold: Decision threshold

    Returns:
        Dictionary of subgroup analysis DataFrames

    Expected clinical_data columns:
        - 'age': Patient age
        - 'gender': Gender (0=male, 1=female by default)
        - 'severity': Optional severity score
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Age analysis
    if 'age' in clinical_data.columns:
        print("Analyzing by age...")
        age_df = analyze_by_age(y_true, y_pred, clinical_data['age'].values, threshold=threshold)
        results['age'] = age_df
        age_df.to_csv(os.path.join(output_dir, 'subgroup_age.csv'), index=False)
        plot_subgroup_comparison(
            age_df, metric='auc', title='AUC by Age Group',
            save_path=os.path.join(output_dir, 'subgroup_age_auc.png')
        )

    # Gender analysis
    if 'gender' in clinical_data.columns:
        print("Analyzing by gender...")
        gender_df = analyze_by_gender(y_true, y_pred, clinical_data['gender'].values, threshold=threshold)
        results['gender'] = gender_df
        gender_df.to_csv(os.path.join(output_dir, 'subgroup_gender.csv'), index=False)
        plot_subgroup_comparison(
            gender_df, metric='auc', title='AUC by Gender',
            save_path=os.path.join(output_dir, 'subgroup_gender_auc.png')
        )

    # Severity analysis (if available)
    if 'severity' in clinical_data.columns:
        print("Analyzing by severity...")
        severity_df = analyze_by_severity(y_true, y_pred, clinical_data['severity'].values, threshold=threshold)
        results['severity'] = severity_df
        severity_df.to_csv(os.path.join(output_dir, 'subgroup_severity.csv'), index=False)
        plot_subgroup_comparison(
            severity_df, metric='auc', title='AUC by Severity',
            save_path=os.path.join(output_dir, 'subgroup_severity_auc.png')
        )

    # Combined heatmap
    if len(results) > 0:
        plot_subgroup_heatmap(
            results,
            save_path=os.path.join(output_dir, 'subgroup_heatmap.png')
        )

    # Summary report
    summary_path = os.path.join(output_dir, 'subgroup_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SUBGROUP ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        for analysis_name, df in results.items():
            f.write(f"\n{analysis_name.upper()} ANALYSIS\n")
            f.write("-" * 60 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")

    print(f"\nSubgroup analysis complete. Results saved to {output_dir}")
    return results
