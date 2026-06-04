# Statistical Analysis and Subgroup Analysis

This directory contains tools for statistical significance testing and subgroup analysis of the stroke prognosis model.

## Overview

Two new experiments have been added:

1. **Statistical Significance Tests**
   - DeLong's test for comparing AUC between models
   - Bootstrap confidence intervals for all metrics

2. **Subgroup Analysis**
   - Performance stratified by age groups
   - Performance stratified by gender
   - Performance stratified by severity scores

## Files

- `utils/statistical_tests.py` - DeLong's test and bootstrap CI implementation
- `utils/subgroup_analysis.py` - Subgroup stratification and analysis
- `run_statistical_analysis.py` - Main script to run all analyses

## Usage

### Basic Usage (Bootstrap CI only)

After training your model with `main.py`, run:

```bash
python run_statistical_analysis.py --seed 42
```

This will:
- Load the trained model from `experiments/seed_42/best_model.pth`
- Compute bootstrap 95% confidence intervals for all metrics
- Perform subgroup analysis by age, gender, and severity
- Save results to `experiments/seed_42/statistical_analysis/`

### With Baseline Comparison (DeLong's Test)

If you have baseline model predictions, you can compare against them:

```bash
python run_statistical_analysis.py --seed 42 --baseline_dir ../baseline_results
```

**Baseline predictions format:**
- Save baseline predictions as `.npy` files in the baseline directory
- Naming convention: `<baseline_name>_predictions.npy`
- Each file should contain predicted probabilities (N,) for the positive class
- Example files:
  - `resnet_only_predictions.npy`
  - `bert_only_predictions.npy`
  - `concat_fusion_predictions.npy`

### Advanced Options

```bash
python run_statistical_analysis.py \
    --seed 42 \
    --baseline_dir ../baseline_results \
    --n_bootstraps 2000 \
    --output_dir custom_output_dir
```

Options:
- `--seed`: Random seed to analyze (default: 42)
- `--baseline_dir`: Directory containing baseline predictions
- `--n_bootstraps`: Number of bootstrap samples (default: 1000)
- `--output_dir`: Custom output directory

## Output Structure

```
experiments/seed_42/statistical_analysis/
├── analysis_summary.txt              # Overall summary
├── bootstrap_ci_results.csv          # Bootstrap confidence intervals
├── delong_test_results.csv           # DeLong's test results (if baselines provided)
└── subgroup_analysis/
    ├── subgroup_summary.txt          # Subgroup analysis summary
    ├── subgroup_age.csv              # Age group results
    ├── subgroup_age_auc.png          # Age group visualization
    ├── subgroup_gender.csv           # Gender results
    ├── subgroup_gender_auc.png       # Gender visualization
    ├── subgroup_severity.csv         # Severity results (if available)
    ├── subgroup_severity_auc.png     # Severity visualization
    └── subgroup_heatmap.png          # Combined heatmap
```

## Example Output

### Bootstrap Confidence Intervals

```
AUC         : 0.8523 [0.8201, 0.8845]
ACCURACY    : 0.7800 [0.7400, 0.8200]
F1          : 0.7654 [0.7234, 0.8074]
PRECISION   : 0.7890 [0.7456, 0.8324]
RECALL      : 0.7432 [0.6987, 0.7877]
```

### DeLong's Test Results

```
Ours vs resnet_only:
  AUC (Ours):     0.8523
  AUC (Baseline): 0.7845
  Difference:     +0.0678
  Z-statistic:    3.2456
  P-value:        0.0012
  Significant:    Yes (p < 0.05)
```

### Subgroup Analysis

```
AGE ANALYSIS
------------------------------------------------------------
  subgroup  n_samples  n_positive    auc  accuracy     f1  precision  recall
       <60         45          23  0.856     0.800  0.783      0.810   0.757
     60-69         62          31  0.842     0.774  0.765      0.789   0.742
     70-79         58          29  0.838     0.793  0.781      0.800   0.763
       ≥80         35          17  0.801     0.743  0.721      0.750   0.694
```

## Interpreting Results

### Bootstrap Confidence Intervals
- The 95% CI tells you the range where the true metric value likely falls
- Narrower intervals indicate more stable estimates
- If the CI doesn't include 0.5 for AUC, the model is significantly better than random

### DeLong's Test
- **p < 0.05**: Significant difference between models
- **p ≥ 0.05**: No significant difference
- Use this to claim your model is "significantly better" than baselines

### Subgroup Analysis
- Look for consistent performance across subgroups
- Large performance drops in certain subgroups indicate potential bias
- Small sample sizes (n < 10) should be interpreted cautiously

## Integration with Paper

### For Methods Section
```
Statistical significance was assessed using DeLong's test for AUC comparison
and bootstrap resampling (1000 iterations) for 95% confidence intervals.
Subgroup analyses were performed stratifying by age (<60, 60-69, 70-79, ≥80),
gender, and stroke severity.
```

### For Results Section
```
Our model achieved an AUC of 0.852 [95% CI: 0.820-0.885], significantly
outperforming the image-only baseline (AUC=0.785, p=0.001, DeLong's test).
Subgroup analysis revealed consistent performance across age groups (AUC range:
0.801-0.856) and genders (male: 0.848, female: 0.857).
```

## Notes

- **Clinical data columns**: The script automatically detects age/gender columns
  - Age: looks for columns containing '年龄' or 'age'
  - Gender: looks for columns containing '性别' or 'gender'
  - Severity: uses 'gt' or 'manual' columns if available

- **Baseline predictions**: To generate baseline predictions, modify your baseline
  training scripts to save `y_pred_probs` as `.npy` files

- **Multiple seeds**: Run the analysis for each seed separately, then aggregate
  results manually or extend the script

## Troubleshooting

**Error: Model not found**
- Make sure you've trained the model first with `main.py`
- Check that the seed matches your trained model

**Warning: Age/Gender column not found**
- Check your clinic data Excel file column names
- Modify the column detection logic in `run_statistical_analysis.py` if needed

**Subgroup has too few samples**
- The script skips subgroups with < 5 samples or only one class
- This is expected for small datasets or rare subgroups

## Citation

If you use DeLong's test in your paper, cite:
```
DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988).
Comparing the areas under two or more correlated receiver operating
characteristic curves: a nonparametric approach. Biometrics, 837-845.
```
