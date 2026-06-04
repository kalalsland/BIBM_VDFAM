"""Central configuration for stroke_multimodal_prognosis.

External paths (raw data, model weights) and API keys are read from
environment variables so this repository can be published without exposing
private paths or secrets.

Quick start
-----------
1. ``cp .env.example .env`` and fill in your values.
2. ``pip install python-dotenv`` (included in requirements.txt).
3. Run ``python main.py`` from the project root.

If python-dotenv is not installed the variables are still read from the
process environment, so ``export VAR=value`` / ``set VAR=value`` also works.
"""

import os
from pathlib import Path

# ── Load .env if python-dotenv is available ───────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass  # python-dotenv is optional; fall back to process environment

# ── Package-relative paths ────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_ENCODED = Path(os.environ.get("ENCODED_DATA_DIR", str(_HERE / "data" / "encoded")))

CONFIG = {
    # ------------------------------------------------------------------ data
    "clinic_path":     str(_ENCODED / "clinic_data_1.xlsx"),
    "image_feat_path": str(_ENCODED / "image_feature.xlsx"),
    "text_feat_path":  str(_ENCODED / "medical_text_features_by_qianwen_best.xlsx"),

    # ---------------------------------------------------------------- training
    "batch_size":         64,
    "epochs":             100,
    "learning_rate":      1e-4,
    "min_lr":             1e-7,
    "lambda_cl":          0.02,
    "lambda_align":       0.20,
    "patience":           30,
    "weight_decay":       0,
    "use_lr_scheduler":   True,
    # "cosine_warmup" (batch-level) | "cosine" (epoch) | "reduce_on_plateau"
    "scheduler_type":     "cosine_warmup",
    "warmup_epochs":      15,
    "grad_clip":          1.0,

    # -------------------------------------------- data augmentation / balance
    "use_smote":          True,
    # "none" | "weighted" | "tabular_copy" | "image_copy"
    "smote_method":       "tabular_copy",
    "use_mixup":          True,
    "mixup_alpha":        0.4,
    "use_gaussian_noise": True,

    # ------------------------------------------------------------------ model
    "hidden_dim":      512,
    "projection_dim":  256,
    "num_heads":       16,
    "dropout_rate":    0.5,

    # ----------------------------------------------------------------- output
    "output_dir":       str(_HERE / "experiments"),
    "save_best_model":  True,

    # ----------------------------------------------------------------- Optuna
    "optuna": {
        "n_trials":  1,
        "timeout":   14400,
        "pruning":   True,
        "direction": "maximize",
    },

    # ----------------------------------------------- cross-val / data splits
    "use_cross_validation": False,
    "n_splits":             5,
    "test_size":            0.2,
    "val_size":             0.2,

    # -------------------------------------------------- multi-seed evaluation
    "num_random_seeds": 4,
    "initial_seed":     39,

    # threshold tuning (grid search on val set to maximise F1)
    "use_threshold_tuning": True,
}
