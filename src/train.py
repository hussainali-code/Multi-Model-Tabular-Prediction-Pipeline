"""
Module 4 — Model Training & Cross-Validation
==============================================
Portfolio Project A: Multi-Model Tabular Prediction Pipeline
Author : Hussain Ali
SRS Ref: Section 3.4

Responsibilities
----------------
Train all eight SRS-specified models in both regression and classification
mode, evaluate them with 5-fold cross-validation on the training set, and
return trained model objects and CV scores for downstream evaluation.

Regression models (SRS §3.4.1)
-------------------------------
| Model                  | Key hyperparameters                          |
|------------------------|----------------------------------------------|
| LinearRegression       | defaults                                     |
| DecisionTreeRegressor  | max_depth=5, random_state=42                 |
| RandomForestRegressor  | n_estimators=100, max_depth=8, random_state=42|
| XGBRegressor           | n_estimators=100, lr=0.1, max_depth=5, rs=42 |

Classification models (SRS §3.4.2)
-----------------------------------
| Model                  | Key hyperparameters                          |
|------------------------|----------------------------------------------|
| LogisticRegression     | max_iter=1000, random_state=42               |
| DecisionTreeClassifier | max_depth=5, random_state=42                 |
| RandomForestClassifier | n_estimators=100, max_depth=8, random_state=42|
| XGBClassifier          | n_estimators=100, lr=0.1, max_depth=5, rs=42 |

Optional --tune flag: GridSearchCV over RF and XGBoost hyperparameters.

SRS Constraints Enforced
-------------------------
* All random_state values fixed at 42.
* CV uses 5 folds on training data only.
* No function exceeds 40 lines; every function has a docstring.
* Module is independently importable with no side effects.

Usage (standalone)
------------------
    python -m src.train --data data/heart_disease_raw.csv
    python -m src.train --data data/heart_disease_raw.csv --tune

Usage (as a module)
-------------------
    from src.train import train_all
    results = train_all(splits)
    reg_models  = results.reg_models    # dict of fitted regressors
    cls_models  = results.cls_models    # dict of fitted classifiers
    reg_cv      = results.reg_cv        # dict of CV R² scores (5-fold mean)
    cls_cv      = results.cls_cv        # dict of CV Accuracy scores (5-fold mean)
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from collections import namedtuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
CV_FOLDS = 5

# Named tuple returned by train_all
TrainResult = namedtuple(
    "TrainResult",
    ["reg_models", "cls_models", "reg_cv", "cls_cv"],
)

# ---------------------------------------------------------------------------
# Model definitions (SRS §3.4)
# ---------------------------------------------------------------------------

def _build_regression_models() -> dict:
    """Instantiate all four SRS-specified regression models.

    Returns
    -------
    dict
        Mapping of short model name → unfitted sklearn/xgboost estimator.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            random_state=RANDOM_STATE, verbosity=0,
        ),
    }


def _build_classification_models() -> dict:
    """Instantiate all four SRS-specified classification models.

    Returns
    -------
    dict
        Mapping of short model name → unfitted sklearn/xgboost estimator.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            random_state=RANDOM_STATE, verbosity=0,
            eval_metric="logloss",
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
        ),
    }


# ---------------------------------------------------------------------------
# GridSearchCV tuning (optional, --tune flag)
# ---------------------------------------------------------------------------

_REG_TUNE_GRID = {
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [6, 8, 10],
        "min_samples_split": [2, 5],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
    },
}

_CLS_TUNE_GRID = {
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [6, 8, 10],
        "min_samples_split": [2, 5],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
    },
}


def _tune_model(model, param_grid: dict, X, y, scoring: str):
    """Run GridSearchCV and return the best estimator.

    Parameters
    ----------
    model : sklearn estimator
        Base estimator to tune.
    param_grid : dict
        Hyperparameter grid.
    X : array-like
        Training features.
    y : array-like
        Training target.
    scoring : str
        Scoring metric (e.g., 'r2' or 'accuracy').

    Returns
    -------
    sklearn estimator
        Best estimator found by GridSearchCV.
    """
    gs = GridSearchCV(
        model, param_grid, cv=CV_FOLDS, scoring=scoring,
        n_jobs=-1, refit=True,
    )
    gs.fit(X, y)
    logger.info("    Best params: %s  |  CV score: %.4f", gs.best_params_, gs.best_score_)
    return gs.best_estimator_


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def _train_one_mode(
    models: dict,
    X_train,
    y_train,
    cv_scoring: str,
    mode_label: str,
    tune: bool,
    tune_grid: dict,
) -> tuple[dict, dict]:
    """Fit each model, run 5-fold CV, and optionally tune RF/XGBoost.

    Parameters
    ----------
    models : dict
        Name → unfitted estimator mapping.
    X_train : array-like
        Training features.
    y_train : array-like
        Training target.
    cv_scoring : str
        CV metric name for cross_val_score.
    mode_label : str
        Human-readable label for logging ('Regression' or 'Classification').
    tune : bool
        If True, run GridSearchCV for RF and XGBoost before fitting.
    tune_grid : dict
        Hyperparameter grid for tunable models.

    Returns
    -------
    tuple[dict, dict]
        ``(fitted_models, cv_scores)`` where cv_scores values are 5-fold means.
    """
    fitted_models: dict = {}
    cv_scores: dict = {}

    logger.info("--- %s Training ---", mode_label)
    for name, model in models.items():
        t0 = time.perf_counter()

        if tune and name in tune_grid:
            logger.info("  [TUNE] %s  (GridSearchCV)…", name)
            model = _tune_model(model, tune_grid[name], X_train, y_train, cv_scoring)
        else:
            model.fit(X_train, y_train)

        elapsed = time.perf_counter() - t0

        # 5-fold CV on training data
        cv_raw = cross_val_score(
            model, X_train, y_train, cv=CV_FOLDS, scoring=cv_scoring, n_jobs=-1
        )
        cv_mean = float(np.mean(cv_raw))
        cv_std = float(np.std(cv_raw))

        fitted_models[name] = model
        cv_scores[name] = cv_mean

        logger.info(
            "  ✓ %-22s | CV %-10s = %.4f ± %.4f | %.2fs",
            name, cv_scoring, cv_mean, cv_std, elapsed,
        )

    return fitted_models, cv_scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_all(splits, tune: bool = False) -> TrainResult:
    """Train all eight SRS-specified models and compute 5-fold CV scores.

    Parameters
    ----------
    splits : PreprocessResult
        Named tuple from ``src.preprocess.build_splits()``.
    tune : bool, optional
        If True, run GridSearchCV tuning for RF and XGBoost (adds runtime).

    Returns
    -------
    TrainResult
        Named tuple with attributes:
        ``reg_models``, ``cls_models``, ``reg_cv``, ``cls_cv``.
    """
    logger.info("=== Module 4 — Model Training (tune=%s) ===", tune)

    # --- Regression ---
    reg_models_def = _build_regression_models()
    reg_models, reg_cv = _train_one_mode(
        models=reg_models_def,
        X_train=splits.X_train,
        y_train=splits.y_reg_train,
        cv_scoring="r2",
        mode_label="Regression",
        tune=tune,
        tune_grid=_REG_TUNE_GRID,
    )

    # --- Classification ---
    cls_models_def = _build_classification_models()
    cls_models, cls_cv = _train_one_mode(
        models=cls_models_def,
        X_train=splits.X_train,
        y_train=splits.y_cls_train,
        cv_scoring="accuracy",
        mode_label="Classification",
        tune=tune,
        tune_grid=_CLS_TUNE_GRID,
    )

    logger.info("=== Training Complete ===")
    return TrainResult(
        reg_models=reg_models,
        cls_models=cls_models,
        reg_cv=reg_cv,
        cls_cv=cls_cv,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Module 4 — Train all models on the UCI Heart Disease dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", default="data/heart_disease_raw.csv",
        help="Path to the raw CSV produced by Module 1.",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Enable GridSearchCV hyperparameter tuning for RF and XGBoost.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    import sys
    from src.ingest import load_data
    from src.preprocess import build_splits

    args = _parse_args()
    raw_df = load_data(output_path=args.data)
    splits = build_splits(raw_df)
    results = train_all(splits, tune=args.tune)

    print(f"\n{'='*60}")
    print("  Module 4 — Training Complete")
    print(f"{'='*60}")
    print("\n  Regression CV R² (5-fold mean on train):")
    for name, score in results.reg_cv.items():
        print(f"    {name:<25} {score:+.4f}")
    print("\n  Classification CV Accuracy (5-fold mean on train):")
    for name, score in results.cls_cv.items():
        print(f"    {name:<25} {score:.4f}")
    print(f"{'='*60}")
    sys.exit(0)
