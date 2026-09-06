"""
Module 2 — Preprocessing Pipeline
===================================
Portfolio Project A: Multi-Model Tabular Prediction Pipeline
Author : Hussain Ali
SRS Ref: Section 3.2

Responsibilities
----------------
1. Impute missing values:
   - Numerical columns  → median imputation (no rows dropped).
   - Categorical columns → mode imputation.
2. Engineer targets:
   - Regression target  : ``num`` column as-is (integer 0–4).
   - Classification target: binary flag where ``num > 0 = 1``.
3. One-hot encode all ``object``-dtype columns **plus** the declared
   categorical integer columns (cp, restecg, slope, thal) using
   ``pd.get_dummies(drop_first=True)`` to avoid multicollinearity.
4. Perform an 80/20 stratified train/test split (``random_state=42``).
5. Standardise continuous numerical columns with ``StandardScaler``
   fitted **exclusively on training data** — zero data leakage.
6. Return a named tuple containing everything downstream modules need.

SRS Constraints Enforced
-------------------------
* Scaler fitted on X_train only; X_test is transformed, never fitted.
* Categorical columns (cp, restecg, slope, thal) treated as nominal,
  not ordinal, and one-hot encoded.
* random_state=42 everywhere for reproducibility.
* No function exceeds 40 lines; every function has a docstring.

Usage (standalone)
------------------
    python -m src.preprocess --data data/heart_disease_raw.csv

Usage (as a module)
-------------------
    from src.preprocess import build_splits
    splits = build_splits(df)
    X_train, X_test = splits.X_train, splits.X_test
"""

from __future__ import annotations

import argparse
import logging
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
# Constants (SRS §3.2)
# ---------------------------------------------------------------------------
TARGET_COLUMN = "num"

# SRS-declared categorical integer columns to be one-hot encoded
# (These are coded as integers in the raw data but are nominal categories)
CATEGORICAL_INT_COLS: list[str] = ["cp", "restecg", "slope", "thal"]

# Columns to exclude from scaling (binary flags or targets)
BINARY_COLS: list[str] = ["sex", "fbs", "exang"]

# SRS split parameters
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Named tuple returned by build_splits — allows attribute-style access
PreprocessResult = namedtuple(
    "PreprocessResult",
    [
        "X_train",       # pd.DataFrame — scaled training features
        "X_test",        # pd.DataFrame — scaled test features
        "y_reg_train",   # pd.Series    — regression target (train)
        "y_reg_test",    # pd.Series    — regression target (test)
        "y_cls_train",   # pd.Series    — classification target (train)
        "y_cls_test",    # pd.Series    — classification target (test)
        "scaler",        # fitted StandardScaler instance
        "feature_names", # list[str]    — column names of X_train / X_test
    ],
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _impute(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in-place (SRS §3.2: no rows dropped).

    - Numerical columns : median imputation.
    - Object-dtype cols : mode imputation (first mode value).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame; may contain NaN values.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with all NaN values filled.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype == object:
            fill_val = df[col].mode().iloc[0]
            logger.info("  Imputing '%s' (categorical) with mode: %s", col, fill_val)
        else:
            fill_val = df[col].median()
            logger.info("  Imputing '%s' (numerical)   with median: %.4f", col, fill_val)
        df[col] = df[col].fillna(fill_val)
    return df


def _engineer_targets(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Derive regression and classification targets from the 'num' column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the 'num' column.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        ``(y_reg, y_cls)`` where:
        - ``y_reg``  is ``num`` as integer (0–4).
        - ``y_cls``  is a binary flag (``num > 0``).
    """
    y_reg = df[TARGET_COLUMN].astype(int)
    y_cls = (df[TARGET_COLUMN] > 0).astype(int)
    logger.info(
        "Targets engineered — regression: %s classes | "
        "classification: %d positives / %d total",
        sorted(y_reg.unique()),
        y_cls.sum(),
        len(y_cls),
    )
    return y_reg, y_cls


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode nominal features (SRS §3.2, drop_first=True).

    Encodes:
    * All ``object``-dtype columns (if any).
    * SRS-declared categorical integer columns: cp, restecg, slope, thal.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (target column already removed).

    Returns
    -------
    pd.DataFrame
        Encoded DataFrame; shape is (n_rows, n_encoded_cols).
    """
    # Cast declared categoricals to string so get_dummies treats them as nominal
    for col in CATEGORICAL_INT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df = pd.get_dummies(df, drop_first=True)
    logger.info("Encoding complete — %d feature columns after OHE.", df.shape[1])
    return df


def _scale(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standardise continuous features (SRS §3.2 — no data leakage).

    Fits StandardScaler on ``X_train`` only; transforms both splits.
    Binary and boolean columns (already 0/1) are left untouched.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Test feature matrix.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, StandardScaler]
        ``(X_train_scaled, X_test_scaled, fitted_scaler)``
    """
    # Identify columns to scale: numeric, not boolean, not OHE dummies
    # OHE columns are identifiable as boolean dtype after get_dummies
    scale_cols = [
        c for c in X_train.columns
        if X_train[c].dtype in (np.float64, np.int64, float, int)
        and X_train[c].nunique() > 2  # skip binary flags
        and not X_train[c].isin([0, 1]).all()  # skip 0/1 encoded dummies
    ]
    logger.info("Scaling %d continuous columns: %s", len(scale_cols), scale_cols)

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])   # transform only!

    return X_train, X_test, scaler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_splits(df: pd.DataFrame) -> PreprocessResult:
    """Run the full preprocessing pipeline and return train/test splits.

    Steps (SRS §3.2):
    1. Impute missing values (median / mode).
    2. Engineer regression + classification targets.
    3. Drop target column from features.
    4. One-hot encode nominal features.
    5. Stratified 80/20 train/test split.
    6. Fit StandardScaler on X_train; transform both splits.

    Parameters
    ----------
    df : pd.DataFrame
        Raw validated DataFrame from ``src.ingest.load_data()``.

    Returns
    -------
    PreprocessResult
        Named tuple with attributes:
        X_train, X_test, y_reg_train, y_reg_test,
        y_cls_train, y_cls_test, scaler, feature_names.
    """
    logger.info("=== Module 2 — Preprocessing Pipeline ===")

    # Step 1 — Impute
    logger.info("Step 1: Imputing missing values…")
    df = _impute(df)

    # Step 2 — Engineer targets
    logger.info("Step 2: Engineering targets…")
    y_reg, y_cls = _engineer_targets(df)

    # Step 3 — Drop target
    X = df.drop(columns=[TARGET_COLUMN])

    # Step 4 — Encode
    logger.info("Step 3: One-hot encoding nominal features…")
    X = _encode(X)

    # Step 5 — Stratified split on classification target (preserves class ratio)
    logger.info("Step 4: 80/20 stratified split (random_state=%d)…", RANDOM_STATE)
    (X_train, X_test,
     y_reg_train, y_reg_test,
     y_cls_train, y_cls_test) = train_test_split(
        X, y_reg, y_cls,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_cls,         # stratify on binary classification target
    )

    # Step 6 — Scale (train-fit only)
    logger.info("Step 5: Standardising continuous features…")
    X_train, X_test, scaler = _scale(X_train, X_test)

    # Convert bool dtypes to int (prevents sklearn warnings)
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    feature_names = list(X_train.columns)

    logger.info("--- Preprocessing Summary ---")
    logger.info("  X_train shape   : %s", X_train.shape)
    logger.info("  X_test  shape   : %s", X_test.shape)
    logger.info("  Features        : %d", len(feature_names))
    logger.info(
        "  y_cls balance (train): %d positive / %d total (%.1f%%)",
        y_cls_train.sum(), len(y_cls_train),
        100 * y_cls_train.mean(),
    )
    logger.info("=== Preprocessing Complete ===")

    return PreprocessResult(
        X_train=X_train,
        X_test=X_test,
        y_reg_train=y_reg_train,
        y_reg_test=y_reg_test,
        y_cls_train=y_cls_train,
        y_cls_test=y_cls_test,
        scaler=scaler,
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Module 2 — Preprocess the UCI Heart Disease dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default="data/heart_disease_raw.csv",
        help="Path to the raw CSV file produced by Module 1.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    import sys
    from src.ingest import load_data

    args = _parse_args()
    raw_df = load_data(output_path=args.data)
    splits = build_splits(raw_df)

    print(f"\n{'='*55}")
    print("  Module 2 — Preprocessing Complete")
    print(f"{'='*55}")
    print(f"  X_train : {splits.X_train.shape}")
    print(f"  X_test  : {splits.X_test.shape}")
    print(f"  Features: {splits.feature_names}")
    print(f"\n  Class balance (train):")
    vc = splits.y_cls_train.value_counts().sort_index()
    for k, v in vc.items():
        print(f"    Class {k}: {v} samples ({100*v/len(splits.y_cls_train):.1f}%)")
    print(f"\n  Regression target (train) — min: {splits.y_reg_train.min()}, "
          f"max: {splits.y_reg_train.max()}, "
          f"mean: {splits.y_reg_train.mean():.2f}")
    print(f"\n  Scaler mean (first 5): {splits.scaler.mean_[:5].round(4)}")
    print(f"  Scaler std  (first 5): {splits.scaler.scale_[:5].round(4)}")
    print(f"{'='*55}")
    sys.exit(0)
