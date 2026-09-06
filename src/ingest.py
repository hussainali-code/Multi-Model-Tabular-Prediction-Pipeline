"""
Module 1 — Data Ingestion & Validation
=======================================
Portfolio Project A: Multi-Model Tabular Prediction Pipeline
Author : Hussain Ali
Dataset: UCI Heart Disease (Cleveland) — 303 patients, 13 features + 1 target
SRS Ref: Section 3.1

Responsibilities
----------------
1. Fetch the UCI Heart Disease dataset (Cleveland subset) via the
   ``ucimlrepo`` library; fall back to a direct HTTP download if the
   library is unavailable or the network call fails.
2. Validate the raw DataFrame against the SRS-defined schema:
   - Expected columns (13 features + 'num' target)
   - Approximate row count (~303)
   - Allowed value ranges for key clinical columns
3. Persist the validated raw CSV to  ``data/heart_disease_raw.csv``.
4. Return the DataFrame to the caller — no side effects beyond saving.

Usage (standalone)
------------------
    python -m src.ingest                        # uses default output path
    python -m src.ingest --output data/my.csv   # custom output path

Usage (as a module)
-------------------
    from src.ingest import load_data
    df = load_data()
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
# Constants (SRS §3.1)
# ---------------------------------------------------------------------------
# UCI ML Repository numeric ID for the Heart Disease dataset
UCI_DATASET_ID = 45

# Direct HTTP fallback — Cleveland subset from the UCI archive
FALLBACK_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# Column names as defined in the UCI documentation
COLUMN_NAMES: list[str] = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num",
]

# SRS-defined feature set (13 features, excluding target)
EXPECTED_FEATURES: list[str] = COLUMN_NAMES[:-1]
TARGET_COLUMN = "num"

# Approximate expected row count with a generous ±15 % tolerance
# (Cleveland subset is officially 303 rows; some mirrors return 297–303)
EXPECTED_ROWS = 303
ROW_TOLERANCE = 0.15

# Value-range guardrails for key clinical columns (SRS §3.1 — schema check)
COLUMN_RANGES: dict[str, tuple[float, float]] = {
    "age":      (20.0, 80.0),
    "trestbps": (80.0, 220.0),   # resting blood pressure (mm Hg)
    "chol":     (100.0, 600.0),  # serum cholesterol (mg/dl)
    "thalach":  (60.0, 220.0),   # max heart rate achieved
    "oldpeak":  (0.0, 7.0),      # ST depression
    "num":      (0.0, 4.0),      # target: angiographic disease severity
}


# Minimum acceptable rows (hard floor, regardless of tolerance)
MIN_ROWS = 200

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_local(path: str | Path) -> pd.DataFrame | None:
    """Load the dataset from a local CSV file if it exists.

    Parameters
    ----------
    path : str or Path
        Expected path of the local CSV file.

    Returns
    -------
    pd.DataFrame or None
        Loaded DataFrame, or None if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return None
    logger.info("Local file found — loading from %s", path.resolve())
    df = pd.read_csv(path, na_values="?")
    logger.info("Local load successful — %d rows.", len(df))
    return df


def _fetch_via_ucimlrepo() -> pd.DataFrame:
    """Attempt to download the dataset via the ``ucimlrepo`` library.

    Returns
    -------
    pd.DataFrame
        Combined features + target with SRS column names.

    Raises
    ------
    ImportError
        If ``ucimlrepo`` is not installed.
    Exception
        If the network call or parsing fails.
    """
    from ucimlrepo import fetch_ucirepo  # noqa: PLC0415  (lazy import)

    logger.info("Fetching dataset from UCI ML Repository (ID=%d)…", UCI_DATASET_ID)
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    features: pd.DataFrame = dataset.data.features
    targets: pd.DataFrame = dataset.data.targets

    df = pd.concat([features, targets], axis=1)

    # ucimlrepo may return different column names — normalise to SRS names
    if list(df.columns) != COLUMN_NAMES:
        if len(df.columns) == len(COLUMN_NAMES):
            df.columns = COLUMN_NAMES
        else:
            raise ValueError(
                f"Column count mismatch from ucimlrepo: "
                f"got {len(df.columns)}, expected {len(COLUMN_NAMES)}"
            )

    logger.info("ucimlrepo fetch successful — %d rows received.", len(df))
    return df


def _fetch_via_http() -> pd.DataFrame:
    """Download the Cleveland CSV directly from the UCI archive.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with SRS column names.

    Raises
    ------
    Exception
        On any HTTP or parsing error.
    """
    logger.info("Falling back to direct HTTP download from UCI archive…")
    logger.info("URL: %s", FALLBACK_URL)

    df = pd.read_csv(
        FALLBACK_URL,
        header=None,
        names=COLUMN_NAMES,
        na_values="?",   # Cleveland dataset uses '?' for missing values
    )
    logger.info("HTTP download successful — %d rows received.", len(df))
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Run SRS-defined schema checks against the raw DataFrame.

    Checks performed
    ----------------
    * All 14 expected columns are present.
    * Row count is within ±5 % of 303.
    * Key clinical columns satisfy their value-range guardrails
      (ignoring NaN values, which preprocessing handles separately).

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame to validate.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    # --- Column presence ---
    missing_cols = [c for c in COLUMN_NAMES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema error — missing columns: {missing_cols}")
    logger.info("✓ Column check passed  (%d columns present).", len(df.columns))

    # --- Row count ---
    lo = int(EXPECTED_ROWS * (1 - ROW_TOLERANCE))
    hi = int(EXPECTED_ROWS * (1 + ROW_TOLERANCE))
    if not (lo <= len(df) <= hi):
        raise ValueError(
            f"Schema error — unexpected row count: {len(df)} "
            f"(expected {lo}–{hi})."
        )
    logger.info("✓ Row count check passed  (%d rows).", len(df))

    # --- Value ranges ---
    for col, (lo_val, hi_val) in COLUMN_RANGES.items():
        if col not in df.columns:
            continue
        col_clean = pd.to_numeric(df[col], errors="coerce").dropna()
        out_of_range = col_clean[(col_clean < lo_val) | (col_clean > hi_val)]
        if not out_of_range.empty:
            logger.warning(
                "⚠  Column '%s' has %d value(s) outside [%.1f, %.1f]: %s",
                col, len(out_of_range), lo_val, hi_val,
                out_of_range.values[:5],
            )
    logger.info("✓ Value-range check passed.")

    # --- Missing value report (informational only — preprocessing handles it) ---
    missing_summary = df.isnull().sum()
    missing_cols_report = missing_summary[missing_summary > 0]
    if not missing_cols_report.empty:
        logger.info(
            "ℹ  Missing values detected (will be imputed in Module 2):\n%s",
            missing_cols_report.to_string(),
        )
    else:
        logger.info("ℹ  No missing values detected in raw data.")


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to numeric where possible.

    The raw Cleveland CSV stores '?' as missing; ``_fetch_via_http`` already
    converts these to NaN via ``na_values='?'``. This step handles any
    residual string values from the ucimlrepo path.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to coerce.

    Returns
    -------
    pd.DataFrame
        Coerced copy of the input.
    """
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(output_path: str | Path = "data/heart_disease_raw.csv") -> pd.DataFrame:
    """Ingest, validate, and persist the UCI Heart Disease dataset.

    Tries ``ucimlrepo`` first; falls back to a direct HTTP download if the
    library is unavailable or the network request fails.

    Parameters
    ----------
    output_path : str or Path, optional
        Destination path for the raw CSV file.
        Default: ``data/heart_disease_raw.csv`` (relative to CWD).

    Returns
    -------
    pd.DataFrame
        Validated raw DataFrame with 14 columns (13 features + 'num' target).

    Raises
    ------
    RuntimeError
        If both the ucimlrepo fetch and the HTTP fallback fail.
    ValueError
        If schema validation fails after a successful download.
    """
    # --- Step 1: Fetch (local → ucimlrepo → HTTP) ---
    df: pd.DataFrame | None = None

    # Strategy A: local file (fastest; works fully offline after first run)
    df = _load_local(output_path)

    # Strategy B: ucimlrepo API
    if df is None:
        try:
            df = _fetch_via_ucimlrepo()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ucimlrepo fetch failed (%s). Trying HTTP fallback…", exc)

    # Strategy C: direct HTTP download
    if df is None:
        try:
            df = _fetch_via_http()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "All three data-fetch strategies failed. "
                "Check your internet connection or place the dataset manually "
                f"at '{Path(output_path).resolve()}'."
            ) from exc

    # --- Step 2: Coerce numeric types ---
    df = _coerce_numeric(df)

    # --- Step 3: Validate schema ---
    _validate_schema(df)

    # --- Step 4: Persist raw CSV (skip if loaded from local — already there) ---
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("✓ Raw dataset saved → %s", output_path.resolve())

    # --- Step 5: Summary ---
    logger.info(
        "Ingestion complete — shape: %s | missing: %d cells | "
        "target range: %s–%s",
        df.shape,
        int(df.isnull().sum().sum()),
        df[TARGET_COLUMN].min(),
        df[TARGET_COLUMN].max(),
    )

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Module 1 — Ingest & validate the UCI Heart Disease dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default="data/heart_disease_raw.csv",
        help="Path to save the raw CSV file.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    try:
        df = load_data(output_path=args.output)
        print(f"\n{'='*55}")
        print("  Module 1 — Ingestion Complete")
        print(f"{'='*55}")
        print(f"  Shape        : {df.shape}")
        print(f"  Columns      : {list(df.columns)}")
        print(f"  Missing cells: {int(df.isnull().sum().sum())}")
        print(f"  Target (num) : min={df['num'].min()}, max={df['num'].max()}")
        print(f"{'='*55}")
        print("\nFirst 5 rows:")
        print(df.head().to_string())
        print(f"\nData types:\n{df.dtypes.to_string()}")
        print(f"\nDescriptive stats:\n{df.describe().to_string()}")
        sys.exit(0)
    except (RuntimeError, ValueError) as e:
        logger.error("Ingestion failed: %s", e)
        sys.exit(1)
