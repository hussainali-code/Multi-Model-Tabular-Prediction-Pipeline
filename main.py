"""
main.py — CLI Entry Point
==========================
Portfolio Project A: Multi-Model Tabular Prediction Pipeline
Author : Hussain Ali
SRS Ref: Section 3.7

Wires together all six modules into a single end-to-end pipeline
controlled entirely by command-line arguments.

Example Usage
-------------
# Full pipeline (both modes, all models, with SHAP)
    python main.py --data data/heart_disease_raw.csv --mode both --explain

# Classification only, specific models
    python main.py --data data/heart_disease_raw.csv --mode classification --models lr,rf

# Regression with hyperparameter tuning (adds ~2 min)
    python main.py --data data/heart_disease_raw.csv --mode regression --tune

# Fast run — skip all plots
    python main.py --data data/heart_disease_raw.csv --no-plots

CLI Arguments (SRS §3.7)
-------------------------
--data      Path to input CSV file                          [required]
--mode      regression | classification | both              [default: both]
--models    Comma-separated: lr,dt,rf,xgb                  [default: all]
--tune      Enable GridSearchCV for RF and XGBoost          [flag]
--explain   Generate SHAP plots (~30s extra runtime)        [flag]
--sample    Test sample index for SHAP waterfall plot       [default: 0]
--output    Output directory path                           [default: ./outputs]
--no-plots  Skip all plot generation                        [flag]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

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
# Model alias map (SRS §3.7 --models argument)
# ---------------------------------------------------------------------------
MODEL_ALIASES: dict[str, str] = {
    "lr":  "Linear Regression",     # or "Logistic Regression" for cls
    "dt":  "Decision Tree",
    "rf":  "Random Forest",
    "xgb": "XGBoost",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse all SRS-specified CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Multi-Model Tabular Prediction Pipeline\n"
            "Portfolio Project A — Hussain Ali\n"
            "UCI Heart Disease Dataset (Cleveland, 303 patients)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --data data/heart_disease_raw.csv --mode both --explain\n"
            "  python main.py --data data/heart_disease_raw.csv --mode classification --models lr,rf\n"
            "  python main.py --data data/heart_disease_raw.csv --tune --no-plots\n"
        ),
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to input CSV file (raw or pre-ingested).",
    )
    parser.add_argument(
        "--mode", default="both",
        choices=["regression", "classification", "both"],
        help="Which task mode(s) to run.",
    )
    parser.add_argument(
        "--models", default="lr,dt,rf,xgb",
        help="Comma-separated model keys to use: lr, dt, rf, xgb.",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Enable GridSearchCV hyperparameter tuning for RF and XGBoost.",
    )
    parser.add_argument(
        "--explain", action="store_true",
        help="Generate SHAP explainability plots (adds ~30s runtime).",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Index of the test sample to use for the SHAP waterfall plot.",
    )
    parser.add_argument(
        "--output", default="outputs",
        help="Root output directory for plots and result CSVs.",
    )
    parser.add_argument(
        "--no-plots", action="store_true", dest="no_plots",
        help="Skip all matplotlib plot generation (faster automated runs).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Model filtering
# ---------------------------------------------------------------------------

def _filter_models(selected_keys: str) -> set[str]:
    """Parse the --models argument into a set of full model names.

    Parameters
    ----------
    selected_keys : str
        Comma-separated short keys, e.g. 'lr,rf,xgb'.

    Returns
    -------
    set[str]
        Set of full model names (e.g. {'Linear Regression', 'Random Forest'}).
    """
    keys = [k.strip().lower() for k in selected_keys.split(",")]
    invalid = [k for k in keys if k not in MODEL_ALIASES]
    if invalid:
        logger.error(
            "Unknown model key(s): %s  |  Valid keys: %s",
            invalid, list(MODEL_ALIASES.keys()),
        )
        sys.exit(1)
    return {MODEL_ALIASES[k] for k in keys}


def _apply_model_filter(train_result, selected: set[str]):
    """Remove models not selected by --models from the TrainResult.

    Parameters
    ----------
    train_result : TrainResult named tuple from Module 4.
    selected     : Set of full model name strings to keep.

    Returns
    -------
    TrainResult
        Filtered copy of train_result.
    """
    from src.train import TrainResult  # noqa: PLC0415

    # Keep only selected models; always keep all for CV dict consistency
    reg_m = {k: v for k, v in train_result.reg_models.items() if k in selected}
    cls_m = {k: v for k, v in train_result.cls_models.items() if k in selected}
    reg_cv = {k: v for k, v in train_result.reg_cv.items() if k in selected}
    cls_cv = {k: v for k, v in train_result.cls_cv.items() if k in selected}
    return TrainResult(reg_models=reg_m, cls_models=cls_m, reg_cv=reg_cv, cls_cv=cls_cv)


def _apply_mode_filter(train_result, mode: str):
    """Zero out models for the inactive mode so evaluate_all skips them.

    Parameters
    ----------
    train_result : TrainResult.
    mode         : 'regression', 'classification', or 'both'.

    Returns
    -------
    TrainResult
        Mode-filtered copy.
    """
    from src.train import TrainResult  # noqa: PLC0415

    if mode == "regression":
        return TrainResult(
            reg_models=train_result.reg_models, cls_models={},
            reg_cv=train_result.reg_cv, cls_cv={},
        )
    if mode == "classification":
        return TrainResult(
            reg_models={}, cls_models=train_result.cls_models,
            reg_cv={}, cls_cv=train_result.cls_cv,
        )
    return train_result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Run the full end-to-end pipeline based on CLI arguments.

    Parameters
    ----------
    argv : list[str] or None
        Argument list (None = sys.argv).

    Returns
    -------
    int
        Exit code (0 = success, 1 = failure).
    """
    args = _parse_args(argv)
    t_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("  Multi-Model Tabular Prediction Pipeline")
    logger.info("  Author: Hussain Ali")
    logger.info("=" * 60)
    logger.info("  mode=%s | models=%s | tune=%s | explain=%s",
                args.mode, args.models, args.tune, args.explain)
    logger.info("  output=%s | no_plots=%s", args.output, args.no_plots)
    logger.info("=" * 60)

    # -- Imports (lazy: only after arg validation) --
    from src.ingest import load_data
    from src.preprocess import build_splits
    from src.train import train_all
    from src.evaluate import evaluate_all

    # -- Step 1: Ingest --
    raw_df = load_data(output_path=args.data)

    # -- Step 2: Preprocess --
    splits = build_splits(raw_df)

    # -- Step 3: Train (all models; filter after) --
    selected_models = _filter_models(args.models)
    train_result = train_all(splits, tune=args.tune)
    train_result = _apply_model_filter(train_result, selected_models)
    train_result = _apply_mode_filter(train_result, args.mode)

    # -- Step 4: Evaluate --
    eval_result = evaluate_all(
        splits, train_result,
        output_dir=args.output,
        no_plots=args.no_plots,
    )

    # -- Step 5: Explain (optional) --
    if args.explain:
        # Need at least one classifier with feature_importances_ for SHAP
        has_cls = bool(train_result.cls_models)
        if args.mode in ("classification", "both") and has_cls:
            from src.explain import run_explain
            run_explain(
                splits, train_result, eval_result,
                sample_idx=args.sample,
                output_dir=args.output,
            )
        else:
            logger.warning("--explain requires classification mode with RF or XGB. Skipping SHAP.")

    elapsed = time.perf_counter() - t_start
    logger.info("=" * 60)
    logger.info("  Pipeline complete in %.1f seconds.", elapsed)
    logger.info("  Outputs → %s", Path(args.output).resolve())
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
