"""
Module 5 — Evaluation & Metric Comparison Plots
=================================================
Portfolio Project A: Multi-Model Tabular Prediction Pipeline
Author : Hussain Ali
SRS Ref: Section 3.5

Responsibilities
----------------
1. Compute all SRS-specified metrics on the **held-out test set**:
   - Regression  : MAE, RMSE, R², CV R² (from Module 4)
   - Classification: Accuracy, Precision, Recall, F1, ROC-AUC, CV Accuracy
2. Export a unified ``metrics_comparison.csv`` to ``outputs/results/``.
3. Generate comparison bar charts:
   - Regression  : one chart per metric (MAE, RMSE, R²)
   - Classification: one chart per metric (Accuracy, F1, ROC-AUC)
   - Combined overview charts for each mode
4. Print an acceptance-criterion summary table to stdout.

SRS Acceptance Thresholds
--------------------------
Regression:
  LinearRegression  → R² ≥ 0.30, RMSE ≤ 1.00
  DecisionTree      → R² ≥ 0.40, RMSE ≤ 0.95
  RandomForest      → R² ≥ 0.55, RMSE ≤ 0.85
  XGBoost           → R² ≥ 0.60, RMSE ≤ 0.80
Classification:
  LogisticRegression → Acc ≥ 0.80, F1 ≥ 0.78, ROC-AUC ≥ 0.85
  DecisionTree       → Acc ≥ 0.75, F1 ≥ 0.73
  RandomForest       → Acc ≥ 0.83, F1 ≥ 0.82, ROC-AUC ≥ 0.88
  XGBoost            → Acc ≥ 0.84, F1 ≥ 0.83, ROC-AUC ≥ 0.89

SRS Constraints Enforced
-------------------------
* Metrics computed on test set only; CV scores injected from Module 4.
* No function exceeds 40 lines; every function has a docstring.
* Module is independently importable with no side effects.

Usage (standalone)
------------------
    python -m src.evaluate --data data/heart_disease_raw.csv

Usage (as a module)
-------------------
    from src.evaluate import evaluate_all
    report = evaluate_all(splits, train_result)
    print(report.reg_df)   # pd.DataFrame of regression metrics
    print(report.cls_df)   # pd.DataFrame of classification metrics
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

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
# Style (matches Module 3 dark theme)
# ---------------------------------------------------------------------------
BG_COLOR = "#1a1a2e"
PANEL_COLOR = "#16213e"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4a"
PALETTE = ["#e94560", "#16c79a", "#f5a623", "#a8d8ea"]
PASS_COLOR = "#16c79a"
FAIL_COLOR = "#e94560"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": PANEL_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "legend.facecolor": PANEL_COLOR,
    "legend.edgecolor": GRID_COLOR,
})

# ---------------------------------------------------------------------------
# SRS acceptance thresholds
# ---------------------------------------------------------------------------
REG_THRESHOLDS: dict[str, dict] = {
    "Linear Regression": {"r2_min": 0.30, "rmse_max": 1.00},
    "Decision Tree":     {"r2_min": 0.40, "rmse_max": 0.95},
    "Random Forest":     {"r2_min": 0.55, "rmse_max": 0.85},
    "XGBoost":           {"r2_min": 0.60, "rmse_max": 0.80},
}

CLS_THRESHOLDS: dict[str, dict] = {
    "Logistic Regression": {"acc_min": 0.80, "f1_min": 0.78, "auc_min": 0.85},
    "Decision Tree":       {"acc_min": 0.75, "f1_min": 0.73, "auc_min": None},
    "Random Forest":       {"acc_min": 0.83, "f1_min": 0.82, "auc_min": 0.88},
    "XGBoost":             {"acc_min": 0.84, "f1_min": 0.83, "auc_min": 0.89},
}

# Named result
from collections import namedtuple  # noqa: E402
EvalResult = namedtuple("EvalResult", ["reg_df", "cls_df"])


# ---------------------------------------------------------------------------
# Private helpers — metric computation
# ---------------------------------------------------------------------------

def _regression_metrics(model, X_test, y_test, cv_r2: float) -> dict:
    """Compute MAE, RMSE, R², and inject CV R² for one regression model.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_test : array-like
    y_test : array-like
    cv_r2  : float — pre-computed 5-fold CV R² from Module 4

    Returns
    -------
    dict of metric name → value
    """
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R²": r2, "CV R²": cv_r2}


def _classification_metrics(model, X_test, y_test, cv_acc: float) -> dict:
    """Compute Accuracy, Precision, Recall, F1, ROC-AUC for one classifier.

    Parameters
    ----------
    model  : fitted sklearn estimator
    X_test : array-like
    y_test : array-like
    cv_acc : float — pre-computed 5-fold CV Accuracy from Module 4

    Returns
    -------
    dict of metric name → value
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else float("nan")
    return {
        "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1": f1, "ROC-AUC": auc, "CV Accuracy": cv_acc,
    }


# ---------------------------------------------------------------------------
# Private helpers — plotting
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    """Save figure and close it to free memory."""
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("  Saved → %s", path.name)


def _bar_chart(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    higher_is_better: bool = True,
) -> None:
    """Render a horizontal bar chart comparing models on one metric.

    Parameters
    ----------
    df            : pd.DataFrame with 'Model' and ``metric`` columns.
    metric        : Column name to plot.
    title         : Chart title.
    output_path   : Destination PNG path.
    higher_is_better : Controls bar colour ordering (green = better).
    """
    vals = df[metric].values.astype(float)
    names = df["Model"].values

    sorted_idx = np.argsort(vals)[::-1] if higher_is_better else np.argsort(vals)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(BG_COLOR)
    bars = ax.barh(
        [names[i] for i in sorted_idx],
        [vals[i] for i in sorted_idx],
        color=[colors[i] for i in sorted_idx],
        edgecolor=PANEL_COLOR, height=0.55,
    )
    for bar, val in zip(bars, [vals[i] for i in sorted_idx]):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9,
        )
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, output_path)


def _overview_chart(df: pd.DataFrame, metrics: list[str], title: str, output_path: Path) -> None:
    """Multi-metric grouped bar chart for a quick model overview.

    Parameters
    ----------
    df      : DataFrame with 'Model' column + one column per metric.
    metrics : List of metric column names to include.
    title   : Chart title.
    output_path : Destination PNG path.
    """
    n_models = len(df)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_COLOR)
    for i, metric in enumerate(metrics):
        vals = df[metric].values.astype(float)
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=metric,
                      color=PALETTE[i % len(PALETTE)], edgecolor=PANEL_COLOR, alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"].values, rotation=12, ha="right", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Acceptance criterion check
# ---------------------------------------------------------------------------

def _check_regression_thresholds(df: pd.DataFrame) -> None:
    """Log a PASS/FAIL table for regression acceptance criteria."""
    logger.info("\n  ── Regression Acceptance Criteria ──")
    for _, row in df.iterrows():
        name = row["Model"]
        if name not in REG_THRESHOLDS:
            continue
        t = REG_THRESHOLDS[name]
        r2_ok = row["R²"] >= t["r2_min"]
        rmse_ok = row["RMSE"] <= t["rmse_max"]
        status = "✅ PASS" if (r2_ok and rmse_ok) else "❌ FAIL"
        logger.info(
            "  %s  %-22s | R²=%.3f (≥%.2f)%s | RMSE=%.3f (≤%.2f)%s",
            status, name,
            row["R²"], t["r2_min"], " ✓" if r2_ok else " ✗",
            row["RMSE"], t["rmse_max"], " ✓" if rmse_ok else " ✗",
        )


def _check_classification_thresholds(df: pd.DataFrame) -> None:
    """Log a PASS/FAIL table for classification acceptance criteria."""
    logger.info("\n  ── Classification Acceptance Criteria ──")
    for _, row in df.iterrows():
        name = row["Model"]
        if name not in CLS_THRESHOLDS:
            continue
        t = CLS_THRESHOLDS[name]
        acc_ok = row["Accuracy"] >= t["acc_min"]
        f1_ok = row["F1"] >= t["f1_min"]
        auc_ok = (t["auc_min"] is None) or (row["ROC-AUC"] >= t["auc_min"])
        status = "✅ PASS" if (acc_ok and f1_ok and auc_ok) else "❌ FAIL"
        logger.info(
            "  %s  %-22s | Acc=%.3f%s | F1=%.3f%s | AUC=%.3f%s",
            status, name,
            row["Accuracy"], " ✓" if acc_ok else " ✗",
            row["F1"], " ✓" if f1_ok else " ✗",
            row["ROC-AUC"], " ✓" if auc_ok else " ✗",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_all(
    splits,
    train_result,
    output_dir: str | Path = "outputs",
    no_plots: bool = False,
) -> EvalResult:
    """Evaluate all trained models and generate comparison artefacts.

    Parameters
    ----------
    splits : PreprocessResult
        From ``src.preprocess.build_splits()``.
    train_result : TrainResult
        From ``src.train.train_all()``.
    output_dir : str or Path
        Root output directory; plots go to ``{output_dir}/plots/``,
        CSVs go to ``{output_dir}/results/``.
    no_plots : bool
        If True, skip all plot generation.

    Returns
    -------
    EvalResult
        Named tuple with ``reg_df`` and ``cls_df`` DataFrames.
    """
    logger.info("=== Module 5 — Evaluation ===")
    plots_dir = Path(output_dir) / "plots"
    results_dir = Path(output_dir) / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Regression metrics ---
    logger.info("Computing regression metrics on test set…")
    reg_rows = []
    for name, model in train_result.reg_models.items():
        metrics = _regression_metrics(
            model, splits.X_test, splits.y_reg_test,
            cv_r2=train_result.reg_cv[name],
        )
        row = {"Model": name, **metrics}
        reg_rows.append(row)
        logger.info(
            "  %-22s | MAE=%.3f | RMSE=%.3f | R²=%.3f | CV R²=%.3f",
            name, metrics["MAE"], metrics["RMSE"], metrics["R²"], metrics["CV R²"],
        )
    reg_df = pd.DataFrame(reg_rows)

    # --- Classification metrics ---
    logger.info("Computing classification metrics on test set…")
    cls_rows = []
    for name, model in train_result.cls_models.items():
        metrics = _classification_metrics(
            model, splits.X_test, splits.y_cls_test,
            cv_acc=train_result.cls_cv[name],
        )
        row = {"Model": name, **metrics}
        cls_rows.append(row)
        logger.info(
            "  %-22s | Acc=%.3f | P=%.3f | R=%.3f | F1=%.3f | AUC=%.3f | CV Acc=%.3f",
            name, metrics["Accuracy"], metrics["Precision"], metrics["Recall"],
            metrics["F1"], metrics["ROC-AUC"], metrics["CV Accuracy"],
        )
    cls_df = pd.DataFrame(cls_rows)

    # --- Acceptance criterion check ---
    _check_regression_thresholds(reg_df)
    _check_classification_thresholds(cls_df)

    # --- CSV export ---
    reg_df["Mode"] = "Regression"
    cls_df["Mode"] = "Classification"
    combined = pd.concat([reg_df, cls_df], ignore_index=True)
    csv_path = results_dir / "metrics_comparison.csv"
    combined.to_csv(csv_path, index=False, float_format="%.4f")
    logger.info("metrics_comparison.csv saved → %s", csv_path)

    # Restore Mode-less DFs for return
    reg_df = reg_df.drop(columns=["Mode"])
    cls_df = cls_df.drop(columns=["Mode"])

    # --- Plots ---
    if not no_plots:
        logger.info("Generating comparison plots…")
        _bar_chart(reg_df, "R²",   "Regression — R² Score",           plots_dir / "reg_r2.png")
        _bar_chart(reg_df, "RMSE", "Regression — RMSE (lower=better)", plots_dir / "reg_rmse.png", higher_is_better=False)
        _bar_chart(reg_df, "MAE",  "Regression — MAE (lower=better)",  plots_dir / "reg_mae.png",  higher_is_better=False)
        _overview_chart(reg_df, ["R²", "CV R²"],      "Regression Overview",       plots_dir / "reg_overview.png")
        _bar_chart(cls_df, "Accuracy", "Classification — Accuracy",    plots_dir / "cls_accuracy.png")
        _bar_chart(cls_df, "F1",       "Classification — F1 Score",    plots_dir / "cls_f1.png")
        _bar_chart(cls_df, "ROC-AUC",  "Classification — ROC-AUC",     plots_dir / "cls_roc_auc.png")
        _overview_chart(cls_df, ["Accuracy", "F1", "ROC-AUC"], "Classification Overview", plots_dir / "cls_overview.png")

    logger.info("=== Evaluation Complete ===")
    return EvalResult(reg_df=reg_df, cls_df=cls_df)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Module 5 — Evaluate models and generate comparison plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default="data/heart_disease_raw.csv")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    import sys
    from src.ingest import load_data
    from src.preprocess import build_splits
    from src.train import train_all

    args = _parse_args()
    raw_df = load_data(output_path=args.data)
    splits = build_splits(raw_df)
    train_result = train_all(splits, tune=args.tune)
    report = evaluate_all(splits, train_result, output_dir=args.output, no_plots=args.no_plots)

    print(f"\n{'='*70}")
    print("  REGRESSION METRICS (test set)")
    print(f"{'='*70}")
    print(report.reg_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n{'='*70}")
    print("  CLASSIFICATION METRICS (test set)")
    print(f"{'='*70}")
    print(report.cls_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    sys.exit(0)
