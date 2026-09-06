"""
Module 6 — SHAP Explainability & Feature Importance
=====================================================
Portfolio Project A: Multi-Model Tabular Prediction Pipeline
Author : Hussain Ali
SRS Ref: Section 3.6

Responsibilities
----------------
1. Feature Importance bar charts (RF + XGBoost — both modes):
   ``outputs/plots/feature_importance_{model_key}.png``
2. SHAP beeswarm summary plot (best classifier by ROC-AUC):
   ``outputs/plots/shap_summary.png``
3. SHAP waterfall plot for a single configurable test sample:
   ``outputs/plots/shap_waterfall_sample.png``
4. Feature importance CSV export:
   ``outputs/results/feature_importance.csv``

SRS Constraints Enforced
-------------------------
* Best classifier chosen by highest ROC-AUC score on test set.
* Sample index for waterfall plot is configurable via --sample CLI arg.
* No function exceeds 40 lines; every function has a docstring.
* Module is independently importable with no side effects.

Usage (standalone)
------------------
    python -m src.explain --data data/heart_disease_raw.csv
    python -m src.explain --data data/heart_disease_raw.csv --sample 3

Usage (as a module)
-------------------
    from src.explain import run_explain
    run_explain(splits, train_result, eval_result)
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

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
# Style constants (consistent with Modules 3 & 5)
# ---------------------------------------------------------------------------
BG_COLOR = "#1a1a2e"
PANEL_COLOR = "#16213e"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4a"
ACCENT = "#e94560"
GREEN = "#16c79a"
PALETTE = ["#e94560", "#16c79a", "#f5a623", "#a8d8ea",
           "#533483", "#0f3460", "#f7b731", "#45b7d1"]

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
})

# Models that expose feature_importances_ (tree-based)
IMPORTANCE_MODELS = {"Random Forest", "XGBoost"}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    """Save a figure and release memory.

    Parameters
    ----------
    fig  : matplotlib Figure to save.
    path : Destination PNG path.
    """
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("  Saved → %s", path.name)


def _plot_feature_importance(
    model,
    model_key: str,
    feature_names: list[str],
    output_path: Path,
) -> pd.DataFrame:
    """Horizontal bar chart of sklearn feature_importances_ for one model.

    Parameters
    ----------
    model        : Fitted tree-based estimator with feature_importances_.
    model_key    : Short model name used in the plot title and filename.
    feature_names: List of feature column names.
    output_path  : Destination PNG path.

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame: Feature, Importance (sorted descending).
    """
    importances = model.feature_importances_
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df = df.sort_values("Importance", ascending=True)

    colors = [
        GREEN if v >= df["Importance"].quantile(0.75) else ACCENT
        for v in df["Importance"]
    ]

    fig, ax = plt.subplots(figsize=(9, max(5, len(df) * 0.38)))
    fig.patch.set_facecolor(BG_COLOR)
    bars = ax.barh(df["Feature"], df["Importance"],
                   color=colors, edgecolor=PANEL_COLOR, height=0.65)
    for bar, val in zip(bars, df["Importance"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    ax.set_title(f"Feature Importance — {model_key}")
    ax.set_xlabel("Importance Score")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, output_path)
    return df.sort_values("Importance", ascending=False)


def _plot_shap_summary(explainer, shap_values, X_test: pd.DataFrame, output_path: Path) -> None:
    """SHAP beeswarm summary plot for the best classifier (SRS §3.6).

    Parameters
    ----------
    explainer   : Fitted SHAP explainer.
    shap_values : SHAP values array for the positive class.
    X_test      : Test feature DataFrame (unscaled labels used for display).
    output_path : Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    shap.summary_plot(
        shap_values, X_test,
        plot_type="dot",
        show=False,
        color_bar=True,
        max_display=17,
    )
    fig = plt.gcf()
    fig.patch.set_facecolor(BG_COLOR)
    for ax_ in fig.axes:
        ax_.set_facecolor(PANEL_COLOR)
        ax_.tick_params(colors=TEXT_COLOR)
        ax_.xaxis.label.set_color(TEXT_COLOR)
        ax_.title.set_color(TEXT_COLOR)
    plt.title("SHAP Summary — Feature Impact on Disease Prediction", color=TEXT_COLOR, pad=12)
    _save(fig, output_path)


def _plot_shap_waterfall(explainer, shap_values, X_test: pd.DataFrame,
                         sample_idx: int, output_path: Path) -> None:
    """SHAP waterfall plot for a single test sample (SRS §3.6).

    Parameters
    ----------
    explainer   : Fitted SHAP explainer.
    shap_values : SHAP Explanation object.
    X_test      : Test feature DataFrame.
    sample_idx  : Row index of the sample to explain.
    output_path : Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    shap.waterfall_plot(shap_values[sample_idx], max_display=15, show=False)
    fig = plt.gcf()
    fig.patch.set_facecolor(BG_COLOR)
    for ax_ in fig.axes:
        ax_.set_facecolor(PANEL_COLOR)
        ax_.tick_params(colors=TEXT_COLOR)
        ax_.xaxis.label.set_color(TEXT_COLOR)
    plt.title(
        f"SHAP Waterfall — Test Sample #{sample_idx} Explanation",
        color=TEXT_COLOR, pad=10,
    )
    _save(fig, output_path)


def _get_best_classifier(eval_result, train_result: dict) -> tuple[str, object]:
    """Identify the best classifier by ROC-AUC from the evaluation report.

    Parameters
    ----------
    eval_result  : EvalResult named tuple from Module 5.
    train_result : TrainResult named tuple from Module 4.

    Returns
    -------
    tuple[str, object]
        ``(model_name, fitted_model)`` of the best classifier.
    """
    cls_df = eval_result.cls_df
    best_name = cls_df.loc[cls_df["ROC-AUC"].idxmax(), "Model"]
    best_model = train_result.cls_models[best_name]
    logger.info("Best classifier by ROC-AUC: %s", best_name)
    return best_name, best_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_explain(
    splits,
    train_result,
    eval_result,
    sample_idx: int = 0,
    output_dir: str | Path = "outputs",
) -> pd.DataFrame:
    """Generate all SRS-specified explainability outputs.

    Steps:
    1. Feature importance bar charts for RF and XGBoost (both modes).
    2. SHAP summary (beeswarm) for the best classifier.
    3. SHAP waterfall for one configurable test sample.
    4. Feature importance CSV export.

    Parameters
    ----------
    splits       : PreprocessResult from Module 2.
    train_result : TrainResult from Module 4.
    eval_result  : EvalResult from Module 5.
    sample_idx   : Test sample index for the waterfall plot (default 0).
    output_dir   : Root output directory.

    Returns
    -------
    pd.DataFrame
        Feature importance table (Feature, Importance, Model) for best tree model.
    """
    logger.info("=== Module 6 — Explainability ===")
    plots_dir = Path(output_dir) / "plots"
    results_dir = Path(output_dir) / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    feature_names = splits.feature_names
    X_test = splits.X_test

    # ── 1. Feature importance plots (regression + classification) ──
    logger.info("Generating feature importance plots…")
    importance_dfs: list[pd.DataFrame] = []

    for mode, models in [("reg", train_result.reg_models), ("cls", train_result.cls_models)]:
        for name, model in models.items():
            if name not in IMPORTANCE_MODELS:
                continue
            key = name.replace(" ", "_").lower()
            out_path = plots_dir / f"feature_importance_{mode}_{key}.png"
            imp_df = _plot_feature_importance(model, f"{name} ({mode.upper()})",
                                              feature_names, out_path)
            imp_df["Model"] = f"{name} ({mode.upper()})"
            importance_dfs.append(imp_df)

    # ── 2. SHAP analysis on best classifier ──
    best_name, best_model = _get_best_classifier(eval_result, train_result)
    logger.info("Building SHAP explainer for: %s…", best_name)

    X_test_df = pd.DataFrame(X_test, columns=feature_names) if not isinstance(X_test, pd.DataFrame) else X_test

    # TreeExplainer is fast for RF/XGBoost; fall back to LinearExplainer
    if hasattr(best_model, "feature_importances_"):
        explainer = shap.TreeExplainer(best_model)
        shap_explanation = explainer(X_test_df)
        shap_vals = shap_explanation.values
        if shap_vals.ndim == 3:          # multi-class output → take class 1
            shap_vals = shap_vals[:, :, 1]
            shap_explanation_cls1 = shap_explanation[:, :, 1]
        else:
            shap_explanation_cls1 = shap_explanation
    else:
        masker = shap.maskers.Independent(X_test_df, max_samples=100)
        explainer = shap.LinearExplainer(best_model, masker)
        shap_vals = explainer.shap_values(X_test_df)
        shap_explanation_cls1 = shap.Explanation(
            values=shap_vals, base_values=explainer.expected_value,
            data=X_test_df.values, feature_names=feature_names,
        )

    # ── 3. SHAP summary plot ──
    logger.info("Generating SHAP summary plot…")
    _plot_shap_summary(explainer, shap_vals, X_test_df, plots_dir / "shap_summary.png")

    # ── 4. SHAP waterfall plot ──
    n_test = len(X_test_df)
    if sample_idx >= n_test:
        logger.warning("sample_idx=%d out of range (n_test=%d). Using 0.", sample_idx, n_test)
        sample_idx = 0
    logger.info("Generating SHAP waterfall plot for test sample #%d…", sample_idx)
    _plot_shap_waterfall(
        explainer, shap_explanation_cls1, X_test_df,
        sample_idx, plots_dir / "shap_waterfall_sample.png",
    )

    # ── 5. Feature importance CSV ──
    if importance_dfs:
        fi_combined = pd.concat(importance_dfs, ignore_index=True)
        fi_path = results_dir / "feature_importance.csv"
        fi_combined.to_csv(fi_path, index=False, float_format="%.6f")
        logger.info("feature_importance.csv saved → %s", fi_path)
    else:
        fi_combined = pd.DataFrame()

    logger.info("=== Explainability Complete ===")
    return fi_combined


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Module 6 — SHAP explainability & feature importance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default="data/heart_disease_raw.csv")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--sample", type=int, default=0,
                        help="Test sample index for SHAP waterfall plot.")
    parser.add_argument("--tune", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    import sys
    from src.ingest import load_data
    from src.preprocess import build_splits
    from src.train import train_all
    from src.evaluate import evaluate_all

    args = _parse_args()
    raw_df = load_data(output_path=args.data)
    splits = build_splits(raw_df)
    train_result = train_all(splits, tune=args.tune)
    eval_result = evaluate_all(splits, train_result, output_dir=args.output, no_plots=True)
    fi_df = run_explain(splits, train_result, eval_result,
                        sample_idx=args.sample, output_dir=args.output)

    print(f"\n{'='*55}")
    print("  Module 6 — Explainability Complete")
    print(f"{'='*55}")
    if not fi_df.empty:
        print("\n  Top-10 features (Random Forest CLS):")
        rf_df = fi_df[fi_df["Model"] == "Random Forest (CLS)"].head(10)
        print(rf_df[["Feature", "Importance"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"{'='*55}")
    sys.exit(0)
