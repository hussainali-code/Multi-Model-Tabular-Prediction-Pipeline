"""
Module 3 — Exploratory Data Analysis
======================================
Portfolio Project A: Multi-Model Tabular Prediction Pipeline
Author : Hussain Ali
SRS Ref: Section 3.3

Responsibilities
----------------
Generates the five SRS-specified EDA plots and saves them as PNG files:

1. ``dist_{feature}.png``    — Histogram + KDE for each continuous feature.
2. ``correlation_heatmap.png`` — Pearson correlation heatmap (all features).
3. ``class_balance.png``     — Bar chart: Class 0 vs Class 1 counts.
4. ``target_distribution.png`` — Histogram of regression target (num, 0–4).
5. ``pairplot_top5.png``     — Seaborn pairplot of top-5 features by |r| with target.

Design Notes
------------
* All plots use a consistent dark-themed style with a curated palette to
  match portfolio aesthetics.
* ``matplotlib.use('Agg')`` is called before any import of pyplot so the
  module runs headlessly in CI / Colab without a display.
* No plot is shown interactively; all are saved and closed immediately.
* No function exceeds 40 lines; every function has a docstring.

Usage (standalone)
------------------
    python -m src.eda --data data/heart_disease_raw.csv --output outputs/plots

Usage (as a module)
-------------------
    from src.eda import run_eda
    run_eda(df, output_dir="outputs/plots")
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
# Style constants
# ---------------------------------------------------------------------------
TARGET_COLUMN = "num"

# Continuous features to plot individual distributions for
CONTINUOUS_FEATURES: list[str] = [
    "age", "trestbps", "chol", "thalach", "oldpeak",
]

# Dark-mode palette
BG_COLOR = "#1a1a2e"
PANEL_COLOR = "#16213e"
ACCENT_COLORS = ["#e94560", "#0f3460", "#533483", "#16c79a", "#f5a623"]
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4a"

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
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    """Save a figure and close it to free memory.

    Parameters
    ----------
    fig : plt.Figure
        The figure to save.
    path : Path
        Destination file path.
    """
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("  Saved → %s", path.name)


def _plot_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot histogram + KDE for each continuous feature (SRS §3.3, Plot 1).

    Saves one PNG per feature: ``dist_{feature}.png``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with original column names.
    output_dir : Path
        Directory to save PNG files.
    """
    logger.info("Generating distribution plots…")
    feature_labels = {
        "age": "Age (years)",
        "trestbps": "Resting Blood Pressure (mm Hg)",
        "chol": "Serum Cholesterol (mg/dl)",
        "thalach": "Max Heart Rate Achieved",
        "oldpeak": "ST Depression (oldpeak)",
    }
    for feat in CONTINUOUS_FEATURES:
        if feat not in df.columns:
            continue
        data = pd.to_numeric(df[feat], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(BG_COLOR)

        sns.histplot(data, kde=True, color=ACCENT_COLORS[0],
                     edgecolor=PANEL_COLOR, ax=ax, bins=20, alpha=0.85)
        ax.set_title(f"Distribution — {feature_labels.get(feat, feat)}")
        ax.set_xlabel(feature_labels.get(feat, feat))
        ax.set_ylabel("Count")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

        fig.tight_layout()
        _save(fig, output_dir / f"dist_{feat}.png")


def _plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot Pearson correlation heatmap for all numeric features (SRS §3.3, Plot 2).

    Saves: ``correlation_heatmap.png``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame.
    output_dir : Path
        Destination directory.
    """
    logger.info("Generating correlation heatmap…")
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG_COLOR)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap=cmap,
        center=0, linewidths=0.5, linecolor=BG_COLOR,
        ax=ax, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8, "color": TEXT_COLOR},
    )
    ax.set_title("Pearson Correlation Matrix — Heart Disease Features")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()
    _save(fig, output_dir / "correlation_heatmap.png")


def _plot_class_balance(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of Class 0 vs Class 1 (binary classification) (SRS §3.3, Plot 3).

    Saves: ``class_balance.png``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame containing the 'num' column.
    output_dir : Path
        Destination directory.
    """
    logger.info("Generating class balance chart…")
    y_cls = (df[TARGET_COLUMN] > 0).astype(int)
    counts = y_cls.value_counts().sort_index()
    labels = ["Class 0\n(No Disease)", "Class 1\n(Disease)"]
    colors = [ACCENT_COLORS[3], ACCENT_COLORS[0]]

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(BG_COLOR)
    bars = ax.bar(labels, counts.values, color=colors, edgecolor=PANEL_COLOR, width=0.5)

    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            str(count), ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_title("Classification Target — Class Balance")
    ax.set_ylabel("Number of Patients")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, output_dir / "class_balance.png")


def _plot_target_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Histogram of regression target (num, 0–4) (SRS §3.3, Plot 4).

    Saves: ``target_distribution.png``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame.
    output_dir : Path
        Destination directory.
    """
    logger.info("Generating target distribution plot…")
    counts = df[TARGET_COLUMN].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(BG_COLOR)
    colors = sns.color_palette("magma", len(counts))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor=PANEL_COLOR)

    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(count), ha="center", va="bottom", fontsize=10,
        )
    ax.set_title("Regression Target Distribution (num: Angiographic Disease Severity)")
    ax.set_xlabel("Disease Severity (0 = None, 4 = Severe)")
    ax.set_ylabel("Number of Patients")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, output_dir / "target_distribution.png")


def _plot_pairplot_top5(df: pd.DataFrame, output_dir: Path) -> None:
    """Seaborn pairplot of the top-5 features by |Pearson r| with target (SRS §3.3, Plot 5).

    Saves: ``pairplot_top5.png``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame.
    output_dir : Path
        Destination directory.
    """
    logger.info("Generating pairplot (top-5 correlated features)…")
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    corr_with_target = (
        numeric_df.corr()[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .abs()
        .sort_values(ascending=False)
    )
    top5 = corr_with_target.head(5).index.tolist()
    logger.info("  Top-5 features by |r| with 'num': %s", top5)

    plot_df = numeric_df[top5 + [TARGET_COLUMN]].dropna()
    # Map target to binary for colour coding
    plot_df = plot_df.copy()
    plot_df["Disease"] = (plot_df[TARGET_COLUMN] > 0).map({0: "No", 1: "Yes"})

    with plt.style.context("dark_background"):
        grid = sns.pairplot(
            plot_df.drop(columns=[TARGET_COLUMN]),
            hue="Disease",
            palette={"No": ACCENT_COLORS[3], "Yes": ACCENT_COLORS[0]},
            diag_kind="kde",
            plot_kws={"alpha": 0.6, "s": 30},
            diag_kws={"fill": True, "alpha": 0.5},
        )
    grid.figure.suptitle(
        "Pairplot — Top 5 Features Correlated with Disease Severity",
        y=1.02, color=TEXT_COLOR, fontsize=12, fontweight="bold",
    )
    grid.figure.set_facecolor(BG_COLOR)
    for ax in grid.axes.flatten():
        if ax:
            ax.set_facecolor(PANEL_COLOR)
    _save(grid.figure, output_dir / "pairplot_top5.png")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_eda(df: pd.DataFrame, output_dir: str | Path = "outputs/plots") -> None:
    """Generate all five SRS-specified EDA plots and save to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Raw validated DataFrame from ``src.ingest.load_data()``.
    output_dir : str or Path, optional
        Directory where PNG files will be written.
        Created automatically if it does not exist.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Module 3 — EDA Plot Generation ===")
    logger.info("Output directory: %s", output_dir.resolve())

    _plot_distributions(df, output_dir)
    _plot_correlation_heatmap(df, output_dir)
    _plot_class_balance(df, output_dir)
    _plot_target_distribution(df, output_dir)
    _plot_pairplot_top5(df, output_dir)

    saved = sorted(output_dir.glob("*.png"))
    logger.info("=== EDA Complete — %d plots saved ===", len(saved))
    for p in saved:
        logger.info("  %s  (%d KB)", p.name, p.stat().st_size // 1024)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Module 3 — Generate EDA plots for the UCI Heart Disease dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", default="data/heart_disease_raw.csv",
        help="Path to the raw CSV produced by Module 1.",
    )
    parser.add_argument(
        "--output", default="outputs/plots",
        help="Directory to save generated PNG plots.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    import sys
    from src.ingest import load_data

    args = _parse_args()
    raw_df = load_data(output_path=args.data)
    run_eda(raw_df, output_dir=args.output)
    print(f"\nAll EDA plots saved to: {Path(args.output).resolve()}")
    sys.exit(0)
