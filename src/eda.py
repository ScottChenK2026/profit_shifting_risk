"""
eda.py
------
The get-to-know-the-data figures, the ones I look at before trusting any model. Three plots: how
each feature is distributed and whether havens look different from non-havens, how the features
correlate with one another, and how lopsided the haven/non-haven split is. Everything lands in
``outputs/figures`` ready for the report.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import FEATURE_COLUMNS, FIGURE_DIR, TARGET_COLUMN


def plot_feature_distributions(feats: pd.DataFrame,
                               filename: str = "feature_distributions.png") -> None:
    """One histogram per feature, havens and non-havens overlaid, so I can see at a glance whether
    a feature actually separates the two. Densities rather than raw counts, because the classes are
    so unbalanced that counts would be unreadable."""
    n = len(FEATURE_COLUMNS)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLUMNS):
        ax = axes[i]
        for label, sub in feats.groupby(TARGET_COLUMN):
            ax.hist(sub[col].dropna(), bins=40, alpha=0.55,
                    label=("Haven" if label == 1 else "Non-haven"), density=True)
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=7)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Engineered feature distributions by class", y=1.02)
    plt.tight_layout()
    path = FIGURE_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eda] figure -> {path}")


def plot_correlation_heatmap(feats: pd.DataFrame,
                             filename: str = "correlation_heatmap.png") -> pd.DataFrame:
    """Heatmap of how the features move together, mainly as a check for badly redundant pairs - if
    two features are near enough the same thing, that is worth knowing. The matrix comes back as
    well, so the pipeline can log the worst off-diagonal pair."""
    corr = feats[FEATURE_COLUMNS].corr()
    # With eighteen features the annotated cells were colliding at the old figure size. Drawing
    # only the lower triangle fixes it - the matrix is symmetric, so the upper half is a repeat -
    # along with a bigger figure and smaller in-cell numbers. Same information, now readable.
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.4, linecolor="white",
                annot_kws={"size": 6.0},
                cbar_kws={"shrink": 0.65, "label": "Pearson r"}, ax=ax)
    ax.set_title("Feature correlation matrix", fontsize=13, pad=12)
    ax.tick_params(axis="x", labelsize=8, rotation=90)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    plt.tight_layout()
    path = FIGURE_DIR / filename
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    print(f"[eda] figure -> {path}")
    return corr


def plot_class_balance(feats: pd.DataFrame,
                       filename: str = "class_balance.png") -> None:
    """Bar chart of how many haven and non-haven rows there are. Havens are the minority, which is
    exactly why the evaluation later leans on PR-AUC and precision@k rather than plain accuracy -
    accuracy is easy to fake when one class is rare."""
    counts = feats[TARGET_COLUMN].value_counts().sort_index()
    plt.figure(figsize=(4, 4))
    plt.bar(["Non-haven", "Haven"], counts.values,
            color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        plt.text(i, v, str(v), ha="center", va="bottom")
    plt.ylabel("Observations")
    plt.title("Class balance")
    plt.tight_layout()
    path = FIGURE_DIR / filename
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[eda] figure -> {path}")
