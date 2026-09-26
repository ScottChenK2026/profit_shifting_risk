"""
experiment_figures.py
---------------------
The figures for the studies in ``run_experiments.py``.

A comment the interim report earned was that its visualisations were not doing enough work: they
showed a result but left the reader to supply the interpretation. Two things follow from that, and
only one of them is code. In the report, every figure now carries an explicit reading. Here, the
figures are drawn so that the thing being claimed is the thing the eye lands on first - the drop
when a feature block is deleted, the point where a loss curve separates, the flat stretch where
extra capacity stopped buying anything.

A few conventions, applied to all of them so the set reads as one:

1. three models, three fixed colours, assigned once and never reused for anything else, so a
    colour means the same model in every figure;
2. marker shapes differ as well as colours, so the figures survive being printed in grey or read
    by someone who is colour-blind;
3. the grid is there to be measured against, not looked at, so it sits behind everything at low
    contrast;
4. where a number is the point of the figure it is written on the figure, rather than left to be
    estimated off an axis.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURE_DIR

# One colour per model, fixed here and imported everywhere else. The three hues are checked to stay
# distinguishable under the common forms of colour blindness as well as in normal vision.
MODEL_COLOURS = {
    "xgboost": "#2a78d6",
    "mlp": "#eb6834",
    "ft_transformer": "#1baf7a",
}
MODEL_MARKERS = {"xgboost": "o", "mlp": "s", "ft_transformer": "^"}
MODEL_LABELS = {"xgboost": "XGBoost", "mlp": "MLP", "ft_transformer": "FT-Transformer"}

GRID = {"color": "#d8d8d4", "linewidth": 0.6, "alpha": 0.9}
TEXT = "#0b0b0b"
MUTED = "#52514e"


def _style(ax, ylabel: str = "", xlabel: str = "", title: str = "", grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, **GRID)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b9b4")
    ax.tick_params(colors=MUTED, labelsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT, fontsize=10)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, pad=10)


def _save(fig, filename: str) -> None:
    fig.tight_layout()
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[fig] {path}")


# --------------------------------------------------------------------------- #
def plot_ablation(deltas: pd.DataFrame, filename: str = "ablation_blocks.png") -> None:
    """How much test AUC each feature block is worth when it is removed from the full set.

    Drawn as the drop rather than the remaining score because the drop is the claim. A tall bar
    means the model cannot recover that block's information from anything else in the data, which
    is the evidence the contribution argument needs and the thing SHAP could not show.
    """
    blocks = list(dict.fromkeys(deltas["block"]))
    models = [m for m in MODEL_COLOURS if m in set(deltas["model"])]
    x = np.arange(len(blocks))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    labels: list[tuple[float, float]] = []
    for i, model in enumerate(models):
        part = deltas[deltas["model"] == model].set_index("block").reindex(blocks)
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, part["auc_drop"].values, width * 0.9,
                      color=MODEL_COLOURS[model], label=MODEL_LABELS[model],
                      edgecolor="white", linewidth=1.2)
        for b, v in zip(bars, part["auc_drop"].values):
            if np.isfinite(v):
                labels.append((b.get_x() + b.get_width() / 2, v))
                ax.text(b.get_x() + b.get_width() / 2,
                        v + (0.0015 if v >= 0 else -0.0015),
                        f"{v:+.3f}", ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=7.5, color=MUTED)

    ax.axhline(0, color="#8a8a86", linewidth=1)
    # Headroom so the outermost labels are not clipped, and floor space so a negative bar's label
    # does not collide with the category name printed under the axis.
    vals = [v for _, v in labels if np.isfinite(v)]
    if vals:
        lo, hi = min(min(vals), 0.0), max(vals)
        pad = 0.10 * max(hi - lo, 1e-3)
        ax.set_ylim(lo - 2.4 * pad, hi + 2.0 * pad)
    ax.set_xticks(x)
    ax.set_xticklabels(blocks, fontsize=9)
    _style(ax, ylabel="Fall in test AUC when the block is removed",
           title="Cost of deleting each feature block")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _save(fig, filename)


def plot_capacity(ladders: dict[str, pd.DataFrame], filename: str = "capacity_ladder.png") -> None:
    """The scale-up-then-regularise workflow, one column per architecture.

    The top row is the diagnostic: training loss and validation loss at each rung. While they move
    together the model is capacity-limited; once they separate it is memorising the training years,
    which is the signal to stop adding capacity and start regularising. The bottom row is the test
    metric the exercise is meant to improve. The two rows are separate panels rather than two scales
    on one, because a loss and an AUC share no axis and drawing them as if they did invites
    exactly the misreading this figure exists to prevent.
    """
    names = [n for n in ("mlp", "ft_transformer") if n in ladders]
    fig, axes = plt.subplots(2, len(names), figsize=(6.2 * len(names), 7), squeeze=False)

    for col, name in enumerate(names):
        tbl = ladders[name].reset_index(drop=True)
        x = np.arange(len(tbl))
        colour = MODEL_COLOURS[name]

        ax = axes[0][col]
        ax.plot(x, tbl["train_loss"], color=colour, linewidth=2,
                marker=MODEL_MARKERS[name], markersize=7, label="Training loss")
        ax.plot(x, tbl["val_loss"], color=colour, linewidth=2, linestyle="--",
                marker=MODEL_MARKERS[name], markersize=7, markerfacecolor="white",
                label="Validation loss")
        ax.fill_between(x, tbl["train_loss"], tbl["val_loss"], color=colour, alpha=0.12)
        ax.set_xticks(x)
        ax.set_xticklabels(tbl["stage"], rotation=30, ha="right", fontsize=8)
        _style(ax, ylabel="BCE loss at the best epoch",
               title=f"{MODEL_LABELS[name]} — capacity and regularisation")
        ax.legend(frameon=False, fontsize=9)

        ax = axes[1][col]
        bars = ax.bar(x, tbl["auc_roc"], 0.62, color=colour, edgecolor="white",
                      linewidth=1.2)
        for b, v in zip(bars, tbl["auc_roc"]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}",
                    ha="center", fontsize=7.5, color=MUTED)
        lo = float(min(tbl["auc_roc"])) - 0.02
        ax.set_ylim(max(lo, 0.5), float(max(tbl["auc_roc"])) + 0.015)
        ax.set_xticks(x)
        ax.set_xticklabels(tbl["stage"], rotation=30, ha="right", fontsize=8)
        _style(ax, ylabel="Test AUC (2021)",
               title="What the extra capacity bought")

    _save(fig, filename)


def plot_learning_curve(curve: pd.DataFrame,
                        filename: str = "learning_curves.png") -> None:
    """Test AUC against the number of training rows.

    The shape is what matters, not the level. A curve still rising at the right-hand edge says the
    model is data-limited and would improve with more; a curve that has flattened says it has taken
    what the data has to give. Comparing those shapes across the three models is the test of whether
    sample size explains the neural models' deficit.
    """
    fig, ax = plt.subplots(figsize=(7, 4.4))
    models = [m for m in MODEL_COLOURS if m in set(curve["model"])]
    # Where two curves converge their end labels would overlap, so the offsets are staggered by
    # the final value rather than fixed: the lower curve's label goes below, the higher one's above.
    finals = {m: curve[curve["model"] == m].sort_values("n_train")["auc_roc"].iloc[-1]
              for m in models}
    order = sorted(models, key=lambda m: finals[m])
    dy = {m: (-11 + 11 * i) for i, m in enumerate(order)}
    for model in models:
        part = curve[curve["model"] == model].sort_values("n_train")
        ax.plot(part["n_train"], part["auc_roc"], color=MODEL_COLOURS[model],
                linewidth=2, marker=MODEL_MARKERS[model], markersize=8,
                label=MODEL_LABELS[model])
        last = part.iloc[-1]
        ax.annotate(f"{last['auc_roc']:.3f}",
                    (last["n_train"], last["auc_roc"]),
                    textcoords="offset points", xytext=(10, dy[model]),
                    fontsize=8.5, color=MODEL_COLOURS[model], fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(right=float(curve["n_train"].max()) * 1.6)
    _style(ax, xlabel="Training rows (log scale)", ylabel="Test AUC (2021)",
           title="Does more data close the gap?")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _save(fig, filename)


def plot_calibration_effect(cal: dict, filename: str = "calibration_effect.png") -> None:
    """Brier score before and after a correction fitted on the validation year.

    Lower is better. Both corrections are monotone, so neither can change the ordering of the
    predictions; if the bars fall a long way, the models had the ordering right all along and were
    simply reporting it on the wrong scale, which is a far milder criticism than it first appeared.
    """
    models = [m for m in MODEL_COLOURS if m in cal]
    variants = [("brier_raw", "Raw"), ("brier_platt", "Platt"),
                ("brier_isotonic", "Isotonic")]
    x = np.arange(len(models))
    width = 0.8 / len(variants)
    shades = {"brier_raw": 1.0, "brier_platt": 0.62, "brier_isotonic": 0.34}

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (key, label) in enumerate(variants):
        vals = [cal[m].get(key) or np.nan for m in models]
        offset = (i - (len(variants) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      color=[MODEL_COLOURS[m] for m in models],
                      alpha=shades[key], edgecolor="white", linewidth=1.2,
                      label=label)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}",
                        ha="center", fontsize=7.5, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=9)
    _style(ax, ylabel="Brier score on 2021 (lower is better)",
           title="Can a correction fitted on 2020 fix the calibration?")
    ax.legend(frameon=False, fontsize=9, title="Correction",
              title_fontsize=9, loc="upper right")
    _save(fig, filename)


def plot_label_sensitivity(sens: pd.DataFrame,
                           filename: str = "label_sensitivity.png") -> None:
    """Test AUC under each definition of what counts as a haven.

    The middle group is the one the argument rests on. If the score holds when the four contested
    conduits are taken out of the positive class, the result is not an artefact of a particular and
    politically loaded list.
    """
    variants = [v for v in ("baseline", "strict_sinks", "conduits_only")
                if v in set(sens["variant"])]
    pretty = {"baseline": "Baseline list", "strict_sinks": "Strict sinks\n(conduits removed)",
              "conduits_only": "Conduits only"}
    models = [m for m in MODEL_COLOURS if m in set(sens["model"])]
    x = np.arange(len(variants))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for i, model in enumerate(models):
        part = sens[sens["model"] == model].set_index("variant").reindex(variants)
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, part["auc_roc"].values, width * 0.9,
                      color=MODEL_COLOURS[model], label=MODEL_LABELS[model],
                      edgecolor="white", linewidth=1.2)
        for b, v in zip(bars, part["auc_roc"].values):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                        ha="center", fontsize=7.5, color=MUTED)

    ax.axhline(0.5, color="#8a8a86", linewidth=1, linestyle=":")
    ax.text(len(variants) - 0.5, 0.508, "chance", fontsize=8, color=MUTED, ha="right")
    ax.set_ylim(0.45, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[v] for v in variants], fontsize=9)
    _style(ax, ylabel="Test AUC (2021)",
           title="Does the result survive redrawing the haven list?")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _save(fig, filename)


def plot_by_reporter(by_reporter: pd.DataFrame,
                     filename: str = "auc_by_reporter.png") -> None:
    """Per-reporting-jurisdiction AUC for the best model, worst at the top.

    A single overall figure can hide a model that works for the countries with full filings and
    fails for everyone else. Ordering worst-first puts the weak cases where they will be read rather
    than buried under the successes.
    """
    tbl = by_reporter.sort_values("auc_roc").reset_index(drop=True)
    if tbl.empty:                                      # pragma: no cover - defensive
        return
    fig, ax = plt.subplots(figsize=(7, max(3.2, 0.26 * len(tbl) + 1.4)))
    y = np.arange(len(tbl))
    ax.barh(y, tbl["auc_roc"], 0.66, color=MODEL_COLOURS["xgboost"],
            edgecolor="white", linewidth=1.1)
    for yi, v in zip(y, tbl["auc_roc"]):
        ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=7.5, color=MUTED)
    ax.axvline(float(tbl["auc_roc"].median()), color="#8a8a86", linestyle="--",
               linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(tbl["reporting_jurisdiction"], fontsize=8)
    ax.set_xlim(0.5, 1.03)
    _style(ax, xlabel="Test AUC within the reporting jurisdiction",
           title="Where the model works, and where it works less well",
           grid_axis="x")
    _save(fig, filename)
