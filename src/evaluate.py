"""
evaluate.py
-----------
The shared toolbox every model is judged with - the same metrics, tests and
plots applied to all four so the comparison is fair. Everything saves into
``outputs/`` so the figures and tables can go straight into the report.

What's in here:
  * the usual binary-classification metrics, plus the Brier score, which is a
    calibration measure (when the model says 0.7, does that group really turn
    out to be havens about 70% of the time?);
  * DeLong's test - a way of asking whether one model's AUC really beats
    another's or whether the gap could just be luck on this test set;
  * an "audit budget" view (precision@k / top-decile capture). A real tax
    authority can only dig into the top few percent of cases, so what matters
    is how many actual havens land in that top slice - not overall accuracy;
  * the ROC / PR / confusion / calibration / training-curve plots.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve

from config import FIGURE_DIR, METRIC_DIR


def classification_metrics(y_true, y_prob, threshold=0.5) -> dict:
    """The standard scorecard for one model: AUC, PR-AUC, Brier, and the
    threshold-based numbers (F1/precision/recall) at the 0.5 cut-off.

    AUC is how well the score ranks a real haven above a non-haven - 0.5 is a
    coin-flip, 1.0 is perfect. PR-AUC is the version that cares more about the
    rare positives, which is what we have here. Brier is the calibration check.
    """
    # Turn probabilities into yes/no at the chosen cut-off for the F1 etc.
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, np.clip(y_prob, 0, 1))),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def precision_at_k(y_true, y_prob, fractions=(0.05, 0.10, 0.20)) -> dict:
    """The audit-budget view: if you only ever look at the top 5/10/20% of
    cases by score, how clean is that pile (precision) and how much of the total
    haven population did you manage to catch (recall)? This is the realistic
    question - nobody audits everything, so what matters is the very top of the list.
    """
    # Sort everything from highest score to lowest, then walk down the slices.
    order = np.argsort(y_prob)[::-1]
    y_sorted = np.asarray(y_true)[order]
    n, total_pos = len(y_true), float(np.sum(y_true))
    out = {}
    for f in fractions:
        k = max(1, int(round(f * n)))
        top = y_sorted[:k]
        out[f"top_{int(f*100)}pct"] = {
            "precision": float(top.mean()),
            "recall": float(top.sum() / total_pos) if total_pos else 0.0,
            "n": k,
        }
    return out


# --------------------------------------------------------------------------- #
# DeLong test for two correlated ROC AUCs.
# In plain terms: when one model's AUC looks higher than another's, this checks
# whether that gap is real or could just be chance. The two helpers below
# (_compute_midrank and _fast_delong) are the standard machinery for it - I'm
# treating them as a known recipe and not reinventing the maths.
# --------------------------------------------------------------------------- #
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(predictions_sorted_transposed[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, prob_a, prob_b) -> dict:
    """Run the DeLong test on two models scored on the same data and hand back
    both AUCs plus a p-value. A small p-value (say below 0.05) means the gap
    between them is unlikely to be down to chance.
    """
    y_true = np.asarray(y_true)
    order = (-y_true).argsort(kind="mergesort")
    label_1_count = int(y_true.sum())
    preds = np.vstack((np.asarray(prob_a), np.asarray(prob_b)))[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z, p = 0.0, 1.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
            "z": float(z), "p_value": float(p)}


# --------------------------------------------------------------------------- #
# I/O + plots
# --------------------------------------------------------------------------- #
def save_metrics(metrics: dict, name: str) -> None:
    """Dump one model's metrics dict to outputs/metrics/<name>.json."""
    path = METRIC_DIR / f"{name}.json"
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[eval] metrics -> {path}")


def plot_roc_curves(curves, filename="roc_curves.png") -> None:
    plt.figure(figsize=(6, 5))
    for name, (y, prob) in curves.items():
        fpr, tpr, _ = roc_curve(y, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, prob):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves — profit-shifting risk classifiers")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=150); plt.close()
    print(f"[eval] figure -> {FIGURE_DIR / filename}")


def plot_pr_curves(curves, filename="pr_curves.png") -> None:
    plt.figure(figsize=(6, 5))
    for name, (y, prob) in curves.items():
        prec, rec, _ = precision_recall_curve(y, prob)
        plt.plot(rec, prec,
                 label=f"{name} (PR-AUC={average_precision_score(y, prob):.3f})")
    base = float(np.mean(list(curves.values())[0][0]))
    plt.axhline(base, color="k", ls="--", alpha=0.4,
                label=f"Base rate ({base:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–recall curves"); plt.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(FIGURE_DIR / filename, dpi=150); plt.close()
    print(f"[eval] figure -> {FIGURE_DIR / filename}")


def plot_confusion(y_true, y_prob, name, threshold=0.5) -> None:
    cm = confusion_matrix(y_true, (y_prob >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-haven", "Haven"])
    ax.set_yticklabels(["Non-haven", "Haven"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {name}")
    fig.colorbar(im, fraction=0.046, pad=0.04); plt.tight_layout()
    out = FIGURE_DIR / f"confusion_{name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"[eval] figure -> {out}")


def plot_calibration(curves, filename="calibration.png") -> None:
    """Reliability plot: bucket the predictions and check whether, say, the
    "around 0.7" bucket really comes out roughly 70% havens. A model sitting on
    the diagonal is well calibrated; well above or below it is over/under-confident.
    """
    plt.figure(figsize=(6, 5))
    for name, (y, prob) in curves.items():
        # Quantile bins so each point is backed by a similar number of cases.
        frac_pos, mean_pred = calibration_curve(y, np.clip(prob, 0, 1),
                                                n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration curves"); plt.legend(loc="upper left")
    plt.tight_layout(); plt.savefig(FIGURE_DIR / filename, dpi=150); plt.close()
    print(f"[eval] figure -> {FIGURE_DIR / filename}")


def plot_training_curve(train_loss, val_loss, best_epoch, name="MLP",
                        filename=None) -> None:
    filename = filename or f"{name.lower()}_training_curve.png"
    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label="Train loss")
    plt.plot(val_loss, label="Validation loss")
    plt.axvline(best_epoch, color="grey", ls="--", alpha=0.7,
                label=f"Best epoch ({best_epoch})")
    plt.xlabel("Epoch"); plt.ylabel("BCE loss")
    plt.title(f"{name} training and validation loss"); plt.legend()
    plt.tight_layout(); plt.savefig(FIGURE_DIR / filename, dpi=150); plt.close()
    print(f"[eval] figure -> {FIGURE_DIR / filename}")
