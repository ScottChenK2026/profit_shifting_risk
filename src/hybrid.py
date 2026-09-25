"""
hybrid.py
---------
Does the label-free model have anything to add to the labelled one?

The autoencoder is in this project because the labels are a proxy. It is trained only on ordinary
jurisdictions and never sees the haven flag, so whatever it finds is a property of the data rather
than of my list. On its own it is much the weaker scorer, and the interim evaluation said so. But
weaker is not the same as redundant. A model that flags different rows for different reasons can
still be worth having, and the interesting question is whether the supervised score and the anomaly
score disagree in a useful way or merely in a noisy one.

Two ways of putting them together, deliberately different in how much they assume:

  * a rank average, which needs nothing fitted at all and simply says the two opinions count
    equally. It is the version that would survive having no labels to tune on;
  * a logistic stack fitted on the validation year, which learns how much to trust each score. It
    is the stronger method, at the cost of needing labels for the fitting step.

Both are fitted or defined on the validation year and then applied unchanged to the test year.

I should say plainly what I expect, because committing to it in advance is what makes the answer
worth anything: given how far behind the autoencoder is, a gain looks unlikely, and the honest
outcome is probably a small negative result. Reporting that is the point. The interim report framed
this as something to be tested rather than assumed, and an experiment that only gets written up if
it succeeds is not an experiment.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def _rank01(x: np.ndarray) -> np.ndarray:
    """Turn scores into ranks on a 0-1 scale.

    Ranking rather than averaging the raw numbers is what makes the combination meaningful: the
    supervised output is a probability, the autoencoder output is a rescaled reconstruction error,
    and averaging those directly would be adding two quantities on different scales. Ranks throw
    away the units and keep the ordering, which is all the combination needs.
    """
    r = rankdata(np.asarray(x), method="average")
    return (r - 1) / max(len(r) - 1, 1)


def rank_average(p_supervised: np.ndarray, p_anomaly: np.ndarray,
                 weight: float = 0.5) -> np.ndarray:
    """A weighted average of the two rank-transformed scores. weight=0.5 gives the two equal say."""
    return weight * _rank01(p_supervised) + (1 - weight) * _rank01(p_anomaly)


def fit_stack(p_sup_val, p_ae_val, y_val):
    """Fit a logistic regression on the validation year that learns how to weight the two scores.

    Both inputs are rank-transformed first so neither dominates through scale alone, and the fitted
    coefficients are themselves informative: a coefficient near zero on the anomaly score is the
    stack saying the autoencoder adds nothing once the supervised score is known, which is a cleaner
    statement of redundancy than comparing AUCs.
    """
    Z = np.column_stack([_rank01(p_sup_val), _rank01(p_ae_val)])
    lr = LogisticRegression(max_iter=1000)
    lr.fit(Z, np.asarray(y_val).astype(int))

    def transform(p_sup, p_ae):
        return lr.predict_proba(
            np.column_stack([_rank01(p_sup), _rank01(p_ae)]))[:, 1]

    return transform, {"coef_supervised": float(lr.coef_[0][0]),
                       "coef_anomaly": float(lr.coef_[0][1]),
                       "intercept": float(lr.intercept_[0])}


def evaluate_hybrid(y_test, p_sup_test, p_ae_test, p_sup_val, p_ae_val, y_val,
                    weights=(0.9, 0.75, 0.5)) -> dict:
    """Score the two components and every combination of them on the test year.

    The weight sweep is there so the result cannot be dismissed as a badly chosen mixing ratio: if
    even the best weight fails to beat the supervised score alone, the conclusion is about the
    autoencoder rather than about the arithmetic.
    """
    def sc(p):
        return {"auc_roc": float(roc_auc_score(y_test, p)),
                "pr_auc": float(average_precision_score(y_test, p))}

    out = {"supervised_alone": sc(p_sup_test), "anomaly_alone": sc(p_ae_test)}

    for w in weights:
        out[f"rank_avg_w{w}"] = sc(rank_average(p_sup_test, p_ae_test, weight=w))

    transform, coefs = fit_stack(p_sup_val, p_ae_val, y_val)
    out["logistic_stack"] = {**sc(transform(p_sup_test, p_ae_test)), **coefs}

    # The verdict, worked out here rather than left to whoever reads the table, so the write-up
    # cannot quietly pick the most flattering row.
    best_name = max((k for k in out if k != "anomaly_alone"),
                    key=lambda k: out[k]["auc_roc"])
    out["verdict"] = {
        "best_variant": best_name,
        "auc_gain_over_supervised": float(out[best_name]["auc_roc"]
                                          - out["supervised_alone"]["auc_roc"]),
        "helps": bool(out[best_name]["auc_roc"] > out["supervised_alone"]["auc_roc"] + 1e-4),
    }
    return out
