"""
calibration.py
--------------
Two jobs that both come down to the same complaint: a raw probability out of one of these models is
not directly usable.

The first is calibration. A score is calibrated if, among the rows it scores around 0.7, about 70%
really are havens. The evaluation in the preliminary report showed the two neural models are not:
they sit well above the diagonal in the middle of the range, so they overstate risk exactly where a
reviewer would be making a judgement call. That could mean two different things, and the difference
matters. Either the models have genuinely learned a worse ordering of the data, or they have
learned a reasonable ordering and then squashed it onto the wrong numbers. Fitting a correction on
the validation year and re-measuring separates the two. If a one-parameter correction closes most
of the gap, the ordering was fine all along and the Brier gap was mostly a presentation problem.

The second is the threshold. Every model here is trained with the rare class weighted up, which
pushes predicted probabilities higher across the board, so applying the textbook 0.5 cut-off to all
four compares them at different points on their own curves. It flatters whichever model happens to
be least confident. A reviewing team does not have a probability cut-off anyway - it has a capacity,
some number of cases it can actually work through. So the honest thing is to fix the capacity, take
the threshold that fills it on the validation year, and apply that to the test year.

Both corrections are fitted on the 2020 validation year and applied unchanged to 2021. Nothing here
ever looks at the test labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from config import AUDIT_CAPACITY


# --------------------------------------------------------------------------- #
# Post-hoc calibration
# --------------------------------------------------------------------------- #
def fit_platt(p_val: np.ndarray, y_val: np.ndarray):
    """Platt scaling: fit a one-variable logistic regression mapping the model's score onto the
    observed frequency. It has a single slope and intercept, so it can stretch or shift the scores
    but cannot reorder them - which is precisely the property wanted here. If Platt scaling fixes
    the Brier score, the ranking was never the problem.

    The fit is done on the log-odds rather than the raw probability, which is the standard form and
    keeps the correction well behaved near 0 and 1.
    """
    eps = 1e-6
    z = np.log(np.clip(p_val, eps, 1 - eps) / (1 - np.clip(p_val, eps, 1 - eps)))
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(z.reshape(-1, 1), y_val.astype(int))

    def transform(p: np.ndarray) -> np.ndarray:
        zz = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
        return lr.predict_proba(zz.reshape(-1, 1))[:, 1]

    return transform


def fit_isotonic(p_val: np.ndarray, y_val: np.ndarray):
    """Isotonic regression: a free-form monotone staircase rather than a single curve. It is more
    flexible than Platt scaling and so can fix a wider range of distortions, but with only a few
    hundred positives in the validation year it can also fit noise. Reporting both makes the
    difference visible rather than requiring me to pick one and hope."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_val, y_val.astype(int))
    return lambda p: iso.predict(p)


def calibration_report(p_val, y_val, p_test, y_test) -> dict:
    """Fit both corrections on validation, apply to test, and report the Brier score before and
    after. AUC is reported alongside as a control: both corrections are monotone, so AUC must come
    out unchanged. If it moves, something is wrong with the implementation rather than interesting
    about the model."""
    from sklearn.metrics import roc_auc_score

    out = {
        "brier_raw": float(brier_score_loss(y_test, np.clip(p_test, 0, 1))),
        "auc_raw": float(roc_auc_score(y_test, p_test)),
    }
    for name, fitter in (("platt", fit_platt), ("isotonic", fit_isotonic)):
        try:
            transform = fitter(np.asarray(p_val), np.asarray(y_val))
            p_cal = np.clip(transform(np.asarray(p_test)), 0, 1)
            out[f"brier_{name}"] = float(brier_score_loss(y_test, p_cal))
            out[f"auc_{name}"] = float(roc_auc_score(y_test, p_cal))
        except Exception as exc:                       # pragma: no cover - defensive
            out[f"brier_{name}"] = None
            out[f"{name}_error"] = repr(exc)
    return out


# --------------------------------------------------------------------------- #
# Operating threshold from a stated review capacity
# --------------------------------------------------------------------------- #
def threshold_for_capacity(p_val: np.ndarray, capacity: float = AUDIT_CAPACITY) -> float:
    """The cut-off that sends the top ``capacity`` share of validation-year cases for review.

    Taking the quantile of the validation scores rather than of the test scores is the point: at
    deployment nobody knows next year's score distribution, so the threshold has to be set on
    history and then lived with. That also means the share of test cases flagged will not land on
    exactly 10%, and how far it drifts is itself worth reporting - it is a small, honest measure of
    how stable the model's scores are from one year to the next.
    """
    if not 0 < capacity < 1:
        raise ValueError("capacity must be a fraction strictly between 0 and 1")
    return float(np.quantile(np.asarray(p_val), 1.0 - capacity))


def operating_point(y_test, p_test, threshold: float) -> dict:
    """Precision, recall, F1 and the confusion matrix at a given cut-off, plus what share of the
    test year that cut-off actually flags."""
    from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                                 recall_score)

    y_pred = (np.asarray(p_test) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "flagged_share": float(y_pred.mean()),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def capacity_comparison(y_test, probs: dict, val_probs: dict,
                        capacity: float = AUDIT_CAPACITY) -> dict:
    """Put every model at the same review capacity and report what each one delivers there.

    This is the comparison the preliminary report should have made. At 0.5 the models are being
    judged at whatever point their own weighting happens to put them, and the MLP's apparent
    appetite for false alarms is partly an artefact of that. Held to the same capacity, the queue
    is the same length for everyone and the only question left is how much of it is worth reading.
    """
    out = {}
    for name, p_test in probs.items():
        if name not in val_probs:
            continue
        thr = threshold_for_capacity(val_probs[name], capacity)
        out[name] = operating_point(y_test, p_test, thr)
    return out
