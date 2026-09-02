"""
models/xgb_baseline.py
----------------------
The non-neural baseline: XGBoost, a gradient-boosted decision tree model.

This is here for honesty. On tabular data like this, boosted trees are what most practitioners
reach for first and they are famously hard to beat, so before claiming my neural nets are doing
anything clever I have to check them against a strong conventional model. If the fancy networks
cannot beat XGBoost, that is an important finding in its own right.

Gradient boosting means building lots of small decision trees one after another, each new tree
concentrating on the mistakes the previous ones made, so the errors get whittled down step by step.
To choose the settings I run a small grid search - every combination from a shortlist - judged by
cross-validation, which rotates which slice of the data is held out so the score is not a fluke of
one particular split. The haven imbalance is handled here by scale_pos_weight, XGBoost's version of
the positive-class weight used in the neural models.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from config import RANDOM_SEED, XGB_PARAM_GRID


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    param_grid: dict | None = None,
    seed: int = RANDOM_SEED,
    n_splits: int = 3,
) -> tuple[XGBClassifier, dict]:
    """Search over the settings and return the best model along with the combination that won, which
    is handy to report in the write-up."""
    grid = param_grid or XGB_PARAM_GRID
    # The same imbalance correction as the neural models: scale_pos_weight = neg/pos tells XGBoost
    # to weight the rare haven rows more heavily.
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1.0)

    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )

    # Stratified folds keep the haven and non-haven ratio the same in every split, which matters a
    # great deal when the positive class is rare - otherwise a fold could end up with almost no
    # havens and produce a misleading score. I score on ROC-AUC, since it is threshold-independent
    # and copes well with imbalance.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    search = GridSearchCV(base, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_
