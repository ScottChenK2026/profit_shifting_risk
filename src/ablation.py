"""
ablation.py
-----------
The experiment the interim report was missing.

That report claimed the business-activity mix - how many establishments in a jurisdiction do
nothing but hold shares, manage patents, lend within the group or sit dormant - is where the
predictive signal lives, and supported the claim with SHAP. The marker's response was fair: it was
not clear what the contribution actually was, and SHAP does not settle it. SHAP says which inputs a
fitted model leaned on. It does not say what would happen if those inputs were never available,
because a model denied one block of features will reach for whatever else correlates with it. Those
are different questions and only the second one is about the data rather than the model.

So this file deletes feature blocks and retrains. Two directions, and they answer different things:

  * each block alone, which asks how far that block gets you with no help;
  * everything except one block, which asks what that block adds on top of the rest.

The second is the stricter test and the one that matters for the claim. If the activity mix is
merely restating what the substance ratios already say, removing it will barely dent the score and
the contribution is smaller than I argued. If the score falls, the block is carrying information
nothing else in the data supplies.

The learning-curve study at the bottom is here for a different reason: it belongs to the question of
why the transformer loses to the tree, and it needs the same retrain-on-a-subset machinery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import (BLOCK_LABELS, FEATURE_BLOCKS, FEATURE_COLUMNS,
                    LEARNING_CURVE_FRACTIONS, RANDOM_SEED)
from model_runner import run_model


def _column_index(names: list[str]) -> list[int]:
    """Positions of the named features within the full feature matrix."""
    lookup = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
    return [lookup[c] for c in names if c in lookup]


def _score(y_true, p) -> dict:
    return {"auc_roc": float(roc_auc_score(y_true, p)),
            "pr_auc": float(average_precision_score(y_true, p))}


def feature_subsets() -> dict[str, list[str]]:
    """Every feature set the ablation trains on: the full set, each block alone, and each
    leave-one-block-out set. Named so the resulting table reads without a key."""
    subsets: dict[str, list[str]] = {"full": list(FEATURE_COLUMNS)}
    for block, cols in FEATURE_BLOCKS.items():
        subsets[f"only_{block}"] = list(cols)
    for block, cols in FEATURE_BLOCKS.items():
        dropped = set(cols)
        subsets[f"without_{block}"] = [c for c in FEATURE_COLUMNS if c not in dropped]
    return subsets


def run_ablation(splits, models=("xgboost", "mlp", "ft_transformer"),
                 seed: int = RANDOM_SEED, verbose: bool = True) -> pd.DataFrame:
    """Train each model on each feature subset and report test AUC and PR-AUC.

    Everything else is held constant - the same split by year, the same seed, the same architecture
    and training recipe - so the only thing varying between rows is which columns the model was
    allowed to see. The imputation medians and the scaler come from the full-feature fit and are
    simply indexed down to the surviving columns, rather than being refitted per subset, which
    keeps the preprocessing identical across runs.
    """
    subsets = feature_subsets()
    rows = []
    for subset_name, cols in subsets.items():
        idx = _column_index(cols)
        Xtr, Xva, Xte = (splits.X_train[:, idx], splits.X_val[:, idx],
                         splits.X_test[:, idx])
        for model_name in models:
            run = run_model(model_name, Xtr, splits.y_train, Xva, splits.y_val,
                            Xte, seed=seed)
            rec = {"subset": subset_name, "n_features": len(idx),
                   "model": model_name, **_score(splits.y_test, run.p_test)}
            rows.append(rec)
            if verbose:
                print(f"  [ablation] {subset_name:<28} {model_name:<15} "
                      f"AUC={rec['auc_roc']:.3f}  PR-AUC={rec['pr_auc']:.3f}")
    return pd.DataFrame(rows)


def ablation_deltas(table: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw ablation scores into the numbers the argument actually needs: how much each
    block costs when it is removed, and how far it gets on its own.

    A positive ``auc_drop`` means the model got worse without that block, which is the direction
    that supports the block mattering. Reporting both columns together guards against the trap of a
    block that looks strong alone only because it duplicates information held elsewhere."""
    out = []
    for model, part in table.groupby("model"):
        full = part.loc[part["subset"] == "full", "auc_roc"].iloc[0]
        full_pr = part.loc[part["subset"] == "full", "pr_auc"].iloc[0]
        for block in FEATURE_BLOCKS:
            alone = part.loc[part["subset"] == f"only_{block}"]
            without = part.loc[part["subset"] == f"without_{block}"]
            out.append({
                "model": model,
                "block": BLOCK_LABELS.get(block, block),
                "auc_alone": float(alone["auc_roc"].iloc[0]) if len(alone) else np.nan,
                "auc_without": float(without["auc_roc"].iloc[0]) if len(without) else np.nan,
                "auc_drop": float(full - without["auc_roc"].iloc[0]) if len(without) else np.nan,
                "pr_auc_drop": float(full_pr - without["pr_auc"].iloc[0]) if len(without) else np.nan,
                "auc_full": float(full),
            })
    return pd.DataFrame(out)


def run_learning_curve(splits, models=("xgboost", "mlp", "ft_transformer"),
                       fractions=LEARNING_CURVE_FRACTIONS,
                       seed: int = RANDOM_SEED, verbose: bool = True) -> pd.DataFrame:
    """Retrain each model on growing fractions of the training years and score on the same test year.

    This is the evidence for one of the two explanations of why the transformer loses. The tabular
    deep-learning literature says attention models need a lot of rows before they pay for
    themselves, and with around eight thousand training rows this dataset is well below where that
    usually happens. If the explanation is right, the neural curves should still be climbing at full
    data while the tree has flattened off - a gap that would close with more data. If instead all
    three curves have levelled out, sample size is not the reason and the honest conclusion is that
    attention has nothing to add on this problem.

    The subsample is stratified so the share of havens stays constant; otherwise a small draw could
    end up with too few positives and the curve would be measuring class balance, not sample size.
    """
    rng = np.random.default_rng(seed)
    y = splits.y_train
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rows = []
    for frac in fractions:
        take_pos = rng.permutation(pos)[:max(2, int(round(frac * len(pos))))]
        take_neg = rng.permutation(neg)[:max(2, int(round(frac * len(neg))))]
        idx = np.sort(np.concatenate([take_pos, take_neg]))
        for model_name in models:
            run = run_model(model_name, splits.X_train[idx], y[idx], splits.X_val,
                            splits.y_val, splits.X_test, seed=seed)
            rec = {"fraction": float(frac), "n_train": int(len(idx)),
                   "model": model_name, **_score(splits.y_test, run.p_test)}
            rows.append(rec)
            if verbose:
                print(f"  [curve] frac={frac:<5} n={len(idx):<6} {model_name:<15} "
                      f"AUC={rec['auc_roc']:.3f}")
    return pd.DataFrame(rows)
