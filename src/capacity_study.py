"""
capacity_study.py
-----------------
The deep-learning workflow, run properly and written down.

The CM3015 template brief this project adapts asks the student to "improve the chosen test metrics
by network scaling up and regularisation", and to follow the workflow set out in Chollet (2018).
That workflow has a specific shape, and it is easy to nod at without actually doing it:

  1. get a model that can beat a trivial baseline at all;
  2. keep adding capacity until it clearly overfits, because a model that cannot overfit is too
     small to have found the ceiling of the problem;
  3. then add regularisation back, one mechanism at a time, and tune.

The interim report went straight to step 3 with a single fixed architecture, so it could report
that the transformer lost without being able to say why. The obvious rival explanations - it was
too small, it was over-regularised, it needed longer - were all left open, and the marker asked for
exactly that analysis.

Walking the ladder closes them off. The gap between training and validation loss at each rung says
whether a model is capacity-limited or overfitting, and where the test metric stops responding to
extra capacity says whether more of it would ever have helped. The answer might well be that the
neural models still lose, but then it is a measured conclusion rather than an assumed one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import (FT_CAPACITY_LADDER, MLP_CAPACITY_LADDER, RANDOM_SEED)
from model_runner import run_model


def _overfit_gap(history) -> dict:
    """How far apart training and validation loss ended up at the best epoch.

    This is the diagnostic the whole ladder turns on. A gap near zero with both losses still high
    means the model is too small to fit the problem - underfitting, so add capacity. A wide gap
    means it is memorising the training years - overfitting, so regularise. Reading it at the best
    epoch rather than the last one avoids rewarding a model for how long it kept training after it
    stopped improving.
    """
    if not history.train_loss:
        return {"train_loss": np.nan, "val_loss": np.nan, "gap": np.nan}
    best = min(history.best_epoch, len(history.train_loss) - 1)
    tr, va = history.train_loss[best], history.val_loss[best]
    return {"train_loss": float(tr), "val_loss": float(va), "gap": float(va - tr)}


def run_capacity_ladder(splits, model_name: str, ladder: list[dict],
                        seed: int = RANDOM_SEED, verbose: bool = True) -> pd.DataFrame:
    """Train one architecture at every rung of its ladder and record what happened.

    Each row carries the parameter count, the training and validation loss at the best epoch, the
    gap between them, how many epochs it ran before early stopping, and the test-year AUC and
    PR-AUC. The test metrics are reported because that is what the brief asks the study to improve;
    the choice of which rung to carry forward is made on validation loss, so the test year is never
    used to pick anything.
    """
    rows = []
    for rung in ladder:
        cfg = {k: v for k, v in rung.items() if k != "tag"}
        run = run_model(model_name, splits.X_train, splits.y_train, splits.X_val,
                        splits.y_val, splits.X_test, config=cfg, seed=seed)
        hist = run.extra["history"]
        rec = {
            "stage": rung["tag"],
            "model": model_name,
            "parameters": run.n_parameters,
            "epochs": run.epochs_trained,
            "best_epoch": run.best_epoch,
            **_overfit_gap(hist),
            "auc_roc": float(roc_auc_score(splits.y_test, run.p_test)),
            "pr_auc": float(average_precision_score(splits.y_test, run.p_test)),
            "config": {k: v for k, v in cfg.items()},
        }
        rows.append(rec)
        if verbose:
            print(f"  [{model_name}] {rung['tag']:<20} params={run.n_parameters:>8,}  "
                  f"train={rec['train_loss']:.3f} val={rec['val_loss']:.3f} "
                  f"gap={rec['gap']:+.3f}  AUC={rec['auc_roc']:.3f}")
    return pd.DataFrame(rows)


def run_full_study(splits, seed: int = RANDOM_SEED, verbose: bool = True
                   ) -> dict[str, pd.DataFrame]:
    """Both ladders, the MLP's and the transformer's, run under identical conditions.

    Doing both is what makes the result interpretable. If only the transformer were scaled, a flat
    curve could mean anything. Running the same protocol on a plain feed-forward network of matched
    capacity separates "attention does not help here" from "no neural model helps here", and those
    two conclusions carry very different weight.
    """
    if verbose:
        print("\n--- Capacity and regularisation ladder: MLP ---")
    mlp = run_capacity_ladder(splits, "mlp", MLP_CAPACITY_LADDER, seed, verbose)
    if verbose:
        print("\n--- Capacity and regularisation ladder: FT-Transformer ---")
    ft = run_capacity_ladder(splits, "ft_transformer", FT_CAPACITY_LADDER, seed, verbose)
    return {"mlp": mlp, "ft_transformer": ft}


def summarise_ladder(table: pd.DataFrame) -> dict:
    """Pull out the three readings the write-up needs from one ladder.

    ``best_val_stage`` is the rung a practitioner would actually ship, chosen on validation loss.
    ``widest_gap_stage`` is where overfitting peaked, which is the point step 2 of the workflow is
    looking for. ``auc_range`` is how much the test metric moved across the whole ladder, and if
    that is small then no amount of scaling or regularising was ever going to change the verdict.
    """
    best_val = table.loc[table["val_loss"].idxmin()]
    widest = table.loc[table["gap"].idxmax()]
    return {
        "best_val_stage": str(best_val["stage"]),
        "best_val_auc": float(best_val["auc_roc"]),
        "best_val_loss": float(best_val["val_loss"]),
        "widest_gap_stage": str(widest["stage"]),
        "widest_gap": float(widest["gap"]),
        "auc_min": float(table["auc_roc"].min()),
        "auc_max": float(table["auc_roc"].max()),
        "auc_range": float(table["auc_roc"].max() - table["auc_roc"].min()),
        "params_min": int(table["parameters"].min()),
        "params_max": int(table["parameters"].max()),
    }
