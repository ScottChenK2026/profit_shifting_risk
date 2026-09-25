"""
model_runner.py
---------------
One small function that every study in this project calls: hand it arrays and a model name, get
back that model's scores on the validation and test sets.

It exists because the experiments added after the interim report - the ablation, the capacity
ladder, the learning curves, the label-sensitivity runs - all do the same thing to a different
slice of data, and without a shared entry point each of them would grow its own slightly different
copy of the training code. That is how two experiments end up not being comparable for a reason
nobody notices until it is in the report.

Returning validation scores as well as test scores is deliberate. Everything that has to be fitted
after training - the calibration correction, the review threshold, the weights of the hybrid score -
is fitted on the validation year, and the only way to keep the test year genuinely untouched is for
that to be the path of least resistance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from config import RANDOM_SEED

MODEL_NAMES = ("xgboost", "mlp", "ft_transformer", "autoencoder")


@dataclass
class ModelRun:
    """What one trained model leaves behind: its scores, and enough of a record of how it got there
    to put in a table."""
    name: str
    p_val: np.ndarray
    p_test: np.ndarray
    epochs_trained: int = 0
    best_epoch: int = 0
    n_parameters: int = 0
    extra: dict = field(default_factory=dict)


def _count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def run_model(name: str, X_train, y_train, X_val, y_val, X_test,
              config: dict | None = None, seed: int = RANDOM_SEED) -> ModelRun:
    """Train one model and score it.

    ``config`` overrides whatever that model's defaults are in config.py, which is what lets the
    capacity study walk up a ladder of architectures without editing anything global.

    A note on the XGBoost path. In the interim version the baseline was tuned by cross-validation
    inside the training years while the neural models used the 2020 validation year for early
    stopping, so the two were not seeing the same data budget. Here the grid is scored on that same
    validation year, which makes the comparison a fair one and removes an objection to the headline
    result that I could not otherwise answer.
    """
    if name == "xgboost":
        from models.xgb_baseline import train_xgboost
        model, params = train_xgboost(X_train, y_train, X_val=X_val, y_val=y_val,
                                      param_grid=(config or {}).get("param_grid"), seed=seed)
        return ModelRun(
            name=name,
            p_val=model.predict_proba(X_val)[:, 1],
            p_test=model.predict_proba(X_test)[:, 1],
            n_parameters=int(model.get_booster().num_boosted_rounds()),
            extra={"best_params": params, "model": model},
        )

    if name == "mlp":
        from models.mlp import train_mlp
        model, hist = train_mlp(X_train, y_train, X_val, y_val, config=config, seed=seed)
        return ModelRun(
            name=name,
            p_val=model.predict_proba(torch.from_numpy(X_val)).numpy(),
            p_test=model.predict_proba(torch.from_numpy(X_test)).numpy(),
            epochs_trained=len(hist.train_loss),
            best_epoch=hist.best_epoch,
            n_parameters=_count_parameters(model),
            extra={"history": hist, "model": model},
        )

    if name == "ft_transformer":
        from models.ft_transformer import train_ft_transformer
        model, hist = train_ft_transformer(X_train, y_train, X_val, y_val,
                                           config=config, seed=seed)
        return ModelRun(
            name=name,
            p_val=model.predict_proba(torch.from_numpy(X_val)).numpy(),
            p_test=model.predict_proba(torch.from_numpy(X_test)).numpy(),
            epochs_trained=len(hist.train_loss),
            best_epoch=hist.best_epoch,
            n_parameters=_count_parameters(model),
            extra={"history": hist, "model": model},
        )

    if name == "autoencoder":
        from models.autoencoder import train_autoencoder
        # The autoencoder only ever sees ordinary rows, so it is trained on the negatives alone and
        # never touches a positive example during fitting.
        model, meta = train_autoencoder(X_train[y_train == 0], X_val[y_val == 0],
                                        config=config, seed=seed)
        err_val = model.reconstruction_error(torch.from_numpy(X_val)).numpy()
        err_test = model.reconstruction_error(torch.from_numpy(X_test)).numpy()
        # Reconstruction error is not a probability. Both sets are put on a common 0-1 scale using
        # the validation range only, so the test scores are transformed by something fitted before
        # the test year was looked at, and a test row more extreme than anything in validation is
        # allowed to land outside the range rather than being quietly squashed back into it.
        lo, hi = float(err_val.min()), float(err_val.max())
        span = max(hi - lo, 1e-9)
        return ModelRun(
            name=name,
            p_val=(err_val - lo) / span,
            p_test=np.clip((err_test - lo) / span, 0.0, 1.0),
            best_epoch=meta["best_epoch"],
            n_parameters=_count_parameters(model),
            extra={"meta": meta, "model": model,
                   "err_val": err_val, "err_test": err_test},
        )

    raise ValueError(f"unknown model: {name!r}")
