"""
models/autoencoder.py
---------------------
An autoencoder used for anomaly detection, which is a different angle on the problem: it does not
rely on the labels at all.

An autoencoder is a network that learns to squeeze its input down through a narrow bottleneck and
then rebuild it on the other side. If it reconstructs a row well, that row looks like the kind of
thing it was trained on. If reconstruction is poor, the row is unusual.

The trick I am using is to train it on the ordinary, non-haven rows only. It gets very good at
reproducing typical jurisdiction pairs and never learns what a haven looks like, so when it is
shown a haven row it tends to rebuild it badly. That reconstruction error becomes the anomaly
score: high error means this row does not fit the normal pattern, flag it.

Why bother, when there is already a supervised MLP? Two reasons. It gives a second, independent
opinion. And because it never touches the labels while training, it sidesteps the biggest weakness
of this project - my haven labels are only a rough proxy, so a method that does not depend on them
is reassuring.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import AE_CONFIG, RANDOM_SEED
from models import set_seed


class ProfitShiftingAutoencoder(nn.Module):
    """An encoder that funnels down to a small bottleneck, and a decoder that mirrors it back up.

    The encoder shrinks the input to a compact code and the decoder expands it back to the original
    size. Forcing everything through that narrow middle is what stops the net copying the input
    straight through: it has to learn the genuine structure of a normal row in order to rebuild it.
    """

    def __init__(self, input_dim: int,
                 encoder_dims: list[int] | None = None) -> None:
        super().__init__()
        encoder_dims = encoder_dims or AE_CONFIG["encoder_dims"]

        # The encoder: Linear -> ReLU repeated, getting narrower at each step.
        enc: list[nn.Module] = []
        prev = input_dim
        for h in encoder_dims:
            enc += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        # Drop the final ReLU so the bottleneck code can take any value, positive or negative,
        # rather than being clipped at zero.
        self.encoder = nn.Sequential(*enc[:-1])

        # The decoder mirrors the encoder back up to the original width.
        dec: list[nn.Module] = []
        rev = list(reversed(encoder_dims[:-1])) + [input_dim]
        prev = encoder_dims[-1]
        for h in rev:
            dec += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        # Drop the final ReLU here too: the output has to be able to match any value in the scaled
        # input, not only non-negative ones.
        self.decoder = nn.Sequential(*dec[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze down, then rebuild.
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        # The anomaly score itself: how far the rebuilt row sits from the original, averaged across
        # features. Small means it looks normal, large means it looks anomalous.
        self.eval()
        x_hat = self.forward(x)
        return torch.mean((x - x_hat) ** 2, dim=1)


def select_threshold(train_err: np.ndarray, percentile: float) -> float:
    """The cut-off above which a reconstruction error counts as anomalous, as a percentile of the
    errors on ordinary training rows."""
    return float(np.percentile(train_err, percentile))


def tune_threshold_on_validation(val_err: np.ndarray, y_val: np.ndarray,
                                 grid=range(50, 100)) -> dict:
    """Search percentiles for the cut-off that gives the best F1 on the validation year.

    This needs saying clearly, because it cuts against the reason the autoencoder is in the project
    at all. The model is trained without labels and that is its whole appeal: it cannot inherit the
    assumptions baked into my tax-haven list. Choosing its threshold against validation labels does
    not change how it was trained, but it does mean the deployed detector is no longer entirely
    label-free - it is unsupervised in its scoring and supervised in where it draws the line.

    It is offered because the alternative was worse. Fixing the cut-off at the 95th percentile of
    training error, as the interim version did, produced three positive predictions out of 551 real
    havens and a precision of 1.000 computed on those three cases. That number was meaningless and
    reporting it invited a reader to think the model was near-perfect when its recall was 0.5%.

    So both are reported: the label-free rule for what the method can do with no labels at all, and
    this one for what it can do when a year of labels is available to calibrate it. The gap between
    them is itself the finding.
    """
    best = {"percentile": None, "f1": -1.0, "threshold": None}
    y = np.asarray(y_val).astype(int)
    if y.sum() == 0:                                   # pragma: no cover - defensive
        return best
    for pct in grid:
        thr = float(np.percentile(val_err, pct))
        pred = (val_err >= thr).astype(int)
        tp = float((pred & y).sum())
        if tp == 0:
            continue
        precision = tp / max(pred.sum(), 1)
        recall = tp / y.sum()
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        if f1 > best["f1"]:
            best = {"percentile": float(pct), "f1": float(f1), "threshold": thr,
                    "precision": float(precision), "recall": float(recall)}
    return best


def train_autoencoder(
    X_train_neg: np.ndarray,
    X_val_neg: np.ndarray,
    config: dict | None = None,
    seed: int = RANDOM_SEED,
    verbose: bool = False,
    X_val_all: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> tuple[ProfitShiftingAutoencoder, dict]:
    """Train the autoencoder on normal, non-haven rows only.

    Both inputs being the negative class is deliberate, and is the whole point of the approach - see
    the module docstring. What comes back is the trained model plus a small dict of metadata: which
    epoch was best, and the error thresholds above which a row counts as anomalous.

    ``X_val_all`` and ``y_val`` are the full validation year, positives included. They are used only
    to tune the threshold, never to fit any weight, and only if supplied - leaving them out keeps
    the method strictly label-free.
    """
    cfg = {**AE_CONFIG, **(config or {})}
    set_seed(seed)
    device = torch.device("cpu")

    model = ProfitShiftingAutoencoder(input_dim=X_train_neg.shape[1],
                                      encoder_dims=cfg["encoder_dims"]).to(device)
    # MSE loss, because the task here is reconstruction. We are measuring how close the rebuilt row
    # is to the input, not classifying anything.
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])

    ds = TensorDataset(torch.from_numpy(X_train_neg))
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)
    Xv = torch.from_numpy(X_val_neg).to(device)

    best_val = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    patience_left = cfg["patience"]   # the early-stopping counter, as in the other models

    for epoch in range(cfg["max_epochs"]):
        model.train()
        for (xb,) in loader:
            xb = xb.to(device)
            optimiser.zero_grad()
            # The target is the input itself - the net is trying to reproduce xb.
            loss = criterion(model(xb), xb)
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            # Validation loss is reconstruction error on held-out normal rows.
            val_loss = criterion(model(Xv), Xv).item()

        # Early stopping: keep the weights from whichever epoch rebuilt the validation rows best,
        # and bail out once it stops improving.
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_left = cfg["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

        if verbose and epoch % 10 == 0:
            print(f"  AE epoch {epoch:3d}  val_mse={val_loss:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Now for the cut-off that decides what counts as an anomaly. I look at the reconstruction
    # errors on the normal training rows and take a high percentile of them, the 95th by default.
    # The logic is that even normal rows carry some error, so the threshold should sit just above
    # where most of them land. Anything rebuilt worse than that gets flagged.
    with torch.no_grad():
        train_err = model.reconstruction_error(
            torch.from_numpy(X_train_neg).to(device)).cpu().numpy()
    threshold = select_threshold(train_err, cfg["threshold_percentile"])

    meta = {"best_epoch": best_epoch, "threshold": threshold,
            "threshold_percentile": cfg["threshold_percentile"],
            "best_val_mse": best_val}

    # If the full validation year was supplied, also report the cut-off a year of labels would have
    # chosen. Both go into the metadata so the evaluation can quote the label-free number and the
    # tuned one side by side rather than presenting either as the whole story.
    if X_val_all is not None and y_val is not None:
        with torch.no_grad():
            val_err = model.reconstruction_error(
                torch.from_numpy(X_val_all).to(device)).cpu().numpy()
        meta["tuned"] = tune_threshold_on_validation(val_err, y_val)

    return model, meta
