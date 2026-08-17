"""
models/mlp.py
-------------
A plain feed-forward neural network (a multi-layer perceptron, or MLP) that I
use to decide, for each jurisdiction-pair row, whether it looks like a
profit-shifting risk or not. It's a yes/no (binary) classifier built in PyTorch.

This is the simplest of my neural models, so I treat it as the "does a basic
net even help?" check before reaching for the fancier transformer. The general
recipe follows Chollet (2018): start with enough capacity to learn the task,
then keep the model from just memorising the training data using three things
working together -- dropout, weight decay, and early stopping (all explained
where they appear below). Because tax havens are a minority of the rows, I also
lean on the loss to stop the model from cheating by always guessing "not a
haven".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import MLP_CONFIG, RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Pin the random number generators so a run is repeatable.

    Neural nets start from random weights and shuffle data randomly, so two
    runs can give slightly different numbers. Fixing the seed means I (and
    anyone marking this) get the same result every time, which matters for
    being able to trust the comparisons between models.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


class ProfitShiftingMLP(nn.Module):
    """The network itself: a few fully-connected layers stacked up.

    Each hidden layer is Linear -> ReLU -> Dropout. ReLU is just the standard
    "let positive signals through, zero out the rest" non-linearity that lets
    the net learn curved decision boundaries rather than only straight lines.

    One thing worth flagging: the last layer spits out a raw score (a "logit"),
    not a probability. I don't squash it with a sigmoid here on purpose --
    PyTorch's BCEWithLogitsLoss does the squashing internally in a more
    numerically stable way during training, and predict_proba() does it for
    actual predictions.
    """

    def __init__(self, input_dim: int,
                 hidden_dims: list[int] | None = None,
                 dropout_p: float = 0.3) -> None:
        super().__init__()
        hidden_dims = hidden_dims or MLP_CONFIG["hidden_dims"]
        layers: list[nn.Module] = []
        prev = input_dim
        # Build the hidden stack layer by layer. Dropout randomly switches off a
        # fraction of neurons each training step, which stops the net from
        # leaning too hard on any one feature and helps it generalise.
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p=dropout_p)]
            prev = h
        layers.append(nn.Linear(prev, 1))   # final layer collapses to one score
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        # At prediction time I do want a probability, so here's where the
        # sigmoid goes -- it maps the raw score onto 0..1. eval() turns dropout
        # off so predictions are deterministic, and no_grad() skips the
        # gradient bookkeeping we only need during training.
        self.eval()
        return torch.sigmoid(self.forward(x))


@dataclass
class TrainHistory:
    """Just a record of the loss at each epoch so I can plot learning curves.

    Tracking training vs validation loss side by side is the easiest way to
    see over-fitting happening -- if training loss keeps dropping while
    validation loss turns back up, the model has started memorising.
    """
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0


def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    config: dict | None = None,
    seed: int = RANDOM_SEED,
    verbose: bool = False,
) -> tuple[ProfitShiftingMLP, TrainHistory]:
    """Train the net, keep the best version, and hand it back with its history.

    "Best" here means the point where it did best on the validation set, not
    the very last epoch -- see the early-stopping logic in the loop below.
    """
    cfg = {**MLP_CONFIG, **(config or {})}
    set_seed(seed)

    device = torch.device("cpu")
    model = ProfitShiftingMLP(input_dim=X_train.shape[1],
                              hidden_dims=cfg["hidden_dims"],
                              dropout_p=cfg["dropout_p"]).to(device)

    # Class imbalance fix. Havens are roughly a fifth of the rows, so a lazy
    # model can score well just by always saying "not a haven". Weighting the
    # positive (haven) class by neg/pos tells the loss to care proportionally
    # more about getting those rare rows right, which forces it to actually try.
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Adam is the standard go-to optimiser; weight_decay is the L2 penalty
    # (the "weight decay" regulariser) that nudges weights towards small values
    # so the model stays simpler and over-fits less.
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])

    train_ds = TensorDataset(torch.from_numpy(X_train),
                             torch.from_numpy(y_train))
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    history = TrainHistory()
    best_val = float("inf")
    best_state: dict | None = None
    # "Patience" is the early-stopping counter: how many epochs I'll tolerate
    # with no improvement before giving up. Resets every time we hit a new best.
    patience_left = cfg["patience"]

    for epoch in range(cfg["max_epochs"]):
        # --- one full pass over the training data ---
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()             # clear last step's gradients
            loss = criterion(model(xb), yb)
            loss.backward()                   # work out how to adjust weights
            optimiser.step()                  # take the step
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_ds)           # average loss across the epoch

        # --- now check how we're doing on data the net didn't train on ---
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), yv).item()

        history.train_loss.append(epoch_loss)
        history.val_loss.append(val_loss)

        # Early stopping. If this epoch beat the best validation loss so far
        # (with a tiny margin so noise doesn't count), snapshot the weights and
        # reset patience. Otherwise count down -- and once patience runs out,
        # stop and fall back to the best snapshot. This is what stops the net
        # from training past the point where it's still genuinely improving.
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            patience_left = cfg["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"  early stop at epoch {epoch} "
                          f"(best={history.best_epoch}, val={best_val:.4f})")
                break

        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  train={epoch_loss:.4f}  "
                  f"val={val_loss:.4f}")

    # Roll the model back to its best-performing weights before returning it,
    # so we never hand back an over-trained version from the final epoch.
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
