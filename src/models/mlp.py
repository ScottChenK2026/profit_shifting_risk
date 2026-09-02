"""
models/mlp.py
-------------
A plain feed-forward neural network - a multi-layer perceptron - that decides, for each
jurisdiction-pair row, whether it looks like a profit-shifting risk. A yes/no classifier, built in
PyTorch.

This is the simplest of my neural models, so I treat it as the "does a basic net even help?" check
before reaching for the transformer. The recipe follows Chollet (2018): start with enough capacity
to learn the task, then stop the model memorising the training data using three things working
together - dropout, weight decay and early stopping, each explained where it appears below.
Because havens are a minority of the rows, I also lean on the loss function to stop the model
cheating by always guessing "not a haven".
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import MLP_CONFIG, RANDOM_SEED
from models import TrainHistory, set_seed


class ProfitShiftingMLP(nn.Module):
    """The network itself: a few fully-connected layers stacked up.

    Each hidden layer is Linear -> ReLU -> Dropout. ReLU is the standard "let positive signals
    through, zero out the rest" non-linearity, and it is what lets the net learn curved decision
    boundaries rather than only straight lines.

    One thing worth flagging: the last layer emits a raw score, a logit, rather than a probability.
    I do not squash it with a sigmoid here on purpose. PyTorch's BCEWithLogitsLoss does that
    internally in a more numerically stable way during training, and predict_proba() does it when
    actual predictions are wanted.
    """

    def __init__(self, input_dim: int,
                 hidden_dims: list[int] | None = None,
                 dropout_p: float = 0.3) -> None:
        super().__init__()
        hidden_dims = hidden_dims or MLP_CONFIG["hidden_dims"]
        layers: list[nn.Module] = []
        prev = input_dim
        # Build the hidden stack layer by layer. Dropout switches off a random fraction of neurons
        # at each training step, which stops the net leaning too heavily on any one feature.
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p=dropout_p)]
            prev = h
        layers.append(nn.Linear(prev, 1))   # the final layer collapses to a single score
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        # At prediction time I do want a probability, so this is where the sigmoid goes: it maps
        # the raw score onto 0 to 1. eval() turns dropout off so predictions are deterministic, and
        # no_grad() skips the gradient bookkeeping that is only needed during training.
        self.eval()
        return torch.sigmoid(self.forward(x))


def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    config: dict | None = None,
    seed: int = RANDOM_SEED,
    verbose: bool = False,
) -> tuple[ProfitShiftingMLP, TrainHistory]:
    """Train the net, keep the best version of it, and hand it back with its loss history. "Best"
    means the point where it did best on the validation set, not the final epoch - see the
    early-stopping logic in the loop below."""
    cfg = {**MLP_CONFIG, **(config or {})}
    set_seed(seed)

    device = torch.device("cpu")
    model = ProfitShiftingMLP(input_dim=X_train.shape[1],
                              hidden_dims=cfg["hidden_dims"],
                              dropout_p=cfg["dropout_p"]).to(device)

    # Dealing with class imbalance. Havens are roughly a fifth of the rows, so a lazy model could
    # score well simply by always saying "not a haven". Weighting the positive class by neg/pos
    # tells the loss to care proportionally more about getting those rare rows right.
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Adam is the standard go-to optimiser. weight_decay is the L2 penalty that nudges weights
    # towards small values, keeping the model simpler and less prone to over-fitting.
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
    # Patience is the early-stopping counter: how many epochs without improvement I will tolerate
    # before giving up. It resets every time we hit a new best.
    patience_left = cfg["patience"]

    for epoch in range(cfg["max_epochs"]):
        # --- one full pass over the training data ---
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()             # clear the previous step's gradients
            loss = criterion(model(xb), yb)
            loss.backward()                   # work out how the weights should move
            optimiser.step()                  # take the step
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_ds)           # average loss across the epoch

        # --- then see how it does on data it did not train on ---
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), yv).item()

        history.train_loss.append(epoch_loss)
        history.val_loss.append(val_loss)

        # Early stopping. If this epoch beat the best validation loss so far, by a margin big
        # enough that it is not just noise, snapshot the weights and reset patience. Otherwise
        # count down, and when patience runs out stop and fall back to the best snapshot. This is
        # what keeps the net from training past the point where it is genuinely improving.
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

    # Roll the model back to its best-performing weights before returning it, so an over-trained
    # final epoch never escapes.
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
