"""
models/autoencoder.py
---------------------
An autoencoder used for anomaly detection -- a different angle on the problem
that doesn't rely on the labels at all.

An autoencoder is a network that learns to squeeze its input down through a
narrow "bottleneck" and then rebuild it again on the other side. If it can
reconstruct a row well, the row looks like the kind of thing it was trained on;
if reconstruction is poor, the row is unusual.

Here's the trick I'm using: I train it on *only* the normal, non-haven rows.
So it gets really good at reproducing ordinary jurisdiction-pairs but never
learns what havens look like. When I then show it a haven row, it tends to
rebuild it badly, and that reconstruction error becomes an anomaly score --
high error means "this row doesn't fit the normal pattern, flag it".

Why bother, when I already have a supervised MLP? Two reasons. It gives a
second, independent opinion. And because it never touches the labels during
training, it sidesteps a big weakness of this project: my haven labels are only
a rough proxy, so a method that doesn't depend on them is reassuring.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import AE_CONFIG, RANDOM_SEED


class ProfitShiftingAutoencoder(nn.Module):
    """Encoder that funnels down to a small bottleneck, decoder that mirrors it.

    The encoder shrinks the input to a compact code; the decoder expands it
    back to the original size. Forcing everything through that narrow middle is
    what stops the net from just copying the input straight through -- it has to
    learn the genuine structure of "normal" rows to rebuild them.
    """

    def __init__(self, input_dim: int,
                 encoder_dims: list[int] | None = None) -> None:
        super().__init__()
        encoder_dims = encoder_dims or AE_CONFIG["encoder_dims"]

        # Build the encoder: Linear -> ReLU repeated, getting narrower each step.
        enc: list[nn.Module] = []
        prev = input_dim
        for h in encoder_dims:
            enc += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        # Drop the final ReLU so the bottleneck code can take any value
        # (positive or negative), rather than being clipped at zero.
        self.encoder = nn.Sequential(*enc[:-1])

        # Decoder mirrors the encoder back up to the original width.
        dec: list[nn.Module] = []
        rev = list(reversed(encoder_dims[:-1])) + [input_dim]
        prev = encoder_dims[-1]
        for h in rev:
            dec += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        # Drop the final ReLU here too -- the output should be able to match any
        # value in the (scaled) input, not just non-negative ones.
        self.decoder = nn.Sequential(*dec[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze down, then rebuild.
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        # The anomaly score itself: how far the rebuilt row is from the
        # original, averaged across features (mean squared error per row).
        # Small = looks normal; large = looks anomalous.
        self.eval()
        x_hat = self.forward(x)
        return torch.mean((x - x_hat) ** 2, dim=1)


def train_autoencoder(
    X_train_neg: np.ndarray,
    X_val_neg: np.ndarray,
    config: dict | None = None,
    seed: int = RANDOM_SEED,
    verbose: bool = False,
) -> tuple[ProfitShiftingAutoencoder, dict]:
    """Train the autoencoder on normal (non-haven) rows only.

    Note both inputs are the negative class -- this is deliberate and is the
    whole point of the approach (see the module docstring). Hands back the
    trained model plus a little dict of metadata: which epoch was best and the
    error threshold above which a row counts as anomalous.
    """
    cfg = {**AE_CONFIG, **(config or {})}
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")

    model = ProfitShiftingAutoencoder(input_dim=X_train_neg.shape[1],
                                      encoder_dims=cfg["encoder_dims"]).to(device)
    # MSE loss because the task is reconstruction -- we're literally measuring
    # how close the rebuilt row is to the input, not doing classification here.
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
    patience_left = cfg["patience"]   # early-stopping counter, as in the other models

    for epoch in range(cfg["max_epochs"]):
        model.train()
        for (xb,) in loader:
            xb = xb.to(device)
            optimiser.zero_grad()
            # Target is the input itself -- the net is trying to reproduce xb.
            loss = criterion(model(xb), xb)
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            # Validation loss is reconstruction error on held-out normal rows.
            val_loss = criterion(model(Xv), Xv).item()

        # Early stopping: keep the weights from whichever epoch reconstructed
        # the validation rows best, and bail once it stops improving.
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

    # Now set the cut-off for calling something an anomaly. I look at the
    # reconstruction errors on the normal training rows and take a high
    # percentile of them (e.g. the 95th). The logic: even normal rows have some
    # error, so the threshold sits just above where most normal rows land --
    # anything reconstructing worse than that gets flagged as suspicious.
    with torch.no_grad():
        train_err = model.reconstruction_error(
            torch.from_numpy(X_train_neg).to(device)).cpu().numpy()
    threshold = float(np.percentile(train_err, cfg["threshold_percentile"]))

    return model, {"best_epoch": best_epoch, "threshold": threshold,
                   "best_val_mse": best_val}
