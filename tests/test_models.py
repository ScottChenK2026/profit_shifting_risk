"""Tests for the model components. Deliberately tiny - toy data and a handful of
epochs - so they run in seconds and just confirm each model wires up and learns
*something*, rather than chasing real performance.
"""

import numpy as np
import torch

from models.autoencoder import train_autoencoder
from models.mlp import ProfitShiftingMLP, train_mlp
from models.ft_transformer import FTTransformer, train_ft_transformer
from evaluate import delong_roc_test, precision_at_k


def _toy(n=600, d=18, seed=0):
    # A simple made-up problem with an obvious linear signal in the first three
    # features - easy enough that any working model should be able to pick it up.
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = ((X[:, 0] + X[:, 1] - X[:, 2]) > 0.5).astype(np.float32)
    return X, y


def test_mlp_forward_and_proba():
    # The MLP should give one output per row and probabilities that stay in [0, 1].
    m = ProfitShiftingMLP(input_dim=18)
    assert m(torch.randn(5, 18)).shape == (5,)
    p = m.predict_proba(torch.randn(10, 18)).numpy()
    assert ((p >= 0) & (p <= 1)).all()


def test_ft_transformer_forward_and_proba():
    # Same wiring check for the transformer - right output shape, valid probabilities.
    m = FTTransformer(n_features=18, d_token=16, n_heads=2, n_layers=1)
    assert m(torch.randn(4, 18)).shape == (4,)
    p = m.predict_proba(torch.randn(6, 18)).numpy()
    assert ((p >= 0) & (p <= 1)).all()


def test_mlp_learns_signal():
    # Sanity check that training actually does something: on the toy data the MLP
    # should clear 70% accuracy. Not a performance bar, just proof it learns.
    X, y = _toy()
    m, h = train_mlp(X[:450], y[:450], X[450:], y[450:],
                     config={"max_epochs": 60, "patience": 10})
    acc = ((m.predict_proba(torch.from_numpy(X[450:])).numpy() >= .5) == y[450:]).mean()
    assert acc > 0.7


def test_ft_transformer_learns_signal():
    # The transformer trains so briefly here that we don't demand accuracy - just
    # that it produces a spread of scores rather than the same number every time.
    X, y = _toy()
    m, h = train_ft_transformer(X[:450], y[:450], X[450:], y[450:],
                                config={"max_epochs": 25, "patience": 8,
                                        "batch_size": 128})
    auc_ok = m.predict_proba(torch.from_numpy(X[450:])).numpy()
    assert auc_ok.std() > 0   # produces varied scores


def test_autoencoder_reconstruction_error():
    # The autoencoder should give back one reconstruction error per row and a
    # learned threshold in its metadata - the pieces the pipeline relies on later.
    X, _ = _toy()
    m, meta = train_autoencoder(X[:450], X[450:],
                                config={"max_epochs": 30, "patience": 8})
    err = m.reconstruction_error(torch.from_numpy(X[450:])).numpy()
    assert err.shape == (150,) and "threshold" in meta


def test_delong_and_precision_at_k():
    # Quick check the two evaluation helpers behave: build a deliberately good
    # score and a junk one, then confirm DeLong hands back a valid p-value and
    # precision_at_k reports the slice we asked for.
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=400)
    good = y + rng.normal(0, 0.5, 400)   # tracks the labels, so a strong ranker
    bad = rng.normal(0, 1, 400)          # pure noise
    res = delong_roc_test(y, good, bad)
    assert "p_value" in res and 0 <= res["p_value"] <= 1
    pk = precision_at_k(y, good)
    assert "top_10pct" in pk
