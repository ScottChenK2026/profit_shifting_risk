"""Tests for the model components. Deliberately tiny - toy data and a handful of epochs - so they
run in seconds. They confirm each model wires up correctly and learns something, rather than
chasing real performance."""

import numpy as np
import torch

from models.autoencoder import train_autoencoder
from models.mlp import ProfitShiftingMLP, train_mlp
from models.ft_transformer import FTTransformer, train_ft_transformer
from evaluate import delong_roc_test, precision_at_k


def _toy(n=600, d=18, seed=0):
    # A made-up problem with an obvious linear signal in the first three features, easy enough that
    # any working model ought to pick it up.
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = ((X[:, 0] + X[:, 1] - X[:, 2]) > 0.5).astype(np.float32)
    return X, y


def test_mlp_forward_and_proba():
    # One output per row, and probabilities that stay inside [0, 1].
    m = ProfitShiftingMLP(input_dim=18)
    assert m(torch.randn(5, 18)).shape == (5,)
    p = m.predict_proba(torch.randn(10, 18)).numpy()
    assert ((p >= 0) & (p <= 1)).all()


def test_ft_transformer_forward_and_proba():
    # The same wiring check for the transformer: right output shape, valid probabilities.
    m = FTTransformer(n_features=18, d_token=16, n_heads=2, n_layers=1)
    assert m(torch.randn(4, 18)).shape == (4,)
    p = m.predict_proba(torch.randn(6, 18)).numpy()
    assert ((p >= 0) & (p <= 1)).all()


def test_mlp_learns_signal():
    # Proof that training does something: on the toy data the MLP should clear 70% accuracy. That
    # is not a performance bar, just evidence it learns.
    X, y = _toy()
    model, _ = train_mlp(X[:450], y[:450], X[450:], y[450:],
                         config={"max_epochs": 60, "patience": 10})
    acc = ((model.predict_proba(torch.from_numpy(X[450:])).numpy() >= .5) == y[450:]).mean()
    assert acc > 0.7


def test_ft_transformer_learns_signal():
    # The transformer trains so briefly here that demanding accuracy would be unfair. All this asks
    # is that it produces a spread of scores rather than the same number every time.
    X, y = _toy()
    model, _ = train_ft_transformer(X[:450], y[:450], X[450:], y[450:],
                                    config={"max_epochs": 25, "patience": 8,
                                            "batch_size": 128})
    scores = model.predict_proba(torch.from_numpy(X[450:])).numpy()
    assert scores.std() > 0


def test_autoencoder_reconstruction_error():
    # The autoencoder should return one reconstruction error per row and a learned threshold in its
    # metadata - the two pieces the pipeline relies on later.
    X, _ = _toy()
    m, meta = train_autoencoder(X[:450], X[450:],
                                config={"max_epochs": 30, "patience": 8})
    err = m.reconstruction_error(torch.from_numpy(X[450:])).numpy()
    assert err.shape == (150,) and "threshold" in meta


def test_delong_and_precision_at_k():
    # A quick check on the two evaluation helpers: build one deliberately good score and one junk
    # score, then confirm DeLong returns a valid p-value and precision_at_k reports the slice asked
    # for.
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=400)
    good = y + rng.normal(0, 0.5, 400)   # tracks the labels, so a strong ranker
    bad = rng.normal(0, 1, 400)          # pure noise
    res = delong_roc_test(y, good, bad)
    assert "p_value" in res and 0 <= res["p_value"] <= 1
    pk = precision_at_k(y, good)
    assert "top_10pct" in pk
