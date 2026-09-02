"""
models/ft_transformer.py
------------------------
The FT-Transformer - Feature Tokenizer plus Transformer - and the most ambitious model in the
project. It takes the attention idea behind large language models and applies it to plain rows of
numbers instead of text, following Gorishniy et al. (2021), "Revisiting Deep Learning Models for
Tabular Data". The point of including it is to find out whether a genuinely modern deep-learning
architecture beats the simpler MLP and the XGBoost baseline on this kind of data.

The idea, in plain terms:

1. Feature tokeniser. Transformers normally work on a sequence of word tokens. There are no words
   here, so each individual feature value is turned into its own small learned vector, an
   embedding. A row of twenty numbers becomes a sequence of twenty vectors, one token per feature.
2. A special learnable [CLS] token is stuck on the front. Think of it as a blank notepad that the
   model fills in as it reads the row.
3. The Transformer layers then let every token look at every other token, which is self-attention,
   and that is how the model picks up interactions between features - high profit combined with a
   tiny headcount being far more telling than either on its own.
4. Whatever ended up on the [CLS] notepad goes through a small head to produce one final score.

I kept it deliberately tiny - d_token 32, three layers, four heads - so the whole thing still
trains in a reasonable time on a laptop CPU, with no GPU needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import FT_TRANSFORMER_CONFIG, RANDOM_SEED
from models import TrainHistory, set_seed


class NumericalFeatureTokenizer(nn.Module):
    """Turns each single feature value into its own learned vector. Every feature gets its own
    weight and bias, learned during training, so the model can represent "feature 3 = 1.2"
    differently from "feature 7 = 1.2". Those per-feature vectors are what the Transformer layers
    then reason over."""

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        # Small random starting weights - the 0.02 scale - is a common trick. It keeps the initial
        # signals gentle so training starts off stable.
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, F)
        # Broadcast each scalar across its d_token-long vector and apply the per-feature weight and
        # bias, so a batch of F numbers becomes F embedding vectors.
        return x.unsqueeze(-1) * self.weight + self.bias


class FTTransformer(nn.Module):
    """The full model: tokeniser -> [CLS] token -> Transformer stack -> head."""

    def __init__(self, n_features: int, d_token: int = 32, n_heads: int = 4,
                 n_layers: int = 3, dropout_p: float = 0.1):
        super().__init__()
        self.tokenizer = NumericalFeatureTokenizer(n_features, d_token)
        # The [CLS] token is a learnable vector the model uses to summarise the whole row. After
        # attention has done its work, its final state is what we classify on.
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        # One Transformer encoder block. norm_first=True, the pre-norm setup, tends to train more
        # smoothly on small models like this, and GELU is a smoother cousin of ReLU that
        # Transformers usually prefer.
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 2,
            dropout=dropout_p, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        # A small output head that reads the [CLS] summary and emits one score.
        self.head = nn.Sequential(
            nn.LayerNorm(d_token), nn.GELU(), nn.Linear(d_token, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)                              # one token per feature
        cls = self.cls.expand(x.size(0), -1, -1)                # one [CLS] per row in the batch
        seq = torch.cat([cls, tokens], dim=1)                   # [CLS] sits at the front
        z = self.encoder(seq)                                   # attention mixes them all together
        # z[:, 0] picks out the [CLS] position, which is the row's summary, and the head turns it
        # into a single raw score. Same convention as the MLP: the sigmoid happens in the loss
        # function or in predict_proba, never here.
        return self.head(z[:, 0]).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        # Inference path: eval() to disable dropout, sigmoid to turn the raw score into a
        # probability between 0 and 1.
        self.eval()
        return torch.sigmoid(self.forward(x))


def train_ft_transformer(X_train, y_train, X_val, y_val, config=None,
                         seed: int = RANDOM_SEED, verbose: bool = False):
    """Train the FT-Transformer using the same early-stopping recipe as the MLP, and return the
    best model by validation loss along with its loss history."""
    cfg = {**FT_TRANSFORMER_CONFIG, **(config or {})}
    set_seed(seed)
    device = torch.device("cpu")

    model = FTTransformer(n_features=X_train.shape[1], d_token=cfg["d_token"],
                          n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                          dropout_p=cfg["dropout_p"]).to(device)
    # The same imbalance handling as the MLP: weight the rare haven class up by neg/pos so the
    # model cannot win by always predicting "not a haven".
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # AdamW is Adam with cleaner weight decay, and it is the optimiser the Transformer literature
    # recommends, so I follow suit.
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                            weight_decay=cfg["weight_decay"])

    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    hist = TrainHistory()
    # best is the lowest validation loss seen so far; patience is the early-stopping counter.
    best, best_state, patience = float("inf"), None, cfg["patience"]
    for epoch in range(cfg["max_epochs"]):
        # train for one epoch
        model.train()
        run = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            run += loss.item() * len(xb)
        tr = run / len(ds)
        # then measure the loss on held-out data
        model.eval()
        with torch.no_grad():
            vl = criterion(model(Xv), yv).item()
        hist.train_loss.append(tr)
        hist.val_loss.append(vl)
        # A new best means save the weights and reset patience. Otherwise tick the counter down and
        # stop once it reaches zero - the same early-stopping idea as the MLP, written more
        # compactly.
        if vl < best - 1e-5:
            best, best_state, hist.best_epoch = vl, {
                k: v.clone() for k, v in model.state_dict().items()}, epoch
            patience = cfg["patience"]
        else:
            patience -= 1
            if patience <= 0:
                break
        if verbose and epoch % 5 == 0:
            print(f"  FT epoch {epoch:3d}  train={tr:.4f}  val={vl:.4f}")
    # restore the best snapshot, so an over-trained model is never returned
    if best_state:
        model.load_state_dict(best_state)
    return model, hist
