"""
The model zoo, gathered in one package.

Four approaches that I compare against each other:

  - the MLP, a straightforward neural net classifier, my simple deep baseline;
  - the FT-Transformer, a modern attention-based net, and the main showcase;
  - the autoencoder, an unsupervised anomaly detector that ignores the labels entirely;
  - XGBoost, the strong conventional tree-based baseline the others have to beat.

The two small things all three neural models share live here rather than being copy-pasted into
each file: the seed-fixing helper, and the record of losses per epoch used to draw learning curves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from config import RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Pin the random number generators so a run is repeatable.

    Neural nets start from random weights and shuffle their data randomly, so two runs of the same
    code can give slightly different numbers. Fixing the seed means I - and anyone marking this -
    get the same result every time, which is what makes the comparison between models trustworthy.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainHistory:
    """The loss at each epoch, kept so I can plot learning curves.

    Tracking training against validation loss side by side is the easiest way to see over-fitting
    as it happens: if training loss keeps falling while validation loss turns back up, the model has
    started memorising rather than learning.
    """
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
