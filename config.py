"""
config.py
---------
One place for all the settings the rest of the project reads from - folder
paths, the random seed, which columns count as features, the model settings,
and so on. The idea is that if I want to tweak something I change it here once
rather than hunting through every file.

Everything is worked out relative to where this file lives, so the project
should just run on any machine without me pasting in absolute paths.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
FIGURE_DIR: Path = OUTPUT_DIR / "figures"
MODEL_DIR: Path = OUTPUT_DIR / "models"
METRIC_DIR: Path = OUTPUT_DIR / "metrics"

for _d in (DATA_DIR, FIGURE_DIR, MODEL_DIR, METRIC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# The tidy, ready-to-use table that oecd_adapter.py builds out of the raw OECD
# download. If this file exists we use the real data; if not we fall back to a
# synthetic stand-in (see data_generation.py).
REAL_DATA_FILENAME: str = "oecd_cbcr_wide.csv"
SYNTHETIC_DATA_FILENAME: str = "cbcr_synthetic.csv"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
# Fixing the seed means anything random (data shuffling, model init, etc.) comes
# out the same every run, so results are repeatable rather than shifting around.
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------- #
# Features and label
# --------------------------------------------------------------------------- #
# These are the inputs the model actually learns from. Every one is built only
# from what the companies themselves report (revenue, profit, tax, headcount,
# what kinds of offices they run, etc.). I deliberately leave out the name of
# the country itself. That matters: the thing we're trying to predict is whether
# the country is a known tax haven, so if the model could see the country name
# it would just memorise "this one's a haven" instead of learning the patterns -
# that kind of cheating is called leakage, and it would make the results
# meaningless.
FEATURE_COLUMNS: list[str] = [
    # how profitable, and how much tax actually gets paid
    "profit_margin",            # profit before tax / total revenue
    "related_party_share",      # revenue from other parts of the same group / total revenue
    "effective_tax_rate",       # tax actually paid as a share of profit (cash basis)
    "etr_accrued",              # tax booked (owed) as a share of profit
    # "substance" checks: does the booked profit line up with real people/assets?
    "profit_per_employee",      # signed log
    "revenue_per_employee",     # log
    "profit_per_asset",
    "assets_per_employee",      # log
    "capital_per_employee",     # log
    "employees_per_entity",
    "entities_per_group",
    # the mix of what the local offices actually do - this is the part I think
    # is novel, since most ML work on this ignores it
    "holding_share",
    "ip_share",
    "igf_share",
    "dormant_share",
    "shifting_activity_share",  # holding + ip + igf + dormant (the "paper office" types)
    "real_activity_share",      # manufacturing + sales + services + R&D + purchasing (real work)
    # a hint that losses are being parked somewhere on purpose
    "loss_shift_ratio",         # |loss-panel profit| / (profit + |loss|)
]

# The thing we're predicting: 1 if the partner country is on the tax-haven list,
# 0 otherwise.
TARGET_COLUMN: str = "tax_haven_binary"

# --------------------------------------------------------------------------- #
# How we split the data for training vs testing: by year ("out-of-time")
# --------------------------------------------------------------------------- #
# Rather than shuffling all the rows and splitting at random, I train on the
# earlier years and test on the latest one. This is closer to how the model
# would really be used (predicting a future year from past ones) and it's a
# tougher, more honest test - shuffling can let near-identical rows from the
# same year end up in both train and test, which flatters the score. The random
# split below is kept only as a comparison for experiments.
TRAIN_YEARS: list[int] = [2016, 2017, 2018, 2019]
VAL_YEARS: list[int] = [2020]
TEST_YEARS: list[int] = [2021]

# Proportions for the random split - only used in the comparison experiments.
TEST_SIZE: float = 0.20
VAL_SIZE: float = 0.20

# --------------------------------------------------------------------------- #
# Model settings (the dials I can turn for each model)
# --------------------------------------------------------------------------- #
# A plain multi-layer neural network - my baseline neural model.
MLP_CONFIG: dict = {
    "hidden_dims": [128, 64, 32],
    "dropout_p": 0.3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "max_epochs": 200,
    "patience": 20,
}

# A transformer adapted for tabular data - the fancier model I'm comparing against.
FT_TRANSFORMER_CONFIG: dict = {
    "d_token": 32,          # size of the vector each feature gets turned into
    "n_heads": 4,
    "n_layers": 3,
    "dropout_p": 0.1,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 512,
    "max_epochs": 50,
    "patience": 10,
}

# An autoencoder, used the anomaly-detection way: train it on "normal" rows and
# flag the ones it reconstructs badly. threshold_percentile=95 means we treat
# the worst-reconstructed 5% as the suspicious ones.
AE_CONFIG: dict = {
    "encoder_dims": [32, 16, 8],
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 256,
    "max_epochs": 150,
    "patience": 15,
    "threshold_percentile": 95,
}

# XGBoost (gradient-boosted trees) - the strong classic-ML model I want the
# neural nets to beat (or at least keep up with). These are the combinations of
# settings I search over to find the best one.
XGB_PARAM_GRID: dict = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.1],
    "n_estimators": [300, 600],
}
