"""
config.py
---------
Everything the rest of my project needs to be told: where the folders are, the random seed,
which columns count as features, and the model settings, etc. One place to change things, so 
I don't have to go through all files when I want to try a different learning rate.

Paths are all worked out relative to this file, so the project runs on any machine without the 
need to change the paths.
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

# The tidy table that oecd_adapter.py builds out of the raw OECD download. If it exists we use
# the real data; if not, we fall back to a synthetic stand-in (see data_generation.py).
REAL_DATA_FILENAME: str = "oecd_cbcr_wide.csv"
SYNTHETIC_DATA_FILENAME: str = "cbcr_synthetic.csv"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
# Fixing the seed means anything random - shuffling, weight initialisation - comes out the same
# every run, so the numbers in my report can actually be reproduced.
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------- #
# Features and label
# --------------------------------------------------------------------------- #
# These are the inputs the model learns from. Every one is built only from what the companies
# themselves report: revenue, profit, tax, headcount, what kinds of offices they run. I leave the
# name of the country out on purpose. That matters, because what we are trying to predict is
# whether the country is a known haven - if the model could see the country name it would simply
# memorise "this one's a haven" rather than learn anything. That kind of cheating is called
# leakage, and it would make the whole evaluation worthless.
FEATURE_COLUMNS: list[str] = [
    # how profitable the place is, and how much tax actually gets paid
    "profit_margin",            # profit before tax / total revenue
    "related_party_share",      # revenue from elsewhere in the same group / total revenue
    "effective_tax_rate",       # tax actually handed over, as a share of profit
    "etr_accrued",              # tax booked as owed, as a share of profit
    # substance checks: does the booked profit line up with real people and real assets?
    "profit_per_employee",      # signed log
    "revenue_per_employee",     # log
    "profit_per_asset",
    "assets_per_employee",      # log
    "capital_per_employee",     # log
    "employees_per_entity",
    "entities_per_group",
    # what the local offices actually do. This is the block I think is novel, because most of the
    # machine-learning work on profit shifting ignores it completely.
    "holding_share",
    "ip_share",
    "igf_share",
    "dormant_share",
    "shifting_activity_share",  # holding + IP + internal finance + dormant: the paper offices
    "real_activity_share",      # manufacturing, sales, services, R&D, purchasing: real work
    # a hint that losses are being parked somewhere deliberately
    "loss_shift_ratio",         # |loss-panel profit| / (profit + |loss|)
]

# What we are predicting: 1 if the partner country is on the tax-haven list, 0 otherwise.
TARGET_COLUMN: str = "tax_haven_binary"

# --------------------------------------------------------------------------- #
# How the data is split for training and testing: by year, "out of time"
# --------------------------------------------------------------------------- #
# Rather than shuffling all the rows and splitting at random, I train on the earlier years and
# test on the latest one. It is closer to how the model would really be used - predicting a
# future year from past ones - and it is a much tougher test. Shuffling lets the same country pair
# from neighbouring years, with nearly identical figures, land in both train and test, which
# flatters the score. The random split below survives only as a comparison for experiments.
TRAIN_YEARS: list[int] = [2016, 2017, 2018, 2019]
VAL_YEARS: list[int] = [2020]
TEST_YEARS: list[int] = [2021]

# Proportions for that random split.
TEST_SIZE: float = 0.20
VAL_SIZE: float = 0.20

# --------------------------------------------------------------------------- #
# Model settings: the dials I can turn for each model
# --------------------------------------------------------------------------- #
# A plain multi-layer neural network, my baseline neural model.
MLP_CONFIG: dict = {
    "hidden_dims": [128, 64, 32],
    "dropout_p": 0.3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "max_epochs": 200,
    "patience": 20,
}

# A transformer adapted for tabular data. The more ambitious model, and the one the project is
# really built around.
FT_TRANSFORMER_CONFIG: dict = {
    "d_token": 32,          # length of the vector each feature value gets turned into
    "n_heads": 4,
    "n_layers": 3,
    "dropout_p": 0.1,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 512,
    "max_epochs": 50,
    "patience": 10,
}

# An autoencoder used the anomaly-detection way: train it on ordinary rows, then flag the ones it
# rebuilds badly. threshold_percentile=95 means the worst-reconstructed 5% count as suspicious.
AE_CONFIG: dict = {
    "encoder_dims": [32, 16, 8],
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 256,
    "max_epochs": 150,
    "patience": 15,
    "threshold_percentile": 95,
}

# XGBoost, the strong conventional model the neural nets have to beat, or at least keep up with.
# These are the combinations the grid search works through.
XGB_PARAM_GRID: dict = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.1],
    "n_estimators": [300, 600],
}

# --------------------------------------------------------------------------- #
# Feature blocks: the three families of inputs, plus the loss signal
# --------------------------------------------------------------------------- #
# Grouping the features this way lets the ablation study in src/ablation.py ask a question SHAP
# cannot answer on its own: not "which feature does the model lean on?" but "how much worse is the
# model if this whole family of inputs is taken away?". The activity-mix block is the one this
# project claims as its contribution, so being able to delete it and measure the damage matters.
FEATURE_BLOCKS: dict[str, list[str]] = {
    "profitability_tax": [
        "profit_margin", "related_party_share", "effective_tax_rate", "etr_accrued",
    ],
    "substance": [
        "profit_per_employee", "revenue_per_employee", "profit_per_asset",
        "assets_per_employee", "capital_per_employee", "employees_per_entity",
        "entities_per_group",
    ],
    "activity_mix": [
        "holding_share", "ip_share", "igf_share", "dormant_share",
        "shifting_activity_share", "real_activity_share",
    ],
    "loss_shifting": ["loss_shift_ratio"],
}

BLOCK_LABELS: dict[str, str] = {
    "profitability_tax": "Profitability and tax",
    "substance": "Economic substance",
    "activity_mix": "Business-activity mix",
    "loss_shifting": "Loss shifting",
}

# --------------------------------------------------------------------------- #
# Capacity-and-regularisation study (the deep-learning workflow the template asks for)
# --------------------------------------------------------------------------- #
# CM3015 Project Idea 2 asks to "improve the chosen test metrics by network scaling up
# and regularisation", following the workflow in Chollet (2018): get a model that can overfit
# first, then fight the overfitting. These are the rungs of that ladder. Stage A grows capacity
# with the regularisation switched off; stage B puts it back on, one mechanism at a time.
MLP_CAPACITY_LADDER: list[dict] = [
    {"tag": "A1 tiny",    "hidden_dims": [16],            "dropout_p": 0.0, "weight_decay": 0.0},
    {"tag": "A2 small",   "hidden_dims": [64, 32],        "dropout_p": 0.0, "weight_decay": 0.0},
    {"tag": "A3 medium",  "hidden_dims": [128, 64, 32],   "dropout_p": 0.0, "weight_decay": 0.0},
    {"tag": "A4 large",   "hidden_dims": [512, 256, 128], "dropout_p": 0.0, "weight_decay": 0.0},
    {"tag": "B1 +dropout",        "hidden_dims": [512, 256, 128], "dropout_p": 0.3, "weight_decay": 0.0},
    {"tag": "B2 +decay",          "hidden_dims": [512, 256, 128], "dropout_p": 0.3, "weight_decay": 1e-4},
    {"tag": "B3 shrink+regular",  "hidden_dims": [128, 64, 32],   "dropout_p": 0.3, "weight_decay": 1e-4},
]

FT_CAPACITY_LADDER: list[dict] = [
    {"tag": "A1 tiny",   "d_token": 8,  "n_layers": 1, "n_heads": 2, "dropout_p": 0.0, "weight_decay": 0.0},
    {"tag": "A2 small",  "d_token": 16, "n_layers": 2, "n_heads": 4, "dropout_p": 0.0, "weight_decay": 0.0},
    {"tag": "A3 medium", "d_token": 32, "n_layers": 3, "n_heads": 4, "dropout_p": 0.0, "weight_decay": 0.0},
    {"tag": "A4 large",  "d_token": 64, "n_layers": 4, "n_heads": 8, "dropout_p": 0.0, "weight_decay": 0.0},
    {"tag": "B1 +dropout",       "d_token": 64, "n_layers": 4, "n_heads": 8, "dropout_p": 0.2, "weight_decay": 0.0},
    {"tag": "B2 +decay",         "d_token": 64, "n_layers": 4, "n_heads": 8, "dropout_p": 0.2, "weight_decay": 1e-4},
    {"tag": "B3 shrink+regular", "d_token": 32, "n_layers": 3, "n_heads": 4, "dropout_p": 0.1, "weight_decay": 1e-5},
]

# Fractions of the training years used for the learning-curve study. The question is whether the
# neural models are losing because there is simply not enough data for them, which is what the
# tabular deep-learning literature predicts at this sample size.
LEARNING_CURVE_FRACTIONS: list[float] = [0.1, 0.25, 0.5, 0.75, 1.0]

# --------------------------------------------------------------------------- #
# Audit capacity: what share of cases a tax auditor can actually look at
# --------------------------------------------------------------------------- #
# The 0.5 cut-off is an arbitrary default that suits none of these models, because the three
# supervised ones are all trained with the rare class weighted up. A tax audit team has a capacity
# instead, so the threshold should be whatever puts that many cases in front of them. 10% is the
# working assumption and src/calibration.py derives the matching cut-off from the validation year.
AUDIT_CAPACITY: float = 0.10

# How many resamples the grouped bootstrap uses when putting a confidence interval around AUC.
BOOTSTRAP_RESAMPLES: int = 1000
