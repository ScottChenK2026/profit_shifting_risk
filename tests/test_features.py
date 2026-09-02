"""Tests for the feature engineering and the train/validation/test splitting - everything that
happens between the raw data and what the models actually see."""

import numpy as np

from config import FEATURE_COLUMNS, TARGET_COLUMN
from data_generation import add_supplementary_columns, generate_synthetic_cbcr
from features import engineer_features, impute_features
from preprocessing import prepare_random_splits


def _feats(seed=11):
    # Shorthand the tests reuse: generate synthetic data and run it through feature engineering.
    return engineer_features(
        add_supplementary_columns(generate_synthetic_cbcr(seed=seed)))


def test_all_feature_columns_present():
    # Every feature the config promises, and the label, has to exist after engineering, or the
    # models fall over downstream.
    feats = _feats()
    for col in FEATURE_COLUMNS:
        assert col in feats.columns
    assert TARGET_COLUMN in feats.columns


def test_activity_shares_bounded():
    # These are shares, so they belong in [0, 1] give or take floating-point wobble. Drifting
    # outside that range would mean the ratio arithmetic is wrong somewhere.
    feats = _feats()
    for col in ("holding_share", "shifting_activity_share", "real_activity_share"):
        v = feats[col].dropna()
        assert v.between(-0.001, 1.001).mean() > 0.99


def test_label_is_binary():
    # The target really is just 0 or 1, with no stray values sneaking in.
    feats = _feats()
    assert set(feats[TARGET_COLUMN].unique()).issubset({0, 1})


def test_imputation_removes_nans():
    # After imputation there should be no missing values left for the models to choke on.
    feats = _feats()
    imputed, medians = impute_features(feats)
    assert not imputed[FEATURE_COLUMNS].isna().any().any()


def test_no_single_feature_is_circular_with_label():
    # The big one: a guard against label leakage. If any single feature were nearly perfectly
    # correlated with the target, the model would just be reading the answer off that column and
    # the whole exercise would be meaningless.
    feats = impute_features(_feats())[0]
    for col in FEATURE_COLUMNS:
        c = np.corrcoef(feats[col], feats[TARGET_COLUMN])[0, 1]
        assert abs(c) < 0.95, f"{col} too correlated with label ({c:.2f})"


def test_split_shapes_and_scaling():
    # Splitting should neither lose nor duplicate rows, and the standardised training set should
    # sit around zero mean - a quick check that the scaler was actually fitted.
    feats = _feats()
    s = prepare_random_splits(feats)
    total = s.X_train.shape[0] + s.X_val.shape[0] + s.X_test.shape[0]
    assert total == len(feats)
    assert s.X_train.shape[1] == len(FEATURE_COLUMNS)
    assert abs(s.X_train.mean()) < 0.1   # train standardised to roughly zero mean
