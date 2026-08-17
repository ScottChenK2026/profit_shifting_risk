"""
preprocessing.py
----------------
Takes the engineered features and gets them ready to feed a model: splits the
data into train/validation/test, fills any gaps, and rescales the numbers.

By default it splits by time: train on the earliest years, tune on the next
year, and test on the most recent one (the year lists live in config.py). I do
it this way on purpose - predicting a later year from earlier ones is both
more realistic and a tougher test than randomly shuffling everything, and it
rules out a sneaky form of cheating where almost-identical rows from the same
year land in both the training and test sets. There's also a plain random
split here that I use only for comparison experiments.

One detail that matters for fairness: the fill-in values and the rescaling are
worked out using the training data only, then applied unchanged to the
validation and test sets. If the test set helped decide those, it would
effectively be peeking at itself and the scores would be too good to believe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_COLUMNS, RANDOM_SEED, TARGET_COLUMN, TEST_SIZE, VAL_SIZE,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
)
from features import impute_features


# A tidy bundle holding everything a training run needs: the three feature sets
# and their labels, plus the fitted scaler and fill-in medians (kept so the same
# transforms can be reapplied later), the feature names, the test rows in their
# original table form, and a note of which kind of split this was.
@dataclass
class DataSplits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    medians: pd.Series
    feature_names: list[str]
    test_frame: pd.DataFrame      # engineered+imputed rows of the test set
    split_kind: str


def _assemble(train_df, val_df, test_df, kind) -> DataSplits:
    # Work out the fill-in medians on the training set, then reuse them on val
    # and test - so the held-out data never influences how its own gaps are
    # filled.
    train_df, medians = impute_features(train_df)
    val_df, _ = impute_features(val_df, medians)
    test_df, _ = impute_features(test_df, medians)

    # Rescale every feature to a common scale (mean 0, spread 1) so no single
    # feature dominates just because its raw numbers happen to be larger. Same
    # rule as above: fit the scaler on train only, then apply it to the rest.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLUMNS].values)
    X_val = scaler.transform(val_df[FEATURE_COLUMNS].values)
    X_test = scaler.transform(test_df[FEATURE_COLUMNS].values)

    return DataSplits(
        X_train=X_train.astype(np.float32), X_val=X_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=train_df[TARGET_COLUMN].values.astype(np.float32),
        y_val=val_df[TARGET_COLUMN].values.astype(np.float32),
        y_test=test_df[TARGET_COLUMN].values.astype(np.float32),
        scaler=scaler, medians=medians, feature_names=list(FEATURE_COLUMNS),
        test_frame=test_df.reset_index(drop=True), split_kind=kind,
    )


def prepare_temporal_splits(feats: pd.DataFrame) -> DataSplits:
    """The main split: carve the data up by year (earliest years to train,
    later ones to tune and test)."""
    df = feats.reset_index(drop=True)
    train_df = df[df["year"].isin(TRAIN_YEARS)].copy()
    val_df = df[df["year"].isin(VAL_YEARS)].copy()
    test_df = df[df["year"].isin(TEST_YEARS)].copy()
    if len(val_df) == 0 or len(test_df) == 0:
        # If the data doesn't actually cover those years (can happen with
        # synthetic or partial data), there's nothing to test on - so quietly
        # fall back to the random split instead of crashing.
        return prepare_random_splits(feats)
    return _assemble(train_df, val_df, test_df, "temporal")


def prepare_random_splits(feats: pd.DataFrame, seed: int = RANDOM_SEED
                          ) -> DataSplits:
    """The comparison split: shuffle everything and slice off test and
    validation chunks at random. "stratify" keeps the haven/non-haven balance
    roughly the same in each chunk, which matters because havens are the rare
    case and we don't want a slice that accidentally has almost none."""
    df = feats.reset_index(drop=True)
    y = df[TARGET_COLUMN].values
    idx = np.arange(len(df))
    idx_tv, idx_te = train_test_split(idx, test_size=TEST_SIZE, stratify=y,
                                      random_state=seed)
    # Take the validation slice out of what's *left* after removing test, so the
    # final proportions come out as intended rather than off by a bit.
    rel_val = VAL_SIZE / (1 - TEST_SIZE)
    idx_tr, idx_va = train_test_split(idx_tv, test_size=rel_val,
                                      stratify=y[idx_tv], random_state=seed)
    return _assemble(df.iloc[idx_tr].copy(), df.iloc[idx_va].copy(),
                     df.iloc[idx_te].copy(), "random")


if __name__ == "__main__":
    from data_generation import load_cbcr
    from features import engineer_features
    s = prepare_temporal_splits(engineer_features(load_cbcr()))
    print("kind:", s.split_kind)
    print("train/val/test:", s.X_train.shape, s.X_val.shape, s.X_test.shape)
    print("positive rates:", round(s.y_train.mean(), 3),
          round(s.y_val.mean(), 3), round(s.y_test.mean(), 3))
