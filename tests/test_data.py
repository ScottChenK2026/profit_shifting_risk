"""Tests for the data layer: mostly sanity checks that the synthetic generator produces plausible,
well-shaped data and that the haven label behaves itself."""

import numpy as np

from data_generation import (
    CBCR_COLUMNS, add_supplementary_columns, generate_synthetic_cbcr,
)
from reference_data import ACTIVITY_CODES


def test_schema_has_financials_and_activities():
    # The generated table should carry exactly the columns we expect - every activity type and the
    # core financials - so that nothing downstream gets a surprise.
    df = generate_synthetic_cbcr(seed=1)
    assert list(df.columns) == CBCR_COLUMNS
    for name in ACTIVITY_CODES.values():
        assert name in df.columns
    for col in ("total_revenues", "profit_before_tax", "num_employees"):
        assert col in df.columns


def test_no_negative_revenue_or_employees():
    # A basic realism check: revenue cannot be negative and a jurisdiction cannot have fewer than
    # one employee, so the generator should never emit either.
    df = generate_synthetic_cbcr(seed=2)
    assert (df["total_revenues"] >= 0).all()
    assert (df["num_employees"] >= 1).all()


def test_reproducible_with_seed():
    # Same seed in, same data out. If this ever breaks, none of the results are repeatable.
    a = generate_synthetic_cbcr(seed=7)
    b = generate_synthetic_cbcr(seed=7)
    assert np.allclose(a["profit_before_tax"], b["profit_before_tax"])


def test_supplementary_adds_haven_flag():
    # The supplementary step is what attaches the haven label, and it has to be a clean 0/1 flag.
    df = add_supplementary_columns(generate_synthetic_cbcr(seed=3))
    assert "is_known_haven" in df.columns
    assert df["is_known_haven"].isin([0, 1]).all()


def test_havens_have_higher_holding_share():
    # The synthetic data should at least reflect the real-world pattern the project is trying to
    # detect: havens host more pure holding activity. If even the toy data failed to show that, the
    # whole setup would be questionable.
    df = add_supplementary_columns(generate_synthetic_cbcr(seed=5))
    act = list(ACTIVITY_CODES.values())
    tot = df[act].sum(axis=1).replace(0, np.nan)
    hold = df["holding_equity"] / tot
    assert hold[df["is_known_haven"] == 1].mean() > hold[df["is_known_haven"] == 0].mean()
