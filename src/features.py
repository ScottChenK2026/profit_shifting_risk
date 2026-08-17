"""
features.py
-----------
This turns the raw table of numbers into the actual signals the model learns
from. Raw revenue or profit on its own doesn't tell you much - what matters is
the ratios, like "how much profit per employee" - so this file does that maths.
Every signal is built only from what the companies report; I never feed in the
country name or any outside tax rate, for the leakage reasons explained in
config.py.

There are three families of signals, plus one extra:

1. Profitability / tax  - how fat the profit margin is, how much revenue is just
                          money moving within the group, and how much tax is
                          actually paid relative to profit.
2. "Substance" ratios   - profit, revenue, capital per employee, per asset, per
                          entity. The gut-check here: if a place books huge
                          profit but has barely any staff or assets, that's
                          exactly the mismatch we're looking for.
3. Activity mix         - what fraction of the local offices are the "paper
                          office" types (holding, IP, internal finance, dormant)
                          versus places doing real work. This is the part of the
                          OECD data that most machine-learning work on this topic
                          has overlooked, which is a big reason I'm using it.

On top of those, a loss-shifting signal built from the separate profit/loss
sub-group figures - basically, is an unusual amount of loss being parked here?

The answer (label)
------------------
What the model tries to predict is simply whether the partner country is on the
tax-haven list from reference_data.py. That list is decided completely
separately from any of the signals above, so the question is a fair one: can the
reported economics and the office mix, on their own, reveal the known havens?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, TARGET_COLUMN
from reference_data import (
    ACTIVITY_CODES, REAL_ACTIVITIES, SHIFTING_PRONE_ACTIVITIES,
)

ACTIVITY_NAMES = list(ACTIVITY_CODES.values())


# Divide, but turn any divide-by-zero into a missing value instead of letting it
# blow up to infinity. We'll fill those gaps in sensibly later.
def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def _slog(x: pd.Series) -> pd.Series:
    """A log that keeps the sign. Profit can be negative, and a plain log would
    choke on that, but we still want to remember whether the value was a profit
    or a loss - so we log the size and stick the original +/- back on."""
    return np.sign(x) * np.log1p(x.abs())


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Office counts: if a kind of office isn't listed for a country, that almost
    # always means there just aren't any, not that the number is unknown - so we
    # fill the blank with 0, which is the truthful answer here.
    for name in ACTIVITY_NAMES:
        if name not in out.columns:
            out[name] = 0.0
        out[name] = out[name].fillna(0.0)
    act_total = out[ACTIVITY_NAMES].sum(axis=1)

    # --- Profitability / tax ------------------------------------------------ #
    # Margin and related-party share are clipped to sane ranges so a few crazy
    # outliers (often from tiny denominators) don't dominate everything.
    out["profit_margin"] = _safe_div(out["profit_before_tax"], out["total_revenues"]).clip(-5, 5)
    out["related_party_share"] = _safe_div(
        out.get("related_party_revenues", pd.Series(np.nan, index=out.index)),
        out["total_revenues"]).clip(0, 1)

    # Effective tax rate = tax paid divided by profit, i.e. the share of profit
    # that actually leaves as tax. We blank it out when profit is zero or
    # negative, because "tax as a fraction of a loss" is meaningless and would
    # produce nonsense numbers.
    etr = _safe_div(out["income_tax_paid"], out["profit_before_tax"])
    etr[out["profit_before_tax"] <= 0] = np.nan
    out["effective_tax_rate"] = etr.clip(0, 1)

    # Same idea but using tax *owed* (accrued) rather than tax actually handed
    # over - the two can differ, and the gap is itself informative.
    etr_a = _safe_div(out["income_tax_accrued"], out["profit_before_tax"])
    etr_a[out["profit_before_tax"] <= 0] = np.nan
    out["etr_accrued"] = etr_a.clip(0, 1)

    # --- Substance ratios --------------------------------------------------- #
    # The core idea: real business needs real people and real assets. If profit
    # is sky-high per employee or per dollar of assets, the profit probably
    # wasn't earned where it's being booked.
    out["profit_per_employee"] = _slog(
        _safe_div(out["profit_before_tax"], out["num_employees"]))
    out["revenue_per_employee"] = np.log1p(
        _safe_div(out["total_revenues"].clip(lower=0), out["num_employees"]))
    out["profit_per_asset"] = _safe_div(out["profit_before_tax"],
                                        out["tangible_assets"]).clip(-50, 50)
    out["assets_per_employee"] = np.log1p(
        _safe_div(out["tangible_assets"].clip(lower=0), out["num_employees"]))
    out["capital_per_employee"] = _slog(
        _safe_div(out.get("stated_capital",
                          pd.Series(np.nan, index=out.index)),
                  out["num_employees"]))
    out["employees_per_entity"] = _safe_div(
        out["num_employees"],
        out.get("num_entities", pd.Series(np.nan, index=out.index)))
    out["entities_per_group"] = _safe_div(
        out.get("num_entities", pd.Series(np.nan, index=out.index)),
        out.get("num_mne_groups", pd.Series(np.nan, index=out.index)))

    # --- Business-activity mix --------------------------------------------- #
    # Convert raw office counts into shares of all offices in that country, so a
    # big country and a small one are compared on a like-for-like basis. We pull
    # out the individual "paper office" types, then also a combined share of all
    # shifting-prone types vs all real-operations types.
    out["holding_share"] = _safe_div(out["holding_equity"], act_total)
    out["ip_share"] = _safe_div(out["ip_management"], act_total)
    out["igf_share"] = _safe_div(out["internal_group_finance"], act_total)
    out["dormant_share"] = _safe_div(out["dormant"], act_total)
    shifting = out[SHIFTING_PRONE_ACTIVITIES].sum(axis=1)
    real = out[REAL_ACTIVITIES].sum(axis=1)
    out["shifting_activity_share"] = _safe_div(shifting, act_total)
    out["real_activity_share"] = _safe_div(real, act_total)

    # --- Loss-shifting signal ---------------------------------------------- #
    # Of the total profit-or-loss sloshing around here, how much of it is loss?
    # A country that's nearly all losses can be where a group deliberately parks
    # its losses, so this ratio is worth handing to the model.
    pos = out.get("profit_positive_panel", pd.Series(np.nan, index=out.index))
    neg = out.get("profit_negative_panel", pd.Series(np.nan, index=out.index))
    out["loss_shift_ratio"] = _safe_div(neg.abs(), pos.abs() + neg.abs())

    # --- Label -------------------------------------------------------------- #
    # Copy the haven flag into the target column. If for some reason it's not
    # there, default everything to 0 so the pipeline still runs (e.g. when
    # scoring brand-new data we have no labels for).
    if "is_known_haven" in out.columns:
        out[TARGET_COLUMN] = out["is_known_haven"].astype(int)
    else:
        out[TARGET_COLUMN] = 0

    # Hand back just the ID columns, the engineered features, and the label -
    # drop all the intermediate raw stuff we no longer need.
    keep = (["reporting_jurisdiction", "partner_jurisdiction", "year"]
            + FEATURE_COLUMNS + [TARGET_COLUMN])
    keep = [c for c in keep if c in out.columns]
    return out[keep]


def summarise_missing(df: pd.DataFrame) -> pd.DataFrame:
    """A quick sanity-check helper: how many values are missing in each feature,
    as a raw count and a percentage. Handy for spotting features that are mostly
    empty before I trust them."""
    miss = df[FEATURE_COLUMNS].isna().sum()
    pct = (miss / len(df) * 100).round(2)
    return pd.DataFrame({"missing": miss, "missing_pct": pct})


def impute_features(df: pd.DataFrame, medians: pd.Series | None = None
                    ) -> tuple[pd.DataFrame, pd.Series]:
    """Fill in the gaps left by missing or undefined values. We use the median
    (the typical value) of each feature rather than the mean, since medians
    aren't thrown off by extreme outliers. The important bit: when scoring new
    data you pass in the medians worked out on the *training* set, so test rows
    never get to peek at their own statistics - that keeps the evaluation
    honest. We also return the medians so they can be reused that way."""
    out = df.copy()
    if medians is None:
        medians = out[FEATURE_COLUMNS].median(numeric_only=True)
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].fillna(medians)
    # If a whole column was empty there's no median to use, so fall back to 0.
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].fillna(0.0)
    return out, medians


if __name__ == "__main__":
    from data_generation import load_cbcr
    feats = engineer_features(load_cbcr())
    print("shape:", feats.shape)
    print(feats[FEATURE_COLUMNS].describe().T[["mean", "std", "min", "max"]])
    print("\nlabel balance:", feats[TARGET_COLUMN].value_counts().to_dict())
