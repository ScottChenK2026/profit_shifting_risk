"""
features.py
-----------
This is where a raw table of numbers becomes the actual signals a model can learn from. Revenue or
profit on its own says very little; what matters is the ratios, things like profit per employee.
So this file does that arithmetic. Every signal is built only from what the companies report - no
country name, no outside tax rate - for the leakage reasons set out in config.py.

There are three families of signals, plus one extra:

1. Profitability and tax. How fat the margin is, how much of the revenue is just money moving
   within the group, and how much tax is actually paid relative to profit.
2. Substance ratios. Profit, revenue and capital per employee, per asset, per entity. The gut
   check here: if a place books enormous profit but has barely any staff or assets, that mismatch
   is exactly what we are looking for.
3. Activity mix. What fraction of the local offices are paper-office types - holding, IP, internal
   finance, dormant - against the ones doing real work. This is the part of the OECD data that
   machine-learning work on the topic has mostly overlooked, and a big part of why I wanted to use
   it.

On top of those, a loss-shifting signal built from the separate profit and loss sub-group figures.
Put plainly: is an unusual amount of loss being parked here?

The answer (label)
------------------
What the model predicts is simply whether the partner country appears on the haven list in
reference_data.py. That list is decided completely separately from any of the signals above, which
makes the question a fair one: can the reported economics and the office mix, on their own, reveal
the known havens?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, TARGET_COLUMN
from reference_data import (
    ACTIVITY_CODES, REAL_ACTIVITIES, SHIFTING_PRONE_ACTIVITIES,
)

ACTIVITY_NAMES = list(ACTIVITY_CODES.values())


# Divide, but turn a division by zero into a missing value rather than letting it blow up to
# infinity. Those gaps get filled in sensibly later.
def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def _slog(x: pd.Series) -> pd.Series:
    """A log that keeps the sign. Profit can be negative and a plain log would choke on that, but I
    still want to know whether the value was a profit or a loss - so log the size and put the
    original sign back on."""
    return np.sign(x) * np.log1p(x.abs())


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Office counts: if a kind of office is not listed for a country, that almost always means
    # there are none, not that the number is unknown. Filling the blank with 0 is the truthful
    # answer here.
    for name in ACTIVITY_NAMES:
        if name not in out.columns:
            out[name] = 0.0
        out[name] = out[name].fillna(0.0)
    act_total = out[ACTIVITY_NAMES].sum(axis=1)

    # --- Profitability and tax ---------------------------------------------- #
    # Margin and related-party share are clipped to sane ranges so that a few extreme outliers,
    # usually the product of a tiny denominator, do not dominate everything else.
    out["profit_margin"] = _safe_div(out["profit_before_tax"], out["total_revenues"]).clip(-5, 5)
    out["related_party_share"] = _safe_div(
        out.get("related_party_revenues", pd.Series(np.nan, index=out.index)),
        out["total_revenues"]).clip(0, 1)

    # Effective tax rate: tax paid over profit, the share of profit that actually leaves as tax.
    # Blanked out when profit is zero or negative, because tax as a fraction of a loss is
    # meaningless and would produce nonsense.
    etr = _safe_div(out["income_tax_paid"], out["profit_before_tax"])
    etr[out["profit_before_tax"] <= 0] = np.nan
    out["effective_tax_rate"] = etr.clip(0, 1)

    # The same idea using tax owed rather than tax handed over. The two can differ, and the gap is
    # informative in itself.
    etr_a = _safe_div(out["income_tax_accrued"], out["profit_before_tax"])
    etr_a[out["profit_before_tax"] <= 0] = np.nan
    out["etr_accrued"] = etr_a.clip(0, 1)

    # --- Substance ratios --------------------------------------------------- #
    # Real business needs real people and real assets. If profit per employee or per dollar of
    # assets is sky-high, the profit probably was not earned where it is being booked.
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

    # --- Business-activity mix ---------------------------------------------- #
    # Turn raw office counts into shares of all offices in that country, so a large economy and a
    # small one are compared like for like. The individual paper-office types come out separately,
    # and then a combined share of all shifting-prone types against all real-operations types.
    out["holding_share"] = _safe_div(out["holding_equity"], act_total)
    out["ip_share"] = _safe_div(out["ip_management"], act_total)
    out["igf_share"] = _safe_div(out["internal_group_finance"], act_total)
    out["dormant_share"] = _safe_div(out["dormant"], act_total)
    shifting = out[SHIFTING_PRONE_ACTIVITIES].sum(axis=1)
    real = out[REAL_ACTIVITIES].sum(axis=1)
    out["shifting_activity_share"] = _safe_div(shifting, act_total)
    out["real_activity_share"] = _safe_div(real, act_total)

    # --- Loss-shifting signal ----------------------------------------------- #
    # Of all the profit and loss sloshing around here, how much of it is loss? A country that is
    # nearly all losses may be where a group parks them deliberately, so the ratio is worth handing
    # to the model.
    pos = out.get("profit_positive_panel", pd.Series(np.nan, index=out.index))
    neg = out.get("profit_negative_panel", pd.Series(np.nan, index=out.index))
    out["loss_shift_ratio"] = _safe_div(neg.abs(), pos.abs() + neg.abs())

    # --- Label --------------------------------------------------------------- #
    # Copy the haven flag into the target column. If it is not there - scoring brand-new data we
    # have no labels for, say - default to 0 so the pipeline still runs.
    if "is_known_haven" in out.columns:
        out[TARGET_COLUMN] = out["is_known_haven"].astype(int)
    else:
        out[TARGET_COLUMN] = 0

    # Hand back the ID columns, the engineered features and the label, dropping the intermediate
    # raw quantities we no longer need.
    keep = (["reporting_jurisdiction", "partner_jurisdiction", "year"]
            + FEATURE_COLUMNS + [TARGET_COLUMN])
    keep = [c for c in keep if c in out.columns]
    return out[keep]


def summarise_missing(df: pd.DataFrame) -> pd.DataFrame:
    """How many values are missing in each feature, as a count and a percentage. A quick way to
    spot a feature that is mostly empty before I start trusting it."""
    miss = df[FEATURE_COLUMNS].isna().sum()
    pct = (miss / len(df) * 100).round(2)
    return pd.DataFrame({"missing": miss, "missing_pct": pct})


def impute_features(df: pd.DataFrame, medians: pd.Series | None = None
                    ) -> tuple[pd.DataFrame, pd.Series]:
    """Fill the gaps left by missing or undefined values. I use the median of each feature rather
    than the mean, since medians are not dragged around by extreme outliers. The important part is
    the second argument: when scoring later data you pass in the medians worked out on the training
    set, so test rows never get to peek at their own statistics. The medians are returned so they
    can be reused that way."""
    out = df.copy()
    if medians is None:
        medians = out[FEATURE_COLUMNS].median(numeric_only=True)
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].fillna(medians)
    # If a whole column was empty there is no median to use, so fall back to 0.
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].fillna(0.0)
    return out, medians


if __name__ == "__main__":
    from data_generation import load_cbcr
    feats = engineer_features(load_cbcr())
    print("shape:", feats.shape)
    print(feats[FEATURE_COLUMNS].describe().T[["mean", "std", "min", "max"]])
    print("\nlabel balance:", feats[TARGET_COLUMN].value_counts().to_dict())
