"""
data_generation.py
------------------
This is where the rest of the project gets its data from. Call load_cbcr() and
you get a table back - you don't have to care whether it's the real OECD data
or a fake stand-in.

The logic is simple: if the real, cleaned-up OECD table exists (the one
oecd_adapter.py produces), use it. If it doesn't - say I haven't downloaded the
350 MB file yet, or someone's just trying the code out - quietly fall back to a
synthetic dataset I generate here, with exactly the same columns so nothing
downstream notices the difference.

The fake data isn't random noise. I deliberately baked in the patterns the
research says to expect: in tax havens the booked profit doesn't match the
real activity on the ground, a lot of the revenue is just money moving between
parts of the same group, the offices are mostly holding/IP/finance/dormant
"paper" types, and very little tax actually gets paid. Then I add a lot of
random noise on top so the two groups genuinely overlap and the task isn't
trivially easy - otherwise testing on it would prove nothing.

To be clear, the synthetic data is only there to keep the pipeline runnable and
testable; it's not real evidence of anything. As soon as the real table shows
up (with a sensible number of rows) it takes over automatically, no code change
needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DATA_DIR, RANDOM_SEED, REAL_DATA_FILENAME, SYNTHETIC_DATA_FILENAME,
)
from reference_data import ACTIVITY_CODES, TAX_HAVENS

ACTIVITY_NAMES = list(ACTIVITY_CODES.values())

# The exact set of columns, in order, that every table in this project should
# have - the who/where/when, the money figures, the counts, the activity
# breakdown, and the two profit panels. This has to line up with what
# oecd_adapter.py spits out, otherwise the real and synthetic paths would
# disagree.
CBCR_COLUMNS: list[str] = [
    "reporting_jurisdiction", "partner_jurisdiction", "year",
    "total_revenues", "related_party_revenues", "unrelated_party_revenues",
    "profit_before_tax", "income_tax_paid", "income_tax_accrued",
    "stated_capital", "accumulated_earnings", "tangible_assets",
    "num_employees", "num_mne_groups", "num_subgroups", "num_entities",
] + ACTIVITY_NAMES + ["profit_positive_panel", "profit_negative_panel"]

REPORTING_JURISDICTIONS = [
    "USA", "GBR", "DEU", "FRA", "JPN", "CAN", "AUS", "ITA", "ESP", "SWE",
    "KOR", "BEL", "AUT", "FIN", "NOR", "DNK",
]
NON_HAVENS = [
    "USA", "GBR", "DEU", "FRA", "JPN", "CAN", "AUS", "ITA", "ESP", "SWE",
    "KOR", "BEL", "AUT", "FIN", "NOR", "DNK", "POL", "CZE", "PRT", "GRC",
    "MEX", "BRA", "IND", "CHN", "ZAF", "TUR", "NZL", "CHL",
]
YEARS = [2016, 2017, 2018, 2019, 2020, 2021]


# Draws positive numbers from a log-normal distribution. Handy because most of
# these quantities (revenue, assets, headcount) are always positive and span a
# huge range - a few giants, lots of small ones - which is what log-normal looks
# like.
def _ln(rng, m, s, n=1):
    return np.exp(rng.normal(m, s, n))


def generate_synthetic_cbcr(seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    havens = sorted(TAX_HAVENS)[:24]
    partners = sorted(set(havens) | set(NON_HAVENS))
    rows = []
    # One row for every reporting country x partner country x year combo - so we
    # build up a full panel, skipping the case where a country reports against
    # itself.
    for rep in REPORTING_JURISDICTIONS:
        for par in partners:
            if par == rep:
                continue
            for yr in YEARS:
                haven = par in TAX_HAVENS
                employees = max(1.0, _ln(rng, 5.5, 1.3)[0])
                entities = max(1.0, round(_ln(rng, 1.6, 0.8)[0]))
                groups = max(1.0, round(entities * rng.uniform(0.4, 0.9)))
                assets = _ln(rng, 15.5, 1.5)[0]
                unrelated = employees * _ln(rng, 11.0, 0.6)[0]

                # Havens and non-havens get drawn from different distributions.
                # For havens: lots of within-group revenue, high profit margins,
                # a low effective tax rate (the share of profit actually paid as
                # tax), and an office mix tilted towards holding/IP/finance/
                # dormant "paper" types. Non-havens look like normal businesses -
                # more modest margins, a normal-ish tax rate, mostly real
                # operations like factories and sales.
                if haven:
                    related_share = float(np.clip(rng.beta(4, 2.5), 0.05, 0.98))
                    margin = float(np.clip(rng.normal(0.33, 0.22), -0.1, 0.95))
                    etr = float(np.clip(rng.normal(0.09, 0.07), 0, 0.35))
                    act_w = np.array([0.4, 2.5, 2.0, 0.6, 0.3, 0.5, 0.8, 1.2,
                                      2.4, 0.7, 0.5, 1.8, 0.6])
                else:
                    related_share = float(np.clip(rng.beta(2.5, 4), 0.01, 0.92))
                    margin = float(np.clip(rng.normal(0.12, 0.11), -0.2, 0.6))
                    etr = float(np.clip(rng.normal(0.21, 0.08), 0, 0.45))
                    act_w = np.array([1.0, 0.5, 0.4, 1.2, 2.2, 2.0, 1.3, 0.6,
                                      0.4, 0.6, 1.6, 0.4, 0.8])
                # For ~18% of rows, scramble the numbers a bit so the two groups
                # blur into each other. Real data is messy and not every haven
                # looks like a textbook haven, so this stops the task being too
                # easy and keeps it honest.
                if rng.random() < 0.18:   # label noise
                    margin = float(np.clip(margin + rng.normal(0, 0.3), -0.2, 0.95))
                    related_share = float(np.clip(related_share + rng.normal(0, 0.3), 0.01, 0.98))
                    etr = float(np.clip(etr + rng.normal(0, 0.15), 0, 0.45))
                    act_w = act_w + rng.uniform(0, 1.5, size=act_w.size)

                # Build the actual money figures from the shares/margins above,
                # so everything stays internally consistent (e.g. total revenue
                # backs out from the unrelated part and the related-party share,
                # profit is margin x revenue, tax is the tax rate x profit, and
                # so on).
                total_rev = unrelated / max(1e-9, 1 - related_share)
                related = total_rev * related_share
                profit = total_rev * margin
                tax_paid = max(0.0, profit) * etr
                tax_accr = tax_paid * rng.uniform(1.0, 1.4)
                capital = assets * rng.uniform(0.1, 0.6)
                earnings = profit * rng.uniform(2, 6)

                # Turn the activity "weights" into actual whole-number office
                # counts. Poisson is the natural choice for "how many of these
                # do we see", and the weights tilt the counts haven vs non-haven.
                act_counts = rng.poisson(np.maximum(act_w * entities * 0.5, 0.05))

                # The OECD splits profit into the part booked by profit-making
                # sub-groups and the part booked by loss-making ones. We fake
                # both here; the gap between them is what later flags suspicious
                # loss-parking.
                if profit >= 0:
                    pos_panel = profit * rng.uniform(0.8, 1.2)
                    neg_panel = -abs(profit) * rng.uniform(0.05, 0.5)
                else:
                    pos_panel = abs(profit) * rng.uniform(0.1, 0.4)
                    neg_panel = profit * rng.uniform(0.8, 1.2)

                row = {
                    "reporting_jurisdiction": rep, "partner_jurisdiction": par,
                    "year": yr,
                    "total_revenues": total_rev,
                    "related_party_revenues": related,
                    "unrelated_party_revenues": unrelated,
                    "profit_before_tax": profit,
                    "income_tax_paid": tax_paid,
                    "income_tax_accrued": tax_accr,
                    "stated_capital": capital, "accumulated_earnings": earnings,
                    "tangible_assets": assets, "num_employees": employees,
                    "num_mne_groups": groups, "num_subgroups": groups,
                    "num_entities": entities,
                    "profit_positive_panel": pos_panel,
                    "profit_negative_panel": neg_panel,
                }
                for name, c in zip(ACTIVITY_NAMES, act_counts):
                    row[name] = float(c)
                rows.append(row)
    return pd.DataFrame(rows, columns=CBCR_COLUMNS)


def add_supplementary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Tags each row with a 1/0 flag for whether the partner country is on the
    tax-haven list. This is the answer the model learns to predict, so we attach
    it here regardless of whether the data is real or synthetic."""
    df = df.copy()
    df["is_known_haven"] = df["partner_jurisdiction"].isin(TAX_HAVENS).astype(int)
    return df


def load_cbcr(prefer_real: bool = True, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """The single entry point everything else uses to get data. Tries the real
    OECD table first, falls back to a cached synthetic file, and only generates
    fresh synthetic data as a last resort. Either way you get a labelled table
    back."""
    real_path = DATA_DIR / REAL_DATA_FILENAME
    synth_path = DATA_DIR / SYNTHETIC_DATA_FILENAME
    # Guard against a real file that's basically empty (e.g. a failed download) -
    # we'd rather use proper synthetic data than a handful of real rows.
    MIN_REAL_ROWS = 50

    if prefer_real and real_path.exists():
        df = pd.read_csv(real_path)
        if len(df) >= MIN_REAL_ROWS:
            print(f"[data] Using REAL OECD wide table: {real_path.name} "
                  f"({len(df):,} rows)")
            return add_supplementary_columns(df)
        print(f"[data] {real_path.name} has only {len(df)} rows (<50); "
              "falling back to synthetic.")

    if synth_path.exists():
        df = pd.read_csv(synth_path)
        if len(df) >= MIN_REAL_ROWS:
            print(f"[data] Using cached synthetic dataset: {synth_path.name}")
            return add_supplementary_columns(df)

    print("[data] Generating synthetic stand-in.")
    df = generate_synthetic_cbcr(seed=seed)
    df.to_csv(synth_path, index=False)
    print(f"[data] Synthetic dataset written to {synth_path}")
    return add_supplementary_columns(df)


if __name__ == "__main__":
    d = load_cbcr()
    print(d.shape)
    print(d.head())
