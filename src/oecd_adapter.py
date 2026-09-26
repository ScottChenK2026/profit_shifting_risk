"""
oecd_adapter.py
---------------
A one-off cleanup script. It takes the raw file you download from the OECD Data Explorer - their
Table I country-by-country reporting statistics - and reshapes it into the neat table the rest of
the project expects, saved as ``data/oecd_cbcr_wide.csv``.

Why this has to exist: the raw download is about 350 MB and close to a million rows, laid out in a
long SDMX shape with one row per country pair per measure per year. A single country pair is
spread across dozens of rows. It is too big to load in one go and the wrong shape for modelling
anyway. So this reads it in chunks to keep memory sane, throws away the rows we do not need, and
rotates what is left into one tidy row per reporting country, partner country and year, with each
measure as its own column.

What it pulls out
-----------------
1. The money figures, in USD: revenue split into total, within-group and outside, profit before
    tax, an adjusted profit figure, tax paid and tax owed, share capital, accumulated earnings and
    tangible assets.
2. The counts: employees, and how many groups, sub-groups and entities.
3. The office-activity counts for all thirteen OECD activity types, from holding, internal group
    finance, IP and dormant through to manufacturing, sales and services, which feed the
    activity-mix signals in features.py that I think are the novel part of this project.
4. From the separate profit-makers and loss-makers breakdown, which the OECD calls PANELAI and
    PANELAII, how much profit sits with profitable sub-groups versus loss-making ones. That becomes
    the loss-parking signal.

How to run it
-------------
    python src/oecd_adapter.py data/OECD_CbCR_unfiltered_raw.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, REAL_DATA_FILENAME          # noqa: E402
from reference_data import (                              # noqa: E402
    ACTIVITY_CODES, FINANCIAL_CODES, is_real_jurisdiction,
)

# Only the code columns are read, not the human-readable label columns. The codes are all we match
# on, and the labels contain commas that make the CSV fiddlier to parse.
USECOLS = ["REF_AREA", "COUNTERPART_AREA", "MEASURE", "PROFIT_GROUPING",
           "STATISTICAL_OPERATION", "AGGREGATION_TYPE", "TIME_PERIOD",
           "OBS_VALUE"]

# The measures we care about. PROFIT is added separately below because it is also needed for the
# profit and loss panels.
WANTED_MEASURES = set(FINANCIAL_CODES) | set(ACTIVITY_CODES)
# Rows per chunk: big enough to be quick, small enough to sit comfortably in memory.
CHUNK = 200_000


def _stream_filter(raw_path: Path) -> pd.DataFrame:
    """Work through the file a chunk at a time, discarding everything we do not need, so the whole
    thing never has to be held in memory at once."""
    keep = []
    total = 0
    for chunk in pd.read_csv(raw_path, usecols=USECOLS, chunksize=CHUNK, dtype=str, low_memory=False):
        total += len(chunk)
        # Keep the proper totals rather than partial breakdowns, the measures we actually use, and
        # only rows where the partner is a real country rather than a regional roll-up.
        m = (
            (chunk["AGGREGATION_TYPE"] == "TOTAL")
            & (chunk["STATISTICAL_OPERATION"] == "_Z")
            & (chunk["MEASURE"].isin(WANTED_MEASURES | {"PROFIT"}))
            & (chunk["COUNTERPART_AREA"].map(is_real_jurisdiction))
        )
        keep.append(chunk[m])
    df = pd.concat(keep, ignore_index=True)
    # Values arrive as text. Turn them into numbers, and let anything that will not convert become
    # a blank rather than breaking the whole run.
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    print(f"[adapter] scanned {total:,} rows -> kept {len(df):,} relevant rows")
    return df


def adapt_oecd_export(raw_path: str | Path,
                      out_path: str | Path | None = None) -> pd.DataFrame:
    """The whole conversion: read and filter the raw file, reshape it wide, bolt on the profit and
    loss panels, tidy up the column names, and save the result."""
    raw_path = Path(raw_path)
    out_path = Path(out_path) if out_path else (DATA_DIR / REAL_DATA_FILENAME)
    print(f"[adapter] reading {raw_path} ({raw_path.stat().st_size/1e6:.0f} MB)")

    df = _stream_filter(raw_path)
    # The three things that together identify one row of the final table.
    idx = ["REF_AREA", "COUNTERPART_AREA", "TIME_PERIOD"]

    # The main table uses only the "all sub-groups combined" rows, coded _T. Pivot so each measure
    # becomes its own column, then swap OECD's codes for the readable names in reference_data.py.
    tot = df[df["PROFIT_GROUPING"] == "_T"]
    wide = (tot.pivot_table(index=idx, columns="MEASURE", values="OBS_VALUE",
                            aggfunc="first").reset_index())
    rename = {**FINANCIAL_CODES, **ACTIVITY_CODES}
    wide = wide.rename(columns=rename)

    # Now the profit split between profit-making and loss-making sub-groups, each attached as its
    # own column. If a panel is missing from this particular export, leave the column blank instead
    # of failing.
    for panel, name in [("PANELAI", "profit_positive_panel"),
                        ("PANELAII", "profit_negative_panel")]:
        sub = df[(df["PROFIT_GROUPING"] == panel) & (df["MEASURE"] == "PROFIT")]
        if len(sub):
            piv = (sub.pivot_table(index=idx, values="OBS_VALUE",
                                   aggfunc="first").reset_index()
                      .rename(columns={"OBS_VALUE": name}))
            wide = wide.merge(piv, on=idx, how="left")
        else:
            wide[name] = pd.NA

    # Rename the three ID columns to the friendly names used everywhere else.
    wide = wide.rename(columns={"REF_AREA": "reporting_jurisdiction",
                                "COUNTERPART_AREA": "partner_jurisdiction",
                                "TIME_PERIOD": "year"})
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype("Int64")

    # Drop rows with neither revenue nor profit. Without at least one of the two there is nothing
    # to model - they are empty shells.
    core = ["total_revenues", "profit_before_tax"]
    present_core = [c for c in core if c in wide.columns]
    wide = wide.dropna(subset=present_core, how="all")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_path, index=False)
    print(f"[adapter] wrote {len(wide):,} wide rows x {wide.shape[1]} cols "
          f"-> {out_path}")
    # A heads-up if measures we hoped for are simply not in this download. Not fatal, but better to
    # know now than to be puzzled by gaps later.
    miss = [c for c in {**FINANCIAL_CODES, **ACTIVITY_CODES}.values()
            if c not in wide.columns]
    if miss:
        print(f"[adapter] note: measures absent from this export: {miss}")
    return wide


if __name__ == "__main__":
    # The path to the downloaded raw CSV is the one argument. Without it, print the notes at the
    # top of this file and stop.
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nERROR: provide the path to the downloaded OECD CSV.")
        sys.exit(1)
    adapt_oecd_export(sys.argv[1])
