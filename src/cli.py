"""
cli.py
------
The little command-line tool that puts the trained model to use. The pipeline does the heavy
lifting and leaves the trained MLP, the fitted scaler and the training medians in
``outputs/models``; this script loads them back and scores whatever record you hand it, either
typed in as numbers on the command line or as a whole CSV. What comes back is a risk score between
0 and 1, with a friendlier HIGH/MEDIUM/LOW band on top of it.

Examples
--------
    # score a single record from raw CbCR quantities
    python -m src.cli --total-revenues 1000 --related-party-revenues 850 \
        --profit-before-tax 600 --income-tax-paid 12 --num-employees 5 \
        --tangible-assets 30 --stated-capital 10 \
        --holding-establishments 40 --igf-establishments 15 \
        --real-establishments 2

    # score every row of a CSV that has the raw CbCR columns
    python -m src.cli --csv data/cbcr_synthetic.csv --top 10
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# This needs to run both as ``python -m src.cli`` and as ``python src/cli.py``. config.py lives in
# the project root, the other modules live in src/, and both use flat imports - so both directories
# go on the path.
_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))            # src/
sys.path.insert(0, str(_SRC_DIR.parent))     # project root, where config.py is

import numpy as np
import pandas as pd
import torch

from config import FEATURE_COLUMNS, MODEL_DIR
from features import engineer_features
from models.mlp import ProfitShiftingMLP


def _band(score: float) -> str:
    # A plain HIGH/MEDIUM/LOW label is easier for someone to act on than a bare number. The
    # cut-offs are my judgement calls, not something the model learned.
    if score >= 0.70:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"


def load_artifacts() -> tuple[ProfitShiftingMLP, object, pd.Series]:
    """Pull back everything the pipeline saved - the trained MLP, the scaler, the imputation
    medians - so that new records get exactly the same treatment the training data did."""
    with open(MODEL_DIR / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)
    with open(MODEL_DIR / "medians.pkl", "rb") as fh:
        medians = pickle.load(fh)
    state = torch.load(MODEL_DIR / "mlp_state.pt", map_location="cpu")
    model = ProfitShiftingMLP(input_dim=len(FEATURE_COLUMNS))
    model.load_state_dict(state)
    model.eval()
    return model, scaler, medians


def score_frame(df_raw: pd.DataFrame, model, scaler, medians) -> pd.DataFrame:
    """Take raw CbCR rows, run them through the same feature, imputation and scaling steps as
    training, and return each row with a risk score and band attached."""
    # Feature engineering copes fine with missing raw columns - they turn into NaN and get filled
    # with the training medians. The one thing it insists on is a haven flag, and here that exists
    # only to satisfy the label column. It is a neutral 0 and is never fed to the model.
    df = df_raw.copy()
    if "is_known_haven" not in df.columns:
        df["is_known_haven"] = 0

    feats = engineer_features(df)
    X = feats[FEATURE_COLUMNS].fillna(medians)
    X_scaled = scaler.transform(X.values.astype(np.float32))
    with torch.no_grad():
        probs = model.predict_proba(torch.from_numpy(X_scaled)).numpy()

    result = feats[["reporting_jurisdiction", "partner_jurisdiction",
                    "year"]].copy() if "reporting_jurisdiction" in feats else \
        pd.DataFrame(index=feats.index)
    result["risk_score"] = probs.round(4)
    result["risk_band"] = [_band(p) for p in probs]
    return result


def build_single_record(args: argparse.Namespace) -> pd.DataFrame:
    """Build a one-row CbCR frame out of the numbers passed on the command line. Only the core
    financials really matter; the activity-mix and panel fields fall back to the optional flags, or
    are left blank and imputed. A couple of fields, accumulated earnings for instance, are rough
    stand-ins so the feature engineering has something to work with for a one-off lookup."""
    return pd.DataFrame([{
        "reporting_jurisdiction": "CLI",
        "partner_jurisdiction": "CLI",
        "year": 0,
        "num_mne_groups": 1,
        "num_entities": max(1.0, args.num_entities),
        "total_revenues": args.total_revenues,
        "related_party_revenues": args.related_party_revenues,
        "unrelated_party_revenues": max(
            0.0, args.total_revenues - args.related_party_revenues),
        "profit_before_tax": args.profit_before_tax,
        "income_tax_paid": args.income_tax_paid,
        "income_tax_accrued": args.income_tax_paid,
        "stated_capital": args.stated_capital,
        "accumulated_earnings": args.profit_before_tax * 3,
        "tangible_assets": args.tangible_assets,
        "num_employees": args.num_employees,
        "holding_equity": args.holding_establishments,
        "internal_group_finance": args.igf_establishments,
        "ip_management": args.ip_establishments,
        "dormant": args.dormant_establishments,
        "manufacturing": args.real_establishments,
        "is_known_haven": 0,
    }])


def main() -> None:
    p = argparse.ArgumentParser(description="Profit-shifting risk scorer")
    p.add_argument("--csv", help="CSV of raw CbCR records to score")
    p.add_argument("--top", type=int, default=20,
                   help="Show only the N highest-risk rows when scoring a CSV")
    # single-record arguments
    p.add_argument("--total-revenues", type=float)
    p.add_argument("--related-party-revenues", type=float, default=0.0)
    p.add_argument("--profit-before-tax", type=float)
    p.add_argument("--income-tax-paid", type=float, default=0.0)
    p.add_argument("--num-employees", type=float, default=1.0)
    p.add_argument("--num-entities", type=float, default=1.0)
    p.add_argument("--tangible-assets", type=float, default=1.0)
    p.add_argument("--stated-capital", type=float, default=1.0)
    # optional business-activity establishment counts
    p.add_argument("--holding-establishments", type=float, default=0.0)
    p.add_argument("--igf-establishments", type=float, default=0.0)
    p.add_argument("--ip-establishments", type=float, default=0.0)
    p.add_argument("--dormant-establishments", type=float, default=0.0)
    p.add_argument("--real-establishments", type=float, default=0.0)
    args = p.parse_args()

    model, scaler, medians = load_artifacts()

    if args.csv:
        df_raw = pd.read_csv(args.csv)
        from data_generation import add_supplementary_columns
        # Again, the haven flag is only there so feature engineering has a label column to fill. It
        # is never given to the model as an input.
        if "is_known_haven" not in df_raw.columns:
            df_raw = add_supplementary_columns(df_raw)
        scored = score_frame(df_raw, model, scaler, medians)
        scored = scored.sort_values("risk_score", ascending=False)
        print(scored.head(args.top).to_string(index=False))
    else:
        if args.total_revenues is None or args.profit_before_tax is None:
            p.error("Provide --csv OR at least --total-revenues and "
                    "--profit-before-tax for a single record.")
        df_raw = build_single_record(args)
        scored = score_frame(df_raw, model, scaler, medians)
        row = scored.iloc[0]
        print(f"Risk score: {row['risk_score']:.4f}  "
              f"(band: {row['risk_band']})")


if __name__ == "__main__":
    main()
