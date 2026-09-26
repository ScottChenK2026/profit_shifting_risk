"""
cli.py
------
The command-line scorer: the part of the project a person would actually touch. The pipeline does
the training and leaves its artefacts in ``outputs/models``; this loads them back and scores either
one record typed as arguments or a whole CSV, returning a probability and a HIGH/MEDIUM/LOW band.

Two things about it changed after the interim submission, both because they were wrong rather than
because they were missing.

The first is which model it used. It loaded the MLP, always, while the README advertised it as
scoring with the best model and the evaluation had already established that the MLP is the weakest
of the three supervised models and the worst calibrated of them. Anyone taking the tool at its word
was getting the least trustworthy number the project produces. It now reads the pipeline's own
record of which model won and loads that one, with ``--model`` to override.

The second is subtler and was doing quiet damage. Building a single record from command-line
arguments, it copied tax paid into tax accrued, so two features that exist precisely to be compared
against each other - the gap between tax handed over and tax booked as owed is informative in itself
- were identical by construction for every record scored this way. There is now a separate argument
for it, and leaving it out means unknown rather than equal, which is the truthful reading.

Examples
--------
    # score a single record from raw CbCR quantities
    python src/cli.py --total-revenues 1000 --related-party-revenues 850 \\
        --profit-before-tax 600 --income-tax-paid 12 --num-employees 5 \\
        --tangible-assets 30 --stated-capital 10 \\
        --holding-establishments 40 --igf-establishments 15 \\
        --real-establishments 2

    # score every row of a CSV that has the raw CbCR columns
    python src/cli.py --csv data/oecd_cbcr_wide.csv --top 10

    # force a particular model rather than the pipeline's best
    python src/cli.py --csv data/oecd_cbcr_wide.csv --model mlp
"""

from __future__ import annotations

import argparse
import json
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

from config import FEATURE_COLUMNS, METRIC_DIR, MODEL_DIR
from features import engineer_features

AVAILABLE_MODELS = ("best", "xgboost", "mlp", "ft_transformer")


def _band(score: float) -> str:
    # A plain HIGH/MEDIUM/LOW label is easier for someone to act on than a bare number. The
    # cut-offs are my judgement calls, not something the model learned, and section 5.5 of the
    # report derives a defensible review threshold that a real deployment should use instead.
    if score >= 0.70:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"


def resolve_model_name(requested: str = "best") -> str:
    """Work out which model to score with.

    ``best`` means whichever model the last pipeline run recorded as having the highest test AUC,
    read from the summary the pipeline writes. Deciding this from the pipeline's own record rather
    than hard-coding a favourite means the tool cannot drift out of step with the evaluation: if a
    different model wins after a change, the scorer follows.
    """
    if requested != "best":
        return requested
    summary_path = METRIC_DIR / "pipeline_summary.json"
    if summary_path.exists():
        try:
            best = json.loads(summary_path.read_text()).get("best_model")
            if best in AVAILABLE_MODELS:
                return best
        except (ValueError, OSError):                  # pragma: no cover - defensive
            pass
    # No pipeline run to consult. XGBoost is the documented default because it is the model the
    # evaluation found strongest, but say so rather than choosing silently.
    print("[cli] no pipeline summary found; defaulting to xgboost.")
    return "xgboost"


def load_artifacts(model_name: str = "best"):
    """Load the scorer, the scaler and the imputation medians, so new records get exactly the
    treatment the training data did. Returns a callable that maps a scaled matrix to probabilities,
    alongside the preprocessing objects and the name of the model actually loaded."""
    name = resolve_model_name(model_name)

    with open(MODEL_DIR / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)
    with open(MODEL_DIR / "medians.pkl", "rb") as fh:
        medians = pickle.load(fh)

    if name == "xgboost":
        from xgboost import XGBClassifier
        booster = XGBClassifier()
        booster.load_model(str(MODEL_DIR / "xgboost.json"))
        return (lambda X: booster.predict_proba(X)[:, 1]), scaler, medians, name

    if name == "mlp":
        from models.mlp import ProfitShiftingMLP
        model = ProfitShiftingMLP(input_dim=len(FEATURE_COLUMNS))
        model.load_state_dict(torch.load(MODEL_DIR / "mlp_state.pt", map_location="cpu"))
        model.eval()
        return (lambda X: model.predict_proba(torch.from_numpy(X)).numpy()), scaler, medians, name

    if name == "ft_transformer":
        from models.ft_transformer import FTTransformer
        from config import FT_TRANSFORMER_CONFIG as ft_cfg
        model = FTTransformer(n_features=len(FEATURE_COLUMNS), d_token=ft_cfg["d_token"],
                n_heads=ft_cfg["n_heads"], n_layers=ft_cfg["n_layers"],
                dropout_p=ft_cfg["dropout_p"])
        model.load_state_dict(torch.load(MODEL_DIR / "ft_state.pt", map_location="cpu"))
        model.eval()
        return (lambda X: model.predict_proba(torch.from_numpy(X)).numpy()), scaler, medians, name

    raise ValueError(f"unknown model {name!r}; choose one of {AVAILABLE_MODELS}")


def score_frame(df_raw: pd.DataFrame, predict, scaler, medians) -> pd.DataFrame:
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
    X_scaled = scaler.transform(X.values).astype(np.float32)
    probs = np.asarray(predict(X_scaled)).ravel()

    id_cols = [c for c in ("reporting_jurisdiction", "partner_jurisdiction", "year")
            if c in feats.columns]
    result = feats[id_cols].copy() if id_cols else pd.DataFrame(index=feats.index)
    result["risk_score"] = probs.round(4)
    result["risk_band"] = [_band(p) for p in probs]
    return result


def build_single_record(args: argparse.Namespace) -> pd.DataFrame:
    """Build a one-row CbCR frame out of the numbers passed on the command line.

    Arguments left out take the defaults set in main(): zero for related-party revenue, tax paid
    and the establishment counts, and one for employees, entities, tangible assets and stated
    capital, so the ratios stay defined. Tax accrued is the exception: left out, it stays missing
    and the feature stage fills it with the training median, the same treatment a genuinely missing
    value gets in training. Accumulated earnings is always missing, since no feature uses it and it
    is there only so the column exists. The --real-establishments count is entered as
    manufacturing, standing in for all real-activity establishments.
    """
    accrued = args.income_tax_accrued
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
        # Not supplied means not known. Copying tax paid in here, as an earlier version did, made
        # the accrued-tax feature a duplicate of the paid-tax one for every record scored this way.
        "income_tax_accrued": accrued if accrued is not None else np.nan,
        "stated_capital": args.stated_capital,
        "accumulated_earnings": np.nan,
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
    p.add_argument("--model", default="best", choices=AVAILABLE_MODELS,
                   help="Which trained model to score with (default: the pipeline's best)")
    # single-record arguments
    p.add_argument("--total-revenues", type=float)
    p.add_argument("--related-party-revenues", type=float, default=0.0)
    p.add_argument("--profit-before-tax", type=float)
    p.add_argument("--income-tax-paid", type=float, default=0.0)
    p.add_argument("--income-tax-accrued", type=float, default=None,
                   help="Tax booked as owed. Left out means unknown, not equal to tax paid.")
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

    predict, scaler, medians, used = load_artifacts(args.model)
    print(f"[cli] scoring with: {used}")

    if args.csv:
        df_raw = pd.read_csv(args.csv)
        from data_generation import add_supplementary_columns
        # Again, the haven flag is only there so feature engineering has a label column to fill. It
        # is never given to the model as an input.
        if "is_known_haven" not in df_raw.columns:
            df_raw = add_supplementary_columns(df_raw)
        scored = score_frame(df_raw, predict, scaler, medians)
        scored = scored.sort_values("risk_score", ascending=False)
        print(scored.head(args.top).to_string(index=False))
    else:
        if args.total_revenues is None or args.profit_before_tax is None:
            p.error("Provide --csv OR at least --total-revenues and "
                    "--profit-before-tax for a single record.")
        row = score_frame(build_single_record(args), predict, scaler, medians).iloc[0]
        print(f"Risk score: {row['risk_score']:.4f}  (band: {row['risk_band']})")


if __name__ == "__main__":
    main()
