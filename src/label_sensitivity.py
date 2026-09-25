"""
label_sensitivity.py
--------------------
The study the interim report promised and could not yet deliver.

Everything in this project is predicted against a proxy. There is no register of confirmed profit
shifting, so the target is membership of a combined tax-haven list assembled from the EU's
non-cooperative jurisdictions, the Tax Justice Network rankings and the offshore-centre work of
Garcia-Bernardo et al. (2017). That list is a judgement, and other people's judgements differ. The
sharpest objection is that it puts the Netherlands, Switzerland, Ireland and Singapore in the same
box as the Cayman Islands, when those four are conduits with real economies attached - defensible,
but arguable, and arguable in a way that could be doing the model's work for it.

The worry is concrete rather than abstract. Those four are large, well-documented jurisdictions
with plenty of rows in the data. If the model is mostly learning to recognise them, the reported
AUC is partly an artefact of how the list was drawn and would not survive someone else drawing it
differently.

So the label is redrawn and everything is run again. Strict sinks removes the four contested
conduits from the positive class entirely, leaving only the end destinations - a narrower, harder,
less contestable target. Conduits only inverts the test: it keeps just those four as positives and
drops the uncontested havens from the data, asking whether the reported economics can pick out a
conduit at all when the easy cases have been taken away.

Whatever comes back is reportable. A result that holds under a redrawn label is worth much more
than one that needs a particular list; a result that collapses is a limitation I would rather find
myself than have a marker find for me.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import RANDOM_SEED, TARGET_COLUMN
from model_runner import run_model
from preprocessing import prepare_temporal_splits
from reference_data import CONTESTED_CONDUITS, haven_set

LABEL_VARIANTS = ("baseline", "strict_sinks", "conduits_only")


def relabel(feats: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Rebuild the target column under an alternative definition of what counts as a haven.

    Only the label moves. The features are untouched, the split by year is untouched, and the
    models are untouched, so any change in the results is attributable to the labelling and nothing
    else.

    ``conduits_only`` additionally drops the uncontested havens from the data rather than relabelling
    them as ordinary countries. Calling the Cayman Islands a non-haven would be worse than useless:
    the model would be trained to treat the clearest positive in the dataset as a negative example,
    and the resulting number would measure nothing. Removing those rows asks the cleaner question.
    """
    out = feats.copy()
    havens = haven_set(variant)

    if variant == "conduits_only":
        uncontested = haven_set("baseline") - set(CONTESTED_CONDUITS)
        out = out[~out["partner_jurisdiction"].isin(uncontested)].copy()

    out[TARGET_COLUMN] = out["partner_jurisdiction"].isin(havens).astype(int)
    return out.reset_index(drop=True)


def run_label_sensitivity(feats: pd.DataFrame,
                          models=("xgboost", "mlp", "ft_transformer"),
                          variants=LABEL_VARIANTS, seed: int = RANDOM_SEED,
                          verbose: bool = True) -> pd.DataFrame:
    """Re-run the whole out-of-time evaluation under each label definition.

    The splits are rebuilt from scratch for each variant, so the scaler and the imputation medians
    are refitted on the training years of that variant. That matters for ``conduits_only``, where
    rows have been removed and the training distribution is genuinely different.
    """
    rows = []
    for variant in variants:
        relabelled = relabel(feats, variant)
        splits = prepare_temporal_splits(relabelled)
        base_rate = float(splits.y_test.mean())
        if verbose:
            print(f"\n  [label] variant={variant}  rows={len(relabelled):,}  "
                  f"test base rate={base_rate:.1%}")
        for model_name in models:
            run = run_model(model_name, splits.X_train, splits.y_train, splits.X_val,
                            splits.y_val, splits.X_test, seed=seed)
            rec = {
                "variant": variant,
                "model": model_name,
                "n_rows": int(len(relabelled)),
                "test_base_rate": base_rate,
                "auc_roc": float(roc_auc_score(splits.y_test, run.p_test)),
                "pr_auc": float(average_precision_score(splits.y_test, run.p_test)),
                # PR-AUC is only interpretable against the base rate it sits on, and the base rate
                # changes between variants, so the lift is what makes the columns comparable.
                "pr_lift": float(average_precision_score(splits.y_test, run.p_test) / base_rate)
                if base_rate else float("nan"),
            }
            rows.append(rec)
            if verbose:
                print(f"    {model_name:<15} AUC={rec['auc_roc']:.3f}  "
                      f"PR-AUC={rec['pr_auc']:.3f}  lift={rec['pr_lift']:.1f}x")
    return pd.DataFrame(rows)
