"""
stability.py
------------
How much of the headline number should be believed, and does it hold up when the test year is cut
into pieces?

Two weaknesses in the preliminary evaluation prompted this file. The first is that every AUC was
reported as a bare point estimate, so a gap of two points between models had nothing attached to it
that would say whether two points is a lot. DeLong's test partly covers that, but only for a
pairwise comparison, and it assumes the rows are independent.

They are not, and that is the second weakness. The unit of analysis is a reporting-partner-year
combination, so the Cayman Islands appears once for every country that reports against it - dozens
of rows describing the same jurisdiction, and a model that gets Cayman right gets all of them right
together. Treating those as independent observations overstates how much evidence the test set
contains. The fix used here is a grouped bootstrap: resample whole partner jurisdictions with
replacement rather than individual rows, so a jurisdiction is either in a resample or out of it.
That gives a confidence interval that respects the real unit of variation, and it is wider than a
naive row-level interval, which is the honest outcome.

The per-slice breakdown answers a different question. An overall AUC can hide a model that works
well for large reporting countries and badly for small ones, and a screening tool that only works
for the countries with the most complete filings is much less useful than the headline suggests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import BOOTSTRAP_RESAMPLES, RANDOM_SEED


def grouped_bootstrap_auc(y_true, y_prob, groups, n_resamples: int = BOOTSTRAP_RESAMPLES,
                          seed: int = RANDOM_SEED, alpha: float = 0.05) -> dict:
    """A percentile confidence interval for AUC, resampling whole groups rather than rows.

    ``groups`` is the partner jurisdiction for each row. Each resample draws that many jurisdictions
    with replacement and stacks up all of their rows, so the correlation between rows about the same
    place is carried through into the interval instead of being assumed away. Resamples that end up
    with only one class present are skipped, since AUC is undefined there.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    groups = np.asarray(groups)

    unique = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)

    scores: list[float] = []
    for _ in range(n_resamples):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in drawn])
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        scores.append(roc_auc_score(yt, y_prob[idx]))

    if not scores:                                     # pragma: no cover - defensive
        return {"point": float(roc_auc_score(y_true, y_prob)), "n_effective": 0}

    arr = np.asarray(scores)
    return {
        "point": float(roc_auc_score(y_true, y_prob)),
        "lo": float(np.quantile(arr, alpha / 2)),
        "hi": float(np.quantile(arr, 1 - alpha / 2)),
        "sd": float(arr.std(ddof=1)),
        "n_groups": int(len(unique)),
        "n_effective": int(len(arr)),
    }


def paired_group_bootstrap(y_true, prob_a, prob_b, groups,
                           n_resamples: int = BOOTSTRAP_RESAMPLES,
                           seed: int = RANDOM_SEED) -> dict:
    """The same idea applied to the gap between two models.

    Both models are scored on identical resamples, so the difference in AUC is measured on the same
    jurisdictions every time and the shared variation cancels out. The fraction of resamples in
    which model A comes out ahead is a bootstrap analogue of a p-value, and unlike DeLong's test it
    does not assume the rows are independent. Where the two disagree, this one is the more
    conservative and I report it as the stricter check.
    """
    y_true = np.asarray(y_true)
    prob_a, prob_b = np.asarray(prob_a), np.asarray(prob_b)
    groups = np.asarray(groups)

    unique = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)

    diffs: list[float] = []
    for _ in range(n_resamples):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in drawn])
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        diffs.append(roc_auc_score(yt, prob_a[idx]) - roc_auc_score(yt, prob_b[idx]))

    arr = np.asarray(diffs)
    if arr.size == 0:                                  # pragma: no cover - defensive
        return {"mean_diff": 0.0, "n_effective": 0}
    return {
        "mean_diff": float(arr.mean()),
        "lo": float(np.quantile(arr, 0.025)),
        "hi": float(np.quantile(arr, 0.975)),
        "p_a_better": float((arr > 0).mean()),
        "n_effective": int(arr.size),
    }


def performance_by_slice(test_frame: pd.DataFrame, y_true, y_prob,
                         column: str, min_rows: int = 60,
                         min_positives: int = 10) -> pd.DataFrame:
    """AUC and PR-AUC computed separately within each value of a column, for slices big enough for
    the number to mean anything.

    The thresholds are there to stop the table filling up with a jurisdiction that contributed four
    rows and one haven, where an AUC of 1.000 says nothing. Slices that fall below them are counted
    and reported as excluded rather than quietly dropped.
    """
    df = test_frame.reset_index(drop=True).copy()
    df["_y"] = np.asarray(y_true)
    df["_p"] = np.asarray(y_prob)

    rows, skipped = [], 0
    for value, part in df.groupby(column):
        if len(part) < min_rows or part["_y"].sum() < min_positives \
                or part["_y"].nunique() < 2:
            skipped += 1
            continue
        rows.append({
            column: value,
            "n": int(len(part)),
            "positives": int(part["_y"].sum()),
            "base_rate": float(part["_y"].mean()),
            "auc_roc": float(roc_auc_score(part["_y"], part["_p"])),
            "pr_auc": float(average_precision_score(part["_y"], part["_p"])),
        })

    out = pd.DataFrame(rows).sort_values("auc_roc", ascending=False) \
        if rows else pd.DataFrame(columns=[column, "n", "positives", "base_rate",
                                           "auc_roc", "pr_auc"])
    out.attrs["skipped_slices"] = skipped
    return out
