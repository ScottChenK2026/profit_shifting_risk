"""Tests for the analysis modules added after the interim submission.

These are mostly guards against the kind of mistake that does not raise an exception and does not
look wrong in a table - a correction that quietly reorders predictions, a threshold fitted on the
wrong set, a bootstrap that resamples rows when it claims to resample jurisdictions. Every one of
those would produce a plausible number in the report, which is exactly why they need a test rather
than an eyeball.
"""

import numpy as np
import pandas as pd
import pytest

from ablation import _column_index, feature_subsets
from calibration import (fit_platt, operating_point, threshold_for_capacity)
from config import FEATURE_BLOCKS, FEATURE_COLUMNS
from data_generation import add_supplementary_columns, generate_synthetic_cbcr
from features import engineer_features
from hybrid import _rank01, evaluate_hybrid, rank_average
from label_sensitivity import relabel
from models.autoencoder import tune_threshold_on_validation
from reference_data import CONTESTED_CONDUITS, TAX_HAVENS, haven_set
from stability import grouped_bootstrap_auc, performance_by_slice


@pytest.fixture(scope="module")
def scores():
    """A synthetic pair of score vectors with a real but imperfect signal, plus labels."""
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.2, 900)
    p = np.clip(0.25 * y + rng.normal(0.35, 0.18, 900), 0.001, 0.999)
    return y, p


# --------------------------------------------------------------------------- #
# Feature blocks and ablation
# --------------------------------------------------------------------------- #
def test_feature_blocks_partition_the_feature_list():
    """The blocks must between them cover every feature exactly once. If a feature belonged to two
    blocks, the leave-one-out ablation would still be feeding it to the model through the other one
    and the measured drop would understate what the block is worth."""
    covered = [c for cols in FEATURE_BLOCKS.values() for c in cols]
    assert sorted(covered) == sorted(FEATURE_COLUMNS)
    assert len(covered) == len(set(covered)), "a feature appears in more than one block"


def test_leave_one_out_subsets_actually_drop_their_block():
    subsets = feature_subsets()
    assert subsets["full"] == list(FEATURE_COLUMNS)
    for block, cols in FEATURE_BLOCKS.items():
        without = set(subsets[f"without_{block}"])
        assert without.isdisjoint(cols)
        assert len(without) == len(FEATURE_COLUMNS) - len(cols)
        assert set(subsets[f"only_{block}"]) == set(cols)


def test_column_index_matches_feature_order():
    """The ablation slices columns out of the scaled matrix by position, so the positions had
    better line up with the names."""
    idx = _column_index(FEATURE_BLOCKS["activity_mix"])
    assert [FEATURE_COLUMNS[i] for i in idx] == FEATURE_BLOCKS["activity_mix"]


# --------------------------------------------------------------------------- #
# Calibration and thresholds
# --------------------------------------------------------------------------- #
def test_platt_scaling_preserves_the_ordering(scores):
    """Platt scaling may move the numbers as much as it likes but must not change which row ranks
    above which. If it did, it would be changing the model rather than correcting its scale, and
    the AUC reported before and after would no longer be comparable."""
    y, p = scores
    transform = fit_platt(p[:600], y[:600])
    out = transform(p[600:])
    order_before = np.argsort(p[600:])
    order_after = np.argsort(out)
    assert np.array_equal(order_before, order_after)


def test_threshold_matches_the_requested_capacity(scores):
    """A 10% capacity should flag about 10% of the set it was derived from."""
    _, p = scores
    thr = threshold_for_capacity(p, 0.10)
    assert 0.08 <= float((p >= thr).mean()) <= 0.12


def test_threshold_rejects_impossible_capacity():
    with pytest.raises(ValueError):
        threshold_for_capacity(np.array([0.1, 0.5]), 1.5)


def test_operating_point_reports_a_consistent_confusion_matrix(scores):
    y, p = scores
    op = operating_point(y, p, threshold=0.5)
    cm = np.array(op["confusion_matrix"])
    assert cm.sum() == len(y)
    tp, fn = cm[1, 1], cm[1, 0]
    assert op["recall"] == pytest.approx(tp / (tp + fn), abs=1e-9)


# --------------------------------------------------------------------------- #
# Autoencoder threshold
# --------------------------------------------------------------------------- #
def test_tuned_threshold_beats_the_degenerate_one():
    """The defect this replaced: a cut-off so high that almost nothing was flagged, giving a
    precision computed on a couple of cases. A tuned threshold has to flag enough to make the
    number mean something."""
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.2, 800)
    err = rng.normal(0, 1, 800) + 1.6 * y
    best = tune_threshold_on_validation(err, y)
    assert best["percentile"] is not None
    flagged = (err >= best["threshold"]).sum()
    assert flagged > 0.02 * len(y), "tuned threshold still flags almost nothing"
    assert best["f1"] > 0.3


# --------------------------------------------------------------------------- #
# Hybrid score
# --------------------------------------------------------------------------- #
def test_rank_transform_is_bounded_and_monotone(scores):
    _, p = scores
    r = _rank01(p)
    assert r.min() >= 0.0 and r.max() <= 1.0
    assert np.array_equal(np.argsort(p), np.argsort(r))


def test_rank_average_reduces_to_its_components_at_the_extremes(scores):
    _, p = scores
    other = p[::-1].copy()
    assert np.allclose(rank_average(p, other, weight=1.0), _rank01(p))
    assert np.allclose(rank_average(p, other, weight=0.0), _rank01(other))


def test_hybrid_reports_a_verdict_either_way(scores):
    """The hybrid study has to be able to return a negative result. A version that only produced a
    number when the combination helped would be useless as evidence."""
    y, p = scores
    rng = np.random.default_rng(11)
    noise = rng.uniform(0, 1, len(p))
    out = evaluate_hybrid(y[300:], p[300:], noise[300:], p[:300], noise[:300], y[:300])
    assert "verdict" in out and isinstance(out["verdict"]["helps"], bool)
    # Combining a real score with pure noise should not be recorded as an improvement.
    assert out["verdict"]["auc_gain_over_supervised"] < 0.05


# --------------------------------------------------------------------------- #
# Label sensitivity
# --------------------------------------------------------------------------- #
def test_haven_variants_are_nested_as_described():
    assert haven_set("strict_sinks") < haven_set("baseline")
    assert haven_set("conduits_only") <= set(CONTESTED_CONDUITS)
    assert haven_set("strict_sinks").isdisjoint(haven_set("conduits_only"))


def test_relabelling_changes_only_the_label():
    feats = engineer_features(add_supplementary_columns(generate_synthetic_cbcr(seed=5)))
    strict = relabel(feats, "strict_sinks")
    assert len(strict) == len(feats)
    # Features untouched, positives strictly fewer.
    pd.testing.assert_frame_equal(strict[FEATURE_COLUMNS], feats[FEATURE_COLUMNS])
    assert strict["tax_haven_binary"].sum() <= feats["tax_haven_binary"].sum()


def test_conduits_only_removes_the_uncontested_havens():
    """Relabelling the Cayman Islands as an ordinary country would poison the comparison, so those
    rows are dropped instead. This checks they really are gone."""
    feats = engineer_features(add_supplementary_columns(generate_synthetic_cbcr(seed=5)))
    conduits = relabel(feats, "conduits_only")
    uncontested = TAX_HAVENS - set(CONTESTED_CONDUITS)
    assert not conduits["partner_jurisdiction"].isin(uncontested).any()


# --------------------------------------------------------------------------- #
# Stability
# --------------------------------------------------------------------------- #
def test_grouped_bootstrap_brackets_the_point_estimate(scores):
    y, p = scores
    groups = np.repeat(np.arange(90), 10)
    out = grouped_bootstrap_auc(y, p, groups, n_resamples=120, seed=1)
    assert out["lo"] <= out["point"] <= out["hi"]
    assert out["n_groups"] == 90


def _clustered_scores(n_groups=90, per_group=10, seed=4):
    """Data shaped like the real test set: the label belongs to the jurisdiction, not the row, and
    every row about the same jurisdiction carries almost the same score."""
    rng = np.random.default_rng(seed)
    y_group = rng.binomial(1, 0.2, n_groups)
    p_group = np.clip(0.3 * y_group + rng.normal(0.35, 0.16, n_groups), 0.01, 0.99)
    y = np.repeat(y_group, per_group)
    # A little within-group jitter, far smaller than the spread between groups.
    p = np.clip(np.repeat(p_group, per_group)
                + rng.normal(0, 0.01, n_groups * per_group), 0.001, 0.999)
    return y, p, np.repeat(np.arange(n_groups), per_group)


def test_grouped_bootstrap_is_wider_when_rows_are_clustered():
    """The whole argument for the grouped bootstrap is that treating correlated rows as independent
    understates the uncertainty. On data where the label really belongs to the jurisdiction rather
    than the row - which is the situation in the test set, where one jurisdiction contributes dozens
    of near-identical rows - resampling whole jurisdictions has to give a wider interval than
    resampling rows, because there are far fewer independent things to resample."""
    y, p, groups = _clustered_scores()
    grouped = grouped_bootstrap_auc(y, p, groups, n_resamples=300, seed=2)
    row_level = grouped_bootstrap_auc(y, p, np.arange(len(y)), n_resamples=300, seed=2)
    assert (grouped["hi"] - grouped["lo"]) > 1.5 * (row_level["hi"] - row_level["lo"])


def test_grouped_bootstrap_matches_row_level_when_nothing_is_clustered(scores):
    """The mirror of the test above, and the one that stops it passing for the wrong reason. With
    no correlation inside the groups there is nothing for the grouped version to correct, so the
    two intervals should come out close. A grouped bootstrap that is always wider would be
    inflating the uncertainty rather than measuring it."""
    y, p = scores
    arbitrary = np.repeat(np.arange(90), 10)
    grouped = grouped_bootstrap_auc(y, p, arbitrary, n_resamples=300, seed=2)
    row_level = grouped_bootstrap_auc(y, p, np.arange(len(y)), n_resamples=300, seed=2)
    width_g = grouped["hi"] - grouped["lo"]
    width_r = row_level["hi"] - row_level["lo"]
    assert abs(width_g - width_r) < 0.4 * width_r


def test_performance_by_slice_drops_slices_that_are_too_small(scores):
    y, p = scores
    frame = pd.DataFrame({"reporting_jurisdiction":
                          ["BIG"] * 800 + ["TINY"] * 100})
    out = performance_by_slice(frame, y, p, "reporting_jurisdiction",
                               min_rows=200, min_positives=10)
    assert list(out["reporting_jurisdiction"]) == ["BIG"]
    assert out.attrs["skipped_slices"] == 1
