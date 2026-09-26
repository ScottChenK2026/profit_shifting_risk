"""
run_experiments.py
------------------
The second half of the project: everything that goes beyond "train four models and report their
scores".

``run_pipeline.py`` produces the headline comparison. This script produces the analysis that makes
the comparison mean something, and each study here exists because of a specific question the
interim evaluation left open:

    1. Ablation: is the business-activity mix actually carrying the signal, or does SHAP only make 
                 it look that way?
    2. Capacity ladder: would a bigger or less regularised neural network have won? This is the
                        scale-up-then-regularise workflow the project template asks for.
    3. Learning curves: is the neural models' deficit a sample-size problem?
    4. Calibration: is the neural models' poor Brier score a ranking failure or a scaling one?
    5. Operating point: what do the models look like when all four are held to the same review
                        capacity instead of an arbitrary 0.5 cut-off?
    6. Label sensitivity:how much of the result depends on my particular tax-haven list?
    7. Hybrid score: does the label-free autoencoder add anything to the supervised score?
    8. Stability: how wide is the uncertainty once rows about the same jurisdiction are treated as 
                  the correlated observations they are, and does performance hold up across 
                  reporting countries?

Everything is written to ``outputs/metrics/experiments.json`` and the figures to
``outputs/figures``. Studies are cached individually, so a run that is interrupted or a study that
is being reworked does not mean repeating the lot; ``--only NAME`` runs a single study and
``--fresh`` ignores the cache.

Run time is roughly forty minutes on two CPU cores, dominated by the transformer, which gets
retrained about twenty-five times across the reference run, the ablation, the ladder, the curves
and the label study.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import AUDIT_CAPACITY, METRIC_DIR, RANDOM_SEED     # noqa: E402
import experiment_figures as figs                             # noqa: E402
from ablation import ablation_deltas, run_ablation, run_learning_curve  # noqa: E402
from calibration import calibration_report, capacity_comparison         # noqa: E402
from capacity_study import run_full_study, summarise_ladder             # noqa: E402
from data_generation import load_cbcr                                   # noqa: E402
from features import engineer_features                                  # noqa: E402
from hybrid import evaluate_hybrid                                      # noqa: E402
from label_sensitivity import run_label_sensitivity                     # noqa: E402
from model_runner import run_model                                      # noqa: E402
from preprocessing import prepare_temporal_splits                       # noqa: E402
from stability import (grouped_bootstrap_auc, paired_group_bootstrap,   # noqa: E402
                       performance_by_slice)

RESULTS_PATH = METRIC_DIR / "experiments.json"
SUPERVISED = ("xgboost", "mlp", "ft_transformer")


def _load_cache() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {}


def _save(results: dict) -> None:
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[experiments] saved -> {RESULTS_PATH}")


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrames go into the JSON as plain records, so the report builder and any later reader do
    not need pandas to make sense of the file."""
    return json.loads(df.to_json(orient="records"))


def main() -> dict:
    ap = argparse.ArgumentParser(description="Post-interim analysis studies")
    ap.add_argument("--fresh", action="store_true", help="ignore cached study results")
    ap.add_argument("--only", help="run a single study by name")
    args = ap.parse_args()

    np.random.seed(RANDOM_SEED)
    results = {} if args.fresh else _load_cache()

    def wanted(name: str) -> bool:
        if args.only:
            return args.only == name
        return args.fresh or name not in results

    print("=== Loading data and rebuilding the out-of-time split ===")
    feats = engineer_features(load_cbcr())
    splits = prepare_temporal_splits(feats)
    print(f"train={splits.X_train.shape}  val={splits.X_val.shape}  "
          f"test={splits.X_test.shape}  test base rate={splits.y_test.mean():.3f}")

    # Every model is trained once up front at its production settings. The studies below either
    # reuse these scores or retrain deliberately, and keeping the reference run in one place stops
    # the studies drifting apart from the headline numbers.
    print("\n=== Reference run: all four models at production settings ===")
    runs = {}
    for name in (*SUPERVISED, "autoencoder"):
        t0 = time.time()
        runs[name] = run_model(name, splits.X_train, splits.y_train, splits.X_val,
                               splits.y_val, splits.X_test, seed=RANDOM_SEED)
        from sklearn.metrics import roc_auc_score
        print(f"  {name:<15} AUC={roc_auc_score(splits.y_test, runs[name].p_test):.4f}"
              f"  ({time.time() - t0:.0f}s)")

    results.setdefault("reference", {})
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    for name, run in runs.items():
        results["reference"][name] = {
            "auc_roc": float(roc_auc_score(splits.y_test, run.p_test)),
            "pr_auc": float(average_precision_score(splits.y_test, run.p_test)),
            "brier": float(brier_score_loss(splits.y_test, np.clip(run.p_test, 0, 1))),
            "n_parameters": int(run.n_parameters),
            "best_epoch": int(run.best_epoch),
        }
    if "best_params" in runs["xgboost"].extra:
        results["reference"]["xgboost"]["best_params"] = runs["xgboost"].extra["best_params"]

    # ---- 1. Feature-block ablation ---------------------------------------- #
    if wanted("ablation"):
        print("\n=== 1. Feature-block ablation ===")
        table = run_ablation(splits, models=SUPERVISED)
        deltas = ablation_deltas(table)
        results["ablation"] = {"raw": _records(table), "deltas": _records(deltas)}
        figs.plot_ablation(deltas)

    # ---- 2. Capacity and regularisation ladder ---------------------------- #
    if wanted("capacity"):
        print("\n=== 2. Capacity and regularisation ladder ===")
        ladders = run_full_study(splits)
        results["capacity"] = {
            name: {"table": _records(tbl), "summary": summarise_ladder(tbl)}
            for name, tbl in ladders.items()
        }
        figs.plot_capacity(ladders)

    # ---- 3. Learning curves ------------------------------------------------ #
    if wanted("learning_curve"):
        print("\n=== 3. Learning curves ===")
        curve = run_learning_curve(splits, models=SUPERVISED)
        results["learning_curve"] = _records(curve)
        figs.plot_learning_curve(curve)

    # ---- 4. Post-hoc calibration ------------------------------------------- #
    if wanted("calibration"):
        print("\n=== 4. Post-hoc calibration ===")
        cal = {name: calibration_report(runs[name].p_val, splits.y_val,
                                        runs[name].p_test, splits.y_test)
               for name in SUPERVISED}
        results["calibration"] = cal
        for name, rep in cal.items():
            print(f"  {name:<15} Brier raw={rep['brier_raw']:.4f}  "
                  f"Platt={rep.get('brier_platt', float('nan')):.4f}  "
                  f"isotonic={rep.get('brier_isotonic', float('nan')):.4f}")
        figs.plot_calibration_effect(cal)

    # ---- 5. Operating point from a stated review capacity ------------------ #
    if wanted("operating_point"):
        print("\n=== 5. Operating point at a fixed review capacity ===")
        probs = {n: runs[n].p_test for n in SUPERVISED}
        val_probs = {n: runs[n].p_val for n in SUPERVISED}
        ops = capacity_comparison(splits.y_test, probs, val_probs, AUDIT_CAPACITY)
        results["operating_point"] = {"capacity": AUDIT_CAPACITY, "models": ops}
        for name, op in ops.items():
            print(f"  {name:<15} thr={op['threshold']:.3f}  flagged={op['flagged_share']:.1%}"
                  f"  precision={op['precision']:.3f}  recall={op['recall']:.3f}")

    # ---- 6. Label sensitivity ---------------------------------------------- #
    if wanted("label_sensitivity"):
        print("\n=== 6. Label sensitivity ===")
        sens = run_label_sensitivity(feats, models=SUPERVISED)
        results["label_sensitivity"] = _records(sens)
        figs.plot_label_sensitivity(sens)

    # ---- 7. Hybrid supervised + anomaly score ------------------------------ #
    if wanted("hybrid"):
        print("\n=== 7. Hybrid score ===")
        best = max(SUPERVISED, key=lambda n: results["reference"][n]["auc_roc"])
        hyb = evaluate_hybrid(splits.y_test, runs[best].p_test, runs["autoencoder"].p_test,
                              runs[best].p_val, runs["autoencoder"].p_val, splits.y_val)
        hyb["supervised_model"] = best
        results["hybrid"] = hyb
        print(f"  base model={best}  best variant={hyb['verdict']['best_variant']}  "
              f"gain={hyb['verdict']['auc_gain_over_supervised']:+.4f}  "
              f"helps={hyb['verdict']['helps']}")

    # ---- 8. Stability: grouped bootstrap and per-slice performance --------- #
    if wanted("stability"):
        print("\n=== 8. Stability ===")
        groups = splits.test_frame["partner_jurisdiction"].values
        boots = {n: grouped_bootstrap_auc(splits.y_test, runs[n].p_test, groups)
                 for n in SUPERVISED}
        for name, b in boots.items():
            print(f"  {name:<15} AUC={b['point']:.3f}  "
                  f"95% CI [{b['lo']:.3f}, {b['hi']:.3f}]  over {b['n_groups']} jurisdictions")
        pairs = {
            f"xgboost_vs_{n}": paired_group_bootstrap(
                splits.y_test, runs["xgboost"].p_test, runs[n].p_test, groups)
            for n in ("mlp", "ft_transformer")
        }
        for name, pr in pairs.items():
            print(f"  {name:<24} mean diff={pr['mean_diff']:+.4f}  "
                  f"95% CI [{pr['lo']:+.4f}, {pr['hi']:+.4f}]  "
                  f"XGBoost ahead in {pr['p_a_better']:.1%} of resamples")

        best = max(SUPERVISED, key=lambda n: results["reference"][n]["auc_roc"])
        by_reporter = performance_by_slice(splits.test_frame, splits.y_test,
                                           runs[best].p_test, "reporting_jurisdiction")
        results["stability"] = {
            "bootstrap": boots,
            "paired": pairs,
            "by_reporting_jurisdiction": _records(by_reporter),
            "by_reporting_skipped": int(by_reporter.attrs.get("skipped_slices", 0)),
            "slice_model": best,
        }
        if len(by_reporter):
            print(f"  per-reporter AUC across {len(by_reporter)} jurisdictions: "
                  f"median={by_reporter['auc_roc'].median():.3f}  "
                  f"range {by_reporter['auc_roc'].min():.3f}-{by_reporter['auc_roc'].max():.3f}")
            figs.plot_by_reporter(by_reporter)

    _save(results)
    return results


if __name__ == "__main__":
    main()
