"""
run_pipeline.py
---------------
The whole project, start to finish, in one runnable script. It loads the data, builds the features,
trains the four models, and evaluates them on a test set drawn from later years than the training
data - so the question being asked is really "does this generalise to next year?". Everything it
produces lands in ``outputs/``, ready to drop into the report.

The order of events:

    1. load the real OECD wide table, or fall back to synthetic, then features and label
    2. EDA figures
    3. out-of-time split: train 2016-19, validate 2020, test 2021
    4. train and evaluate XGBoost, MLP, FT-Transformer and the autoencoder
    5. comparative plots, calibration, audit budget (precision@k)
    6. DeLong significance tests against the XGBoost baseline
    7. SHAP interpretability
    8. a face-validity check against the known-haven list
    9. save everything the CLI needs to score new records

The whole thing is resumable, which matters because training four models takes a while. After each
model finishes, its test-set predictions are cached as outputs/models/<name>_probs.npy, and
re-running the script reloads anything already there and skips past it. So I can train one model
now and another later and still end up with a complete run. Pass --fresh, or delete the .npy files,
when a genuine retrain from scratch is wanted.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# config.py sits in the project root while the rest of the code lives in src/, and everything uses
# flat imports, so both directories need to be importable.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import (FEATURE_COLUMNS, METRIC_DIR, MODEL_DIR, RANDOM_SEED,  # noqa
                    TARGET_COLUMN)
import eda                                                   # noqa: E402
import evaluate as ev                                        # noqa: E402
from data_generation import load_cbcr                        # noqa: E402
from features import engineer_features, summarise_missing    # noqa: E402
from preprocessing import prepare_temporal_splits            # noqa: E402
from models.xgb_baseline import train_xgboost                # noqa: E402
from models.mlp import train_mlp                             # noqa: E402
from models.ft_transformer import train_ft_transformer       # noqa: E402
from models.autoencoder import train_autoencoder             # noqa: E402

# --fresh forces a full retrain; otherwise whatever is already cached gets reused.
FRESH = "--fresh" in sys.argv


def _probs_path(name):
    # Where a model's cached test-set predictions live, which is what makes resuming possible.
    return MODEL_DIR / f"{name}_probs.npy"


def main() -> dict:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    summary = {}

    print("\n=== 1-2. Data + features ===")
    raw = load_cbcr()
    feats = engineer_features(raw)
    summary["n_rows"] = int(len(feats))
    summary["class_balance"] = feats[TARGET_COLUMN].value_counts().to_dict()
    print("rows:", len(feats), "| class balance:", summary["class_balance"])

    # Which features have the most gaps. Worth knowing before trusting any of them, and worth
    # recording in the summary so the report can quote real numbers rather than an impression.
    miss = summarise_missing(feats).sort_values("missing_pct", ascending=False)
    summary["missing_pct_worst"] = miss["missing_pct"].head(5).to_dict()
    print("most incomplete features:", summary["missing_pct_worst"])

    # EDA only needs to run once, so a small flag file lets reruns skip it.
    if FRESH or not (METRIC_DIR / "eda_done.flag").exists():
        eda.plot_feature_distributions(feats)
        corr = eda.plot_correlation_heatmap(feats)
        eda.plot_class_balance(feats)
        summary["max_abs_offdiag_corr"] = float(
            corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max())
        (METRIC_DIR / "eda_done.flag").write_text("ok")

    # The out-of-time split: train on the earlier years, test on the latest. A random split would
    # let the model see the same companies in adjacent years and flatter the scores, so the test
    # set deliberately sits in the future.
    print("\n=== 3. Out-of-time split ===")
    s = prepare_temporal_splits(feats)
    print(f"kind={s.split_kind}  train={s.X_train.shape}  "
          f"val={s.X_val.shape}  test={s.X_test.shape}")
    print("positive rates:", round(float(s.y_train.mean()), 3),
          round(float(s.y_val.mean()), 3), round(float(s.y_test.mean()), 3))
    np.save(MODEL_DIR / "y_test.npy", s.y_test)
    summary["split"] = {"kind": s.split_kind,
                        "train": int(len(s.y_train)),
                        "val": int(len(s.y_val)), "test": int(len(s.y_test))}

    probs = {}

    # ---- XGBoost ---------------------------------------------------------- #
    name = "xgboost"
    if not FRESH and _probs_path(name).exists():
        probs[name] = np.load(_probs_path(name)); print(f"[skip] {name} cached")
    else:
        print("\n=== 4a. XGBoost ===")
        xgb, params = train_xgboost(s.X_train, s.y_train)
        probs[name] = xgb.predict_proba(s.X_test)[:, 1]
        np.save(_probs_path(name), probs[name])
        xgb.save_model(str(MODEL_DIR / "xgboost.json"))
        m = ev.classification_metrics(s.y_test, probs[name])
        ev.save_metrics({**m, "best_params": params}, name)
        print("best params:", params, "| AUC=%.3f" % m["auc_roc"])

    # ---- MLP -------------------------------------------------------------- #
    name = "mlp"
    if not FRESH and _probs_path(name).exists():
        probs[name] = np.load(_probs_path(name)); print(f"[skip] {name} cached")
    else:
        print("\n=== 4b. MLP ===")
        mlp, hist = train_mlp(s.X_train, s.y_train, s.X_val, s.y_val,
                              verbose=True)
        probs[name] = mlp.predict_proba(torch.from_numpy(s.X_test)).numpy()
        np.save(_probs_path(name), probs[name])
        torch.save(mlp.state_dict(), MODEL_DIR / "mlp_state.pt")
        ev.save_metrics(ev.classification_metrics(s.y_test, probs[name]), name)
        ev.plot_training_curve(hist.train_loss, hist.val_loss, hist.best_epoch,
                               name="MLP")

    # ---- FT-Transformer --------------------------------------------------- #
    name = "ft_transformer"
    if not FRESH and _probs_path(name).exists():
        probs[name] = np.load(_probs_path(name)); print(f"[skip] {name} cached")
    else:
        print("\n=== 4c. FT-Transformer ===")
        ft, hist = train_ft_transformer(s.X_train, s.y_train, s.X_val, s.y_val,
                                        verbose=True)
        probs[name] = ft.predict_proba(torch.from_numpy(s.X_test)).numpy()
        np.save(_probs_path(name), probs[name])
        torch.save(ft.state_dict(), MODEL_DIR / "ft_state.pt")
        ev.save_metrics(ev.classification_metrics(s.y_test, probs[name]), name)
        ev.plot_training_curve(hist.train_loss, hist.val_loss, hist.best_epoch,
                               name="FT-Transformer")

    # ---- Autoencoder ------------------------------------------------------ #
    name = "autoencoder"
    if not FRESH and _probs_path(name).exists():
        probs[name] = np.load(_probs_path(name)); print(f"[skip] {name} cached")
    else:
        print("\n=== 4d. Autoencoder ===")
        # The autoencoder only ever sees the ordinary, non-haven rows. It learns what a normal
        # record looks like, so anything it struggles to rebuild is unusual and more suspicious.
        ae, meta = train_autoencoder(s.X_train[s.y_train == 0],
                                     s.X_val[s.y_val == 0], verbose=True)
        err = ae.reconstruction_error(torch.from_numpy(s.X_test)).numpy()
        # Reconstruction error is not a probability, so it gets min-max scaled into [0, 1] purely
        # so it can be used as a ranking score. It is never read as a calibrated probability, which
        # is why the autoencoder is reported only as a ranker and left out of the calibration plot.
        probs[name] = (err - err.min()) / (err.max() - err.min() + 1e-9)
        np.save(_probs_path(name), probs[name])
        nt = float((meta["threshold"] - err.min())
                   / (err.max() - err.min() + 1e-9))
        ev.save_metrics({**ev.classification_metrics(
            s.y_test, probs[name], threshold=min(max(nt, 0), 0.999)),
            **{k: meta[k] for k in ("best_epoch", "threshold")}}, name)

    # ---- 5-6. Assembly: plots, audit budget, significance ------------------ #
    print("\n=== 5-6. Comparative evaluation ===")
    y_test = np.load(MODEL_DIR / "y_test.npy")
    pretty = {"ft_transformer": "FT-Transformer", "mlp": "MLP",
              "xgboost": "XGBoost", "autoencoder": "Autoencoder"}
    curves = {pretty[k]: (y_test, probs[k]) for k in probs}
    ev.plot_roc_curves(curves)
    ev.plot_pr_curves(curves)
    # Autoencoder scores are not real probabilities, so calibration would be meaningless for it.
    ev.plot_calibration({k: v for k, v in curves.items()
                         if k != "Autoencoder"})
    for k in ("mlp", "xgboost", "ft_transformer"):
        if k in probs:
            ev.plot_confusion(y_test, probs[k], pretty[k])

    metrics_all = {k: ev.classification_metrics(y_test, probs[k]) for k in probs}
    summary["metrics"] = metrics_all
    summary["audit_budget"] = {k: ev.precision_at_k(y_test, probs[k])
                               for k in probs}

    # XGBoost is the baseline. For each neural net, DeLong's test says whether its AUC is genuinely
    # different or just noise on this particular test set.
    sig = {}
    for k in ("ft_transformer", "mlp"):
        if k in probs and "xgboost" in probs:
            sig[f"{k}_vs_xgboost"] = ev.delong_roc_test(
                y_test, probs[k], probs["xgboost"])
    summary["delong_vs_xgboost"] = sig

    # ---- 7. SHAP ---------------------------------------------------------- #
    # SHAP can be brittle across shap, torch and xgboost version combinations, and it is not
    # essential to the run, so the whole block sits in a try/except. If it falls over we skip it
    # rather than losing the rest of the results.
    print("\n=== 7. SHAP ===")
    try:
        from explain import explain_mlp, explain_xgboost
        from xgboost import XGBClassifier
        if (MODEL_DIR / "xgboost.json").exists():
            xgb = XGBClassifier()
            xgb.load_model(str(MODEL_DIR / "xgboost.json"))
            summary["shap_xgboost_top"] = explain_xgboost(
                xgb, s.X_test[:400], s.feature_names)[:5]
        if (MODEL_DIR / "mlp_state.pt").exists():
            from models.mlp import ProfitShiftingMLP
            mlp = ProfitShiftingMLP(input_dim=len(FEATURE_COLUMNS))
            mlp.load_state_dict(torch.load(MODEL_DIR / "mlp_state.pt"))
            summary["shap_mlp_top"] = explain_mlp(
                mlp, s.X_train[:200], s.X_test[:300], s.feature_names)[:5]
    except Exception as exc:
        print(f"[shap] skipped: {exc!r}")

    # ---- 8. Face validity -------------------------------------------------- #
    # A sanity check that the scores mean something in the real world: take the best model, look at
    # the 50 records it scores highest, and see how many really are known havens and which
    # jurisdictions turn up. If the top of the list looked random, the whole thing would be suspect.
    print("\n=== 8. Face validity ===")
    best = max(metrics_all, key=lambda k: metrics_all[k]["auc_roc"])
    tf = s.test_frame.copy()
    tf["score"] = probs[best]
    top = tf.sort_values("score", ascending=False).head(50)
    summary["best_model"] = best
    summary["face_validity_top50_haven_rate"] = float(top[TARGET_COLUMN].mean())
    summary["face_validity_top_jurisdictions"] = (
        top["partner_jurisdiction"].value_counts().head(10).to_dict())
    print(f"best={best}  top-50 haven rate="
          f"{summary['face_validity_top50_haven_rate']:.1%}")

    # ---- 9. Persist what the CLI needs ------------------------------------- #
    # The CLI scores brand-new records, so it needs exactly the scaler and medians the models were
    # trained with. Saving them alongside the summary keeps later scoring consistent with training.
    with open(MODEL_DIR / "scaler.pkl", "wb") as fh:
        pickle.dump(s.scaler, fh)
    with open(MODEL_DIR / "medians.pkl", "wb") as fh:
        pickle.dump(s.medians, fh)
    with open(METRIC_DIR / "pipeline_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print("\n================ RESULTS (out-of-time test set) ================")
    print(f"{'Model':<16}{'AUC':>7}{'PR-AUC':>8}{'F1':>7}{'Prec':>7}"
          f"{'Rec':>7}{'Brier':>8}")
    for k in ("ft_transformer", "mlp", "xgboost", "autoencoder"):
        if k in metrics_all:
            m = metrics_all[k]
            print(f"{pretty[k]:<16}{m['auc_roc']:>7.3f}{m['pr_auc']:>8.3f}"
                  f"{m['f1']:>7.3f}{m['precision']:>7.3f}{m['recall']:>7.3f}"
                  f"{m['brier']:>8.3f}")
    print("================================================================")
    print(f"Summary -> {METRIC_DIR / 'pipeline_summary.json'}")
    return summary


if __name__ == "__main__":
    main()
