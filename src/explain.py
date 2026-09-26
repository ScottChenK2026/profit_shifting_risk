"""
explain.py
----------
Asks each trained model which features actually drove its predictions, using SHAP. SHAP gives every
feature a signed contribution for every prediction; averaging the absolute values across many cases
gives a global sense of which features matter most overall.

SHAP has several engines, and which one works depends on the model and, annoyingly, on the exact
library versions installed. XGBoost is a tree model, so it gets the exact TreeExplainer, with
XGBoost's own built-in version of the same calculation as a fallback. For the MLP three engines are
tried in order of cost: DeepExplainer, then GradientExplainer, then the slow but universal
KernelExplainer, which treats the model as a black box. With the versions this project pins, the
first two fail on this network, so the MLP figure comes from KernelExplainer, run on the first 100
rows of the sample against 50 background rows.

Either way the output is the same: a bar chart of mean absolute SHAP values, saved into
``outputs/figures``, and the ranked (feature, value) list handed back to the pipeline.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from config import FIGURE_DIR


def _bar_plot(mean_abs: np.ndarray, feature_names: list[str],
    title: str, filename: str) -> list[tuple[str, float]]:
    # Shared helper: draw the importance bars with the biggest at the top, and hand back the
    # (feature, value) pairs so the pipeline can log the top few.
    order = np.argsort(mean_abs)[::-1]
    names = [feature_names[i] for i in order]
    vals = mean_abs[order]
    plt.figure(figsize=(7, 4.5))
    plt.barh(range(len(names))[::-1], vals, color="#55A868")
    plt.yticks(range(len(names))[::-1], names)
    plt.xlabel("Mean |SHAP value|")
    plt.title(title)
    plt.tight_layout()
    path = FIGURE_DIR / filename
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[shap] figure -> {path}")
    return list(zip(names, [float(v) for v in vals]))


def explain_xgboost(model, X_sample: np.ndarray, feature_names: list[str],
                    filename: str = "shap_xgboost.png"
                    ) -> list[tuple[str, float]]:
    """Feature importance for XGBoost.

    The normal path is shap's TreeExplainer. Newer XGBoost models, though, store a ``base_score``
    field that some shap versions cannot parse, and it throws. When that happens I ask XGBoost
    itself for the contributions with ``pred_contribs=True``. It is literally the same Tree-SHAP
    algorithm, only computed inside XGBoost, so the same answer arrives by another door.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        sv = np.asarray(shap_values)
    except Exception as exc:       # pragma: no cover - version-dependent path
        # The version-mismatch fallback described above.
        print(f"[shap] TreeExplainer unavailable ({exc!r}); "
            "using XGBoost native pred_contribs.")
        import xgboost as xgb
        booster = model.get_booster()
        dmat = xgb.DMatrix(X_sample, feature_names=list(feature_names))
        contribs = booster.predict(dmat, pred_contribs=True)
        sv = np.asarray(contribs)[:, :-1]   # the last column is the bias term, so drop it

    mean_abs = np.abs(sv).mean(axis=0)
    return _bar_plot(mean_abs, feature_names,
        "SHAP global importance - XGBoost", filename)


def explain_mlp(model, X_background: np.ndarray, X_sample: np.ndarray,
                feature_names: list[str],
                filename: str = "shap_mlp.png") -> list[tuple[str, float]]:
    """Feature importance for the MLP. Tries three SHAP engines in order of cost - DeepExplainer,
    GradientExplainer, then the model-agnostic KernelExplainer - and uses the first that works
    with the installed shap and torch versions. On the pinned versions it is KernelExplainer. The
    background sample is the baseline SHAP compares each prediction against."""
    model.eval()
    bg = torch.from_numpy(X_background.astype(np.float32))
    sample_t = torch.from_numpy(X_sample.astype(np.float32))

    try:
        explainer = shap.DeepExplainer(model, bg)
        shap_values = explainer.shap_values(sample_t, check_additivity=False)
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values
        sv = np.asarray(sv)
        if sv.ndim == 3:           # the shape can come back as (rows, features, outputs)
            sv = sv[:, :, 0]       # there is only one output here, so take it
        mean_abs = np.abs(sv).mean(axis=0)
    except Exception as exc:       # pragma: no cover - version-dependent path
        # DeepExplainer reaches inside PyTorch's internals and breaks whenever those internals move;
        # on the version this project now pins it fails outright. Two fallbacks rather than one,
        # tried in order of how much they cost.
        print(f"[shap] DeepExplainer unavailable ({exc!r}); trying GradientExplainer.")
        try:
            # GradientExplainer uses expected gradients and only needs backpropagation, so it is
            # less exposed to version changes than DeepExplainer. It still fails on this MLP: the
            # network returns a one-dimensional output and GradientExplainer expects one column per
            # output, so in practice the run falls through to KernelExplainer below.
            explainer = shap.GradientExplainer(model, bg)
            sv = np.asarray(explainer.shap_values(sample_t))
            if sv.ndim == 3:
                sv = sv[:, :, 0]
            mean_abs = np.abs(sv).mean(axis=0)
        except Exception as exc2:
            # KernelExplainer is slower but treats the model as a black box, so it works whatever
            # the library versions are. It is the engine that produces the MLP figure, and the
            # sample sizes are trimmed to keep the runtime bearable.
            print(f"[shap] GradientExplainer unavailable ({exc2!r}); "
                "falling back to KernelExplainer.")

            # KernelExplainer needs a plain function from inputs to probabilities.
            def f(x: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    return model.predict_proba(
                        torch.from_numpy(x.astype(np.float32))).numpy()

            bg_small = shap.sample(X_background, min(50, len(X_background)),
                                   random_state=0)
            explainer = shap.KernelExplainer(f, bg_small)
            sv = np.asarray(explainer.shap_values(X_sample[:100], nsamples=100,
                                                  silent=True))
            if sv.ndim == 3:
                sv = sv[:, :, 0]
            mean_abs = np.abs(sv).mean(axis=0)

    return _bar_plot(mean_abs, feature_names,
                     "SHAP global importance - MLP", filename)
