# Detecting Profit Shifting Risk in Multinational Enterprises

A neural-network pipeline that scores jurisdiction-level Country-by-Country
Reporting (CbCR) observations for profit-shifting risk, built on the real
**OECD CbCR Table I** statistics. CM3070 Final Project, BSc Computer Science,
University of London — CM3015 Machine Learning and Neural Networks template,
Project Idea 2 (§3.2), adapted from medical imaging to tabular tax data.

It trains and compares four models — a **FT-Transformer** (attention for tabular
data), a regularised **MLP**, an **autoencoder** anomaly detector and an
**XGBoost** baseline — under an out-of-time evaluation, then runs a set of
studies designed to test the result rather than decorate it: a feature-block
ablation, a capacity-and-regularisation ladder, learning curves, post-hoc
calibration, an operating threshold derived from review capacity, a
label-sensitivity study and grouped bootstrap intervals.

The headline contribution is the use of the OECD **business-activity
establishment mix** (holding, internal group finance, IP, dormant shells) as
model inputs — a part of the published data that machine-learning work on profit
shifting has not used, and which the ablation shows carries information no other
block supplies.

---

## Quick start

```bash
# A conda environment is easiest (and avoids issues if the folder path
# contains spaces or punctuation, which can break python -m venv):
conda create -n psr python=3.11 -y && conda activate psr
pip install -r requirements.txt          # PyTorch: see note in requirements.txt

# 1. convert the raw OECD download into the tidy model table (see below)
python src/oecd_adapter.py data/OECD_CbCR_unfiltered_raw.csv

# 2. train and evaluate the four models (resumable)
python run_pipeline.py

# 3. run the analysis studies (about 40 minutes on two CPU cores)
python run_experiments.py

# 4. tests
pytest tests/

# 5. score records with the best trained model
python src/cli.py --csv data/oecd_cbcr_wide.csv --top 20
```

`run_pipeline.py` is **resumable**: each model's test predictions are cached in
`outputs/models/<name>_probs.npy`, so an interrupted run continues where it left
off. Pass `--fresh` to retrain from scratch. `run_experiments.py` caches each
study separately in `outputs/metrics/experiments.json`; `--only NAME` reruns one
study and `--fresh` reruns the lot.

---

## Getting and converting the OECD data

1. On **data-explorer.oecd.org**, open *"Country-by-country reporting (CbCR) —
   Aggregate totals by jurisdiction — Corporate tax statistics"* (Table I).
2. **Download → "Unfiltered data in tabular text (CSV)"** — the complete table
   (~925k rows, all jurisdictions/years/measures). Save it under `data/`.
3. The export is in long (SDMX) format; convert it to the wide model table:

   ```bash
   python src/oecd_adapter.py data/your_export.csv
   ```

   The adapter streams the ~350 MB file, keeps the "Total (all sub-groups)"
   totals, drops regional aggregates, and pivots financial variables, the
   activity-establishment mix and the positive/negative profit panels into one
   row per (reporting jurisdiction, partner jurisdiction, year). It writes
   `data/oecd_cbcr_wide.csv`, which the pipeline then uses automatically.

If no real file is present, the pipeline falls back to a synthetic stand-in of
the same schema so it still runs end to end. The synthetic data exists to keep
the code testable and is never a source of reported results.

---

## Method

**Leakage-aware label.** The target is the partner jurisdiction's membership of
a consolidated tax-haven list (`src/reference_data.py`) — external to, and
independent of, every feature. Jurisdiction identity is deliberately *excluded*
from the features, so the model cannot memorise which partners are havens. A
unit test fails the build if any single feature becomes almost perfectly
correlated with the label.

**Out-of-time evaluation.** Train on 2016–2019, tune on 2020, test on 2021. All
four models are selected against that same 2020 year, so none of them sees a
data budget the others do not.

**Four feature blocks (18 features).** Profitability and tax ratios; substance
ratios (profit, revenue and capital per employee, per asset, per entity); the
business-activity mix (holding / IP / internal-finance / dormant shares against
real-activity share); and a one-feature loss-shifting block.

**Evaluation.** AUC-ROC, PR-AUC, F1/precision/recall, Brier, DeLong's test,
precision@k under a review budget, SHAP importance, a grouped bootstrap that
resamples whole jurisdictions rather than rows, and a per-jurisdiction
performance breakdown.

---

## Reproducing the reported numbers

Results in the report come from a single end-to-end run under `RANDOM_SEED = 42`
with the versions below. XGBoost reproduces exactly; the PyTorch models can move
in the third decimal across library versions.

| Package | Version used |
|---|---|
| Python | 3.11 |
| torch | 2.14.0 |
| xgboost | 3.2.0 |
| scikit-learn | 1.8.0 |
| shap | 0.51.0 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |

```bash
python run_pipeline.py --fresh     # headline comparison
python run_experiments.py --fresh  # the studies
```

---

## Project layout

```
profit_shifting_risk/
├── config.py                # paths, seed, feature list + blocks, split, model config
├── run_pipeline.py          # resumable end-to-end training and evaluation
├── run_experiments.py       # the analysis studies
├── requirements.txt / README.md
├── data/                    # raw OECD export + wide table (+ synthetic cache)
├── outputs/                 # figures, metrics (JSON), trained models
├── src/
│   ├── oecd_adapter.py      # raw OECD long CSV -> wide model table
│   ├── reference_data.py    # haven lists + activity/measure code maps
│   ├── data_generation.py   # real loader + synthetic fallback
│   ├── features.py          # feature engineering (4 blocks) + label
│   ├── preprocessing.py     # temporal/random splits, leakage-safe scaling
│   ├── eda.py               # EDA figures
│   ├── evaluate.py          # metrics, DeLong, precision@k, calibration, plots
│   ├── explain.py           # SHAP (Tree / Gradient / Kernel explainers)
│   ├── model_runner.py      # one entry point every study trains through
│   ├── ablation.py          # feature-block ablation + learning curves
│   ├── capacity_study.py    # scale-up-then-regularise workflow
│   ├── calibration.py       # post-hoc calibration + review thresholds
│   ├── label_sensitivity.py # alternative haven definitions
│   ├── hybrid.py            # supervised + anomaly score combination
│   ├── stability.py         # grouped bootstrap, per-slice performance
│   ├── experiment_figures.py# figures for the studies
│   ├── cli.py               # command-line risk scorer
│   └── models/
│       ├── __init__.py      # shared seed helper + per-epoch loss record
│       ├── ft_transformer.py
│       ├── mlp.py
│       ├── autoencoder.py
│       └── xgb_baseline.py
└── tests/                   # 35 unit tests, including the leakage guard
```

`data/` holds the raw OECD export, the wide table the adapter builds from it and
the synthetic cache. `outputs/` and the caches are regenerated, so neither is
tracked in git.

Scripts that exist only to produce the report, its figures or the demo slides
are **not part of this repository**; they live in a separate
`non_submission_tooling/` folder outside it. Nothing here imports from or
depends on them, so the pipeline runs standalone from a fresh clone.

All libraries are open-source; no proprietary or paid data are included.
