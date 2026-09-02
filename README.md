# Detecting Profit Shifting Risk in Multinational Enterprises

A neural-network pipeline that scores jurisdiction-level Country-by-Country
Reporting (CbCR) observations for profit-shifting risk, built on the real
**OECD CbCR Table I** statistics. CM3070 Final Project, BSc Computer Science,
University of London (ML & Neural Networks specialisation, CM3015).

It trains and rigorously compares four models — a modern **FT-Transformer**
(attention for tabular data), a regularised **MLP**, an **autoencoder** anomaly
detector, and an **XGBoost** baseline — using an out-of-time evaluation,
calibration, the DeLong significance test, an audit-budget analysis, and SHAP
interpretability.

The headline contribution is the use of the OECD **business-activity
establishment mix** (holding, internal group finance, IP, dormant shells) as
features — a part of the data that ML work on profit shifting has largely
ignored, and which SHAP shows to be the most predictive signal.

---

## Quick start

```bash
# A conda environment is easiest (and avoids issues if the folder path
# contains spaces or punctuation, which can break python -m venv):
conda create -n psr python=3.11 -y && conda activate psr
pip install -r requirements.txt          # PyTorch: see note in requirements.txt

# 1. convert the raw OECD download into the tidy model table (see below)
python src/oecd_adapter.py data/OECD_CbCR_unfiltered_raw.csv

# 2. run the full pipeline (resumable)
python run_pipeline.py

# 3. tests
pytest tests/

# 4. score records with the best trained model
python src/cli.py --csv data/oecd_cbcr_wide.csv --top 20
```

`run_pipeline.py` is **resumable**: each model's test predictions are cached in
`outputs/models/<name>_probs.npy`, so if a run is interrupted, re-running
continues where it left off. Pass `--fresh` to retrain from scratch.

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
the same schema so it still runs end to end.

---

## Method highlights

**Leakage-aware label.** The target is the partner jurisdiction's membership of
a consolidated tax-haven list (`src/reference_data.py`) — external to, and
independent of, every feature. Jurisdiction identity is deliberately *excluded*
from the features, so the model cannot simply memorise which partners are
havens. The genuine question is: *do the reported economics and activity mix
alone reveal the known havens?*

**Out-of-time evaluation.** Train on 2016–2019, validate on 2020, test on 2021.
This is far stronger than a random split and removes cross-year leakage.

**Three feature blocks (18 features).** Profitability/tax ratios (margin,
related-party share, effective tax rate); substance ratios (profit/revenue/
capital per employee, per asset, per entity); and the novel business-activity
mix (holding/IP/internal-finance/dormant shares vs real-activity share); plus a
loss-shifting signal from the profit panels.

**Rigorous evaluation.** AUC-ROC, PR-AUC, F1/precision/recall, Brier
(calibration), DeLong test for AUC significance, precision@k (audit budget), and
SHAP global importance for the tree and neural models.

---

## Results (real OECD data, out-of-time 2021 test set)

| Model            | AUC-ROC | PR-AUC | F1    | Precision | Recall | Brier |
|------------------|---------|--------|-------|-----------|--------|-------|
| XGBoost          | 0.941   | 0.842  | 0.765 | 0.776     | 0.755  | 0.071 |
| FT-Transformer   | 0.922   | 0.780  | 0.687 | 0.604     | 0.795  | 0.102 |
| MLP              | 0.920   | 0.782  | 0.675 | 0.573     | 0.822  | 0.108 |
| Autoencoder*     | 0.767   | 0.432  | —     | —         | —      | 0.174 |

\* unsupervised; reported as a ranker (AUC/PR-AUC).

XGBoost is the strongest model, and DeLong's test confirms its AUC advantage
over both neural networks is statistically significant (p ≈ 1.5×10⁻⁶ for the MLP
and 1.7×10⁻⁵ for the FT-Transformer) — the expected outcome on structured
tabular data (Grinsztajn et al., 2022), reported honestly. SHAP shows the
**activity-mix features dominate** for every model. Reviewing the top 10% of
flagged 2021 cases gives 92.5% precision and captures ~49% of all havens. The
50 highest-risk observations are all known havens (Cayman, Luxembourg, Bermuda,
BVI, Hong Kong, …).

Numbers are reproducible under `RANDOM_SEED = 42`; minor variation across
library versions is expected.

---

## Project layout

```
profit_shifting_risk/
├── config.py                # paths, feature list, split + model config
├── run_pipeline.py          # resumable end-to-end pipeline
├── make_figures.py          # architecture + Gantt + code-listing images
├── build_report.py          # builds the report .docx from outputs/
├── toc_pages.json           # cached contents-page numbers (see below)
├── requirements.txt / README.md
├── data/                    # raw OECD export + wide table (+ synthetic cache)
├── outputs/                 # figures, metrics (JSON), trained models
└── src/
    ├── oecd_adapter.py      # raw OECD long CSV -> wide model table
    ├── reference_data.py    # tax-haven list + activity/measure code maps
    ├── data_generation.py   # real loader + synthetic fallback
    ├── features.py          # 3-block feature engineering + label
    ├── preprocessing.py     # temporal / random splits, leakage-safe scaling
    ├── eda.py               # EDA figures
    ├── evaluate.py          # metrics, DeLong, precision@k, calibration, plots
    ├── explain.py           # SHAP (TreeExplainer / DeepExplainer + fallbacks)
    ├── cli.py               # command-line risk scorer
    └── models/
        ├── __init__.py      # shared seed helper + per-epoch loss record
        ├── ft_transformer.py
        ├── mlp.py
        ├── autoencoder.py
        └── xgb_baseline.py
```

`data/` holds the raw OECD export, the wide model table the adapter builds from it, and the
synthetic cache. `outputs/` and the caches are regenerated, so neither is tracked in git.

All libraries are open-source; no proprietary or paid data are included.

---

## Rebuilding the report

The report is generated from the pipeline's own outputs, so the figures and
numbers in it cannot drift away from the code.

```bash
python make_figures.py            # architecture, Gantt, code listings,
                                  # and a refresh of the correlation heatmap
python build_report.py --remap    # builds the .docx, then re-reads the real
                                  # page numbers back into the contents list
```

`build_report.py` on its own is enough for a quick rebuild; `--remap` is only
needed when the pagination has changed. The contents list is built from
hyperlinked `PAGEREF` fields, so Word also corrects the page numbers itself
whenever fields are refreshed (Ctrl+A, then F9).

Per-chapter word counts are printed at the end of every build and checked
against the assignment's limits.
