# Detecting Profit Shifting Risk in Multinational Enterprises

[![tests](https://github.com/ScottChenK2026/profit_shifting_risk/actions/workflows/ci.yml/badge.svg)](https://github.com/ScottChenK2026/profit_shifting_risk/actions/workflows/ci.yml)

A neural-network pipeline that scores jurisdiction-level Country-by-Country
Reporting (CbCR) observations for profit-shifting risk, built on the real
**OECD CbCR Table I** statistics. CM3070 Final Project, BSc Computer Science,
University of London. It follows the CM3015 Machine Learning and Neural Networks
template, Project Idea 2 (§3.2), adapted from medical imaging to tabular tax data.

It trains and compares four models, a **FT-Transformer** (attention for tabular
data), a regularised **MLP**, an **autoencoder** anomaly detector and an
**XGBoost** baseline, under an out-of-time evaluation. It then runs a set of
studies designed to test the result rather than decorate it: a feature-block
ablation, a capacity-and-regularisation ladder, learning curves, post-hoc
calibration, an operating threshold derived from review capacity, a
label-sensitivity study and grouped bootstrap intervals.

The headline contribution is the use of the OECD **business-activity
establishment mix** (holding, internal group finance, IP, dormant shells) as
model inputs. This part of the published data has not been used by
machine-learning work on profit shifting, and the ablation shows it carries
information no other block supplies.

## Contents

1. [What you need](#1-what-you-need)
2. [Set up](#2-set-up)
3. [Run the project](#3-run-the-project)
4. [What the run produces](#4-what-the-run-produces)
5. [Reproducing the reported numbers](#5-reproducing-the-reported-numbers)
6. [Troubleshooting](#6-troubleshooting)
7. [Getting the raw OECD data (optional)](#7-getting-the-raw-oecd-data-optional)
8. [Method](#8-method)
9. [Project layout](#9-project-layout)

---

## 1. What you need

| | |
| --- | --- |
| **Operating system** | Windows 10/11, macOS on Apple silicon (M1 or later), or Linux. PyTorch no longer publishes packages for Intel Macs. |
| **Python** | 3.11, installed through [Miniforge](https://github.com/conda-forge/miniforge) or Anaconda (recommended) |
| **Disk space** | About 2 GB for the environment |
| **Data** | Nothing to download. The converted OECD table, `data/oecd_cbcr_wide.csv`, is included in the repository. |
| **Time** | Headline pipeline: a few minutes on a laptop CPU. Analysis studies: about 40 minutes on two CPU cores. No GPU needed. |

Every push is tested on Windows, macOS and Linux by the GitHub Actions workflow
in `.github/workflows/ci.yml` (the badge at the top).

---

## 2. Set up

Run these once. On Windows use the **Miniforge Prompt** (or Anaconda Prompt);
on macOS and Linux use a normal terminal.

**Step 1. Get the code**

```bash
git clone https://github.com/ScottChenK2026/profit_shifting_risk.git
cd profit_shifting_risk
```

**Step 2. Create and activate an environment**

```bash
conda create -n psr python=3.11 -y
conda activate psr
```

A conda environment is easiest, and it avoids problems if the folder path
contains spaces or punctuation, which can break `python -m venv`.

**Step 3. Install the pinned requirements**

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions the reported results were produced
with, so every machine gets the same libraries.

**Step 4 (macOS only). Set up OpenMP for XGBoost**

XGBoost needs the OpenMP runtime, which pip does not install. PyTorch ships its
own copy of the same runtime, and two copies in one process crash it with a
"segmentation fault". These commands install the runtime and point PyTorch at
the same copy. The original PyTorch file is kept as `libomp.dylib.bak`.

```bash
conda install -c conda-forge llvm-openmp -y
TORCH_LIB="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
mv "$TORCH_LIB/libomp.dylib" "$TORCH_LIB/libomp.dylib.bak"
ln -s "$CONDA_PREFIX/lib/libomp.dylib" "$TORCH_LIB/libomp.dylib"
```

If you use Homebrew's `libomp` instead (`brew install libomp`), link to
`"$(brew --prefix libomp)/lib/libomp.dylib"` in the last line. Reinstalling
PyTorch undoes this step, so repeat it after any reinstall.

**Step 5. Check the installation**

```bash
python -c "import torch, xgboost, shap; print('torch', torch.__version__, '| xgboost', xgboost.__version__)"
pytest tests/ -q
```

Expected: `torch 2.14.0 | xgboost 3.2.0`, then `35 passed`.

---

## 3. Run the project

Run everything from the `profit_shifting_risk` folder with the `psr`
environment active.

**Step 1. Train and evaluate the four models**

```bash
python run_pipeline.py
```

This builds the features, splits the data by year (train 2016–2019, tune 2020,
test 2021), trains the four models, evaluates them, and saves everything to
`outputs/`. It ends with a results table like the one in
[section 5](#5-reproducing-the-reported-numbers).

The run is **resumable**: each model's test predictions are cached in
`outputs/models/<name>_probs.npy`, so an interrupted run continues where it
stopped. Add `--fresh` to retrain everything from scratch. Use `--fresh` too if
an `outputs/` folder copied from another machine is already present, so the
results are your own.

**Step 2. Run the analysis studies**

```bash
python run_experiments.py
```

This runs the eight studies: `ablation`, `capacity`, `learning_curve`,
`calibration`, `operating_point`, `label_sensitivity`, `hybrid` and `stability`.
Each study is cached separately in `outputs/metrics/experiments.json`.
`--only NAME` reruns one study, for example `--only ablation`, and `--fresh`
reruns all of them.

**Step 3. Score records with the command-line tool**

The scorer loads whichever model the pipeline recorded as best (XGBoost, unless
a rerun changes that) and returns a risk score between 0 and 1 with a
HIGH / MEDIUM / LOW band. Step 1 must have been run first.

Rank every row of the included table and show the 20 highest-risk rows:

```bash
python src/cli.py --csv data/oecd_cbcr_wide.csv --top 20
```

Every row shown should be a known haven partner with a score that rounds to 1.0.
Because so many rows tie at that rounded score, the order of the top rows can
differ between machines.

Score one record typed in from raw CbCR figures. These are made-up numbers for a
haven-like filing: five employees, a profit of 600 on revenue of 1,000, 12 paid
in tax, and mostly holding and internal-finance establishments.

```bash
python src/cli.py --total-revenues 1000 --related-party-revenues 850 --profit-before-tax 600 --income-tax-paid 12 --num-employees 5 --num-entities 40 --tangible-assets 30 --stated-capital 10 --holding-establishments 40 --igf-establishments 15 --real-establishments 2
```

Expected output (the last digits can differ slightly between machines):

```
[cli] scoring with: xgboost
Risk score: 0.9991  (band: HIGH)
```

Add `--model mlp` (or `ft_transformer`, `xgboost`) to force a particular model,
and `python src/cli.py --help` lists every input.

---

## 4. What the run produces

Everything is written to `outputs/`, which is not tracked in git and is rebuilt
by the steps above.

| Folder | Contents |
| --- | --- |
| `outputs/metrics/` | `pipeline_summary.json` (headline results, review-budget precision, DeLong tests, SHAP rankings), one JSON file per model, and `experiments.json` (all study results) |
| `outputs/figures/` | ROC, precision-recall, calibration and confusion-matrix plots, training curves, SHAP charts, the exploratory data plots, and one figure per study |
| `outputs/models/` | Trained models, cached test predictions, and the scaler and fill-in values the command-line tool needs |

---

## 5. Reproducing the reported numbers

The results in the report come from one end-to-end run with `RANDOM_SEED = 42`
and the package versions below, all pinned in `requirements.txt`.

| Package | Version | Package | Version |
| --- | --- | --- | --- |
| Python | 3.11 | numpy | 2.4.4 |
| torch | 2.14.0 | pandas | 3.0.2 |
| xgboost | 3.2.0 | scipy | 1.17.1 |
| scikit-learn | 1.8.0 | matplotlib | 3.10.9 |
| shap | 0.51.0 | seaborn | 0.13.2 |

To reproduce them from scratch:

```bash
python run_pipeline.py --fresh     # headline comparison
python run_experiments.py --fresh  # the studies
```

The headline table in the report (test year 2021):

| Model | AUC | PR-AUC |
| --- | --- | --- |
| XGBoost | 0.948 | 0.858 |
| FT-Transformer | 0.920 | 0.764 |
| MLP | 0.919 | 0.780 |
| Autoencoder | 0.769 | 0.438 |

On Linux, a clean clone reproduces this table exactly. On Windows and macOS the
figures can move by about 0.001 to 0.004, because the underlying maths
libraries and thread counts differ between operating systems and processors.
For example, an Apple-silicon Mac gives an XGBoost AUC of 0.949. The comparisons
and conclusions do not change. The CI workflow prints this comparison for each
operating system and fails if any AUC moves further than a small tolerance.

---

## 6. Troubleshooting

| Problem | Cause and fix |
| --- | --- |
| `XGBoostError: ... libxgboost.dylib could not be loaded` (macOS) | The OpenMP runtime is missing. Do [Set up, step 4](#2-set-up). |
| `segmentation fault` when running the pipeline, the studies or the scorer (macOS) | Two copies of OpenMP are loaded. Do the last three commands of [Set up, step 4](#2-set-up). |
| `InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from version ...` | The saved scaler came from a different scikit-learn version. Run `pip install -r requirements.txt`, then `python run_pipeline.py --fresh`. |
| `run_pipeline.py` skips training and prints `[skip] ... cached` | Cached predictions already exist in `outputs/`. Add `--fresh` to retrain. |
| `import torch` fails with a DLL error (Windows) | Install the Microsoft Visual C++ Redistributable (x64) from Microsoft's website, then try again. |
| `pip` cannot find `torch==2.14.0` or `numpy==2.4.4` | The environment is not Python 3.11 or later, or the machine is an Intel Mac or 32-bit. Recreate the environment with `python=3.11` on Windows, Linux or an Apple-silicon Mac. |

---

## 7. Getting the raw OECD data (optional)

Only needed to rebuild `data/oecd_cbcr_wide.csv` from the original download.

1. On **data-explorer.oecd.org**, open _"Country-by-country reporting (CbCR) —
   Aggregate totals by jurisdiction — Corporate tax statistics"_ (Table I).
2. Choose **Download → "Unfiltered data in tabular text (CSV)"**, which gives the
   complete table (about 925,000 rows, all jurisdictions, years and measures).
   Save it under `data/`.
3. The export is in long (SDMX) format. Convert it to the wide model table:

   ```bash
   python src/oecd_adapter.py data/OECD_CbCR_unfiltered_raw.csv
   ```

   The adapter streams the file of about 350 MB, keeps the "Total (all
   sub-groups)" figures, drops regional aggregates, and pivots the financial
   variables, the activity-establishment mix and the profit and loss panels into
   one row per reporting jurisdiction, partner jurisdiction and year. It writes
   `data/oecd_cbcr_wide.csv`, which the pipeline then uses automatically.
   Expected output ends with `wrote 14,137 wide rows x 32 cols`.

If no real table is present, the pipeline falls back to a synthetic stand-in with
the same columns so it still runs end to end. The synthetic data exists only to
keep the code testable and is never a source of reported results.

---

## 8. Method

**Leakage-aware label.** The target is whether the partner jurisdiction is on a
combined tax-haven list (`src/reference_data.py`), which is decided
independently of every feature. Jurisdiction identity is deliberately _excluded_
from the features, so the model cannot memorise which partners are havens. A
unit test fails if any single feature becomes almost perfectly correlated with
the label.

**Out-of-time evaluation.** Train on 2016–2019, tune on 2020, test on 2021. All
four models are selected against that same 2020 year, so none of them sees data
the others do not.

**Four feature blocks (18 features).** Profitability and tax ratios; substance
ratios (profit, revenue and capital per employee, per asset, per entity); the
business-activity mix (holding, IP, internal-finance and dormant shares against
the real-activity share); and a one-feature loss-shifting block.

**Evaluation.** AUC-ROC, PR-AUC, F1, precision and recall, Brier score, DeLong's
test, precision within a review budget, SHAP importance, a grouped bootstrap
that resamples whole jurisdictions rather than rows, and a per-jurisdiction
performance breakdown.

---

## 9. Project layout

```
profit_shifting_risk/
├── .github/workflows/ci.yml  # tests + full run on Windows, macOS and Linux
├── config.py                 # paths, seed, feature list + blocks, split, model config
├── run_pipeline.py           # resumable end-to-end training and evaluation
├── run_experiments.py        # the analysis studies
├── requirements.txt          # pinned package versions
├── data/                     # converted OECD table (+ synthetic cache; raw export if downloaded)
├── outputs/                  # figures, metrics (JSON), trained models - created by the runs
├── src/
│   ├── oecd_adapter.py       # raw OECD long CSV -> wide model table
│   ├── reference_data.py     # haven lists + activity/measure code maps
│   ├── data_generation.py    # real loader + synthetic fallback
│   ├── features.py           # feature engineering (4 blocks) + label
│   ├── preprocessing.py      # temporal/random splits, leakage-safe scaling
│   ├── eda.py                # exploratory data figures
│   ├── evaluate.py           # metrics, DeLong, precision@k, calibration, plots
│   ├── explain.py            # SHAP (TreeExplainer; KernelExplainer for the MLP)
│   ├── model_runner.py       # one entry point every study trains through
│   ├── ablation.py           # feature-block ablation + learning curves
│   ├── capacity_study.py     # scale-up-then-regularise workflow
│   ├── calibration.py        # post-hoc calibration + review thresholds
│   ├── label_sensitivity.py  # alternative haven definitions
│   ├── hybrid.py             # supervised + anomaly score combination
│   ├── stability.py          # grouped bootstrap, per-slice performance
│   ├── experiment_figures.py # figures for the studies
│   ├── cli.py                # command-line risk scorer
│   └── models/
│       ├── __init__.py       # shared seed helper + per-epoch loss record
│       ├── ft_transformer.py
│       ├── mlp.py
│       ├── autoencoder.py
│       └── xgb_baseline.py
└── tests/                    # 35 unit tests, including the leakage guard
```

The raw OECD export (about 350 MB) and `outputs/` are not tracked in git; both
can be regenerated. Scripts that exist only to produce the report, its figures
or the demo slides are **not part of this repository**. Nothing here depends on
them, so the pipeline runs standalone from a fresh clone.

All libraries are open-source, and no proprietary or paid data are included.
