"""
make_figures.py
---------------
Builds the figures the report needs that the modelling pipeline does not
produce by itself:

  * architecture.png   - the system block diagram
  * gantt.png          - the work plan, with tasks and milestones
  * listing_*.png      - the code extracts, rendered as images
  * correlation_heatmap.png (refreshed via src/eda.py, so the readable
    lower-triangle version replaces the cramped one)

Plain matplotlib so these sit alongside the pipeline's own figures rather than
looking like they came from a different document.

Layout note: block heights are DERIVED from the number of text lines, so text
can never spill outside a box or collide with its title.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_ROOT = Path(__file__).resolve().parent
# Write into outputs/figures when run from inside profit_shifting_risk/,
# otherwise into a local figures/ folder.
OUT = (_ROOT / "outputs" / "figures") if (_ROOT / "outputs" / "figures").is_dir() \
    else (_ROOT / "figures")
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#1F3864"
BLUE = "#2E5496"
LIGHT = "#DCE4F2"
GREY = "#444444"
BAND = "#9BA9C4"
ACCENT = "#B03A36"
ACCENT_BG = "#F7E7E6"

PAD = 2.2          # inner padding, top and bottom
TITLE_H = 4.0      # space the bold title occupies
TITLE_GAP = 3.0    # gap between title and first body line
LINE_DY = 4.0      # gap between consecutive body lines
TITLE_FS = 9.0
LINE_FS = 7.3


def box_height(n_lines: int) -> float:
    """How tall a box must be to hold a title plus n_lines of body text."""
    if n_lines == 0:
        return PAD * 2 + TITLE_H
    return PAD * 2 + TITLE_H + TITLE_GAP + (n_lines - 1) * LINE_DY


# --------------------------------------------------------------------------- #
# 1. Architecture diagram
# --------------------------------------------------------------------------- #
def architecture() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 8.6))
    ax.set_xlim(0, 100)
    ax.axis("off")

    def box(x, top, w, title, lines=(), fc=LIGHT, ec=BLUE, tc=NAVY, h=None):
        lines = list(lines)
        h = h if h is not None else box_height(len(lines))
        y = top - h
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.1",
            facecolor=fc, edgecolor=ec, linewidth=1.25))
        cx = x + w / 2
        ty = top - PAD
        ax.text(cx, ty, title, ha="center", va="top", fontsize=TITLE_FS,
                fontweight="bold", color=tc)
        ly = ty - TITLE_H - TITLE_GAP + 1.2
        for i, ln in enumerate(lines):
            ax.text(cx, ly - i * LINE_DY, ln, ha="center", va="center",
                    fontsize=LINE_FS, color=GREY)
        return y

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
            linewidth=1.2, color=BLUE, shrinkA=0, shrinkB=0))

    def band(y, text):
        ax.text(0.0, y, text, ha="left", va="center", fontsize=7.2,
                fontweight="bold", color=BAND)

    ax.text(53, 99,
            "Out-of-time split:  train 2016–2019  ·  validate 2020  ·  test 2021",
            ha="center", va="top", fontsize=8.6, color=ACCENT, fontweight="bold")
    ax.text(53, 95.2,
            "the scaler and the imputation medians are fitted on the training years only",
            ha="center", va="top", fontsize=7.8, color=ACCENT, style="italic")

    TOP1 = 91.0
    h1 = box_height(3)
    box(12, TOP1, 25, "Raw OECD CbCR export",
        ["Table I, long SDMX format", "924,811 rows", "about 350 MB"], h=h1)
    box(41, TOP1, 25, "oecd_adapter.py",
        ["reads 200k-row chunks", "keeps totals, drops", "regional aggregates"], h=h1)
    box(70, TOP1, 25, "Model table",
        ["14,137 rows", "56 reporting × 228 partner", "years 2016–2021"], h=h1)
    band(TOP1 - h1 / 2, "DATA")
    arrow(37, TOP1 - h1 / 2, 41, TOP1 - h1 / 2)
    arrow(66, TOP1 - h1 / 2, 70, TOP1 - h1 / 2)

    BOT1 = TOP1 - h1
    ax.text(53, BOT1 - 3.0,
            "features.py  ·  18 engineered features  ·  the jurisdiction name is left out",
            ha="center", va="center", fontsize=8.0, color=NAVY, style="italic")
    arrow(82.5, BOT1, 82.5, BOT1 - 1.4)

    TOP2 = BOT1 - 6.5
    h2 = box_height(3)
    box(12, TOP2, 19, "Profitability / tax",
        ["profit margin", "related-party share", "tax rate paid, accrued"], h=h2)
    box(33.5, TOP2, 19, "Economic substance",
        ["profit, revenue and", "capital per employee,", "per asset, per entity"], h=h2)
    box(55, TOP2, 21, "Business-activity mix",
        ["holding · IP · internal", "finance · dormant, against",
         "real-activity share"], fc=ACCENT_BG, ec=ACCENT, tc=ACCENT, h=h2)
    box(78.5, TOP2, 16.5, "Loss shifting",
        ["how much loss sits", "here rather than", "elsewhere"], h=h2)
    band(TOP2 - h2 / 2, "FEATURES")

    BOT2 = TOP2 - h2
    ax.text(65.5, BOT2 - 2.4, "the block earlier work has not used",
            ha="center", va="center", fontsize=7.2, color=ACCENT, style="italic")

    TOP3 = BOT2 - 8.5
    h3 = box_height(3)
    for x in (21.5, 43, 65.5, 86.75):
        arrow(x, BOT2, x, TOP3)
    box(12, TOP3, 19, "FT-Transformer",
        ["turns each feature into", "a token, then attention", "main neural model"], h=h3)
    box(33.5, TOP3, 19, "MLP",
        ["128–64–32 with dropout,", "class weighting and", "early stopping"], h=h3)
    box(55, TOP3, 21, "Autoencoder",
        ["trained on non-havens only", "rebuild error is a score",
         "that ignores the label"], h=h3)
    box(78.5, TOP3, 16.5, "XGBoost",
        ["grid-searched", "gradient boosting —", "the model to beat"], h=h3)
    band(TOP3 - h3 / 2, "MODELS")

    BOT3 = TOP3 - h3
    TOP4 = BOT3 - 10.5
    h4 = box_height(4)
    BUS = BOT3 - 5.0
    for x in (21.5, 43, 65.5, 86.75):
        ax.plot([x, x], [BOT3, BUS], color=BLUE, linewidth=1.2,
                solid_capstyle="butt")
    ax.plot([21.5, 86.75], [BUS, BUS], color=BLUE, linewidth=1.2)
    for x in (24, 53, 82):
        arrow(x, BUS, x, TOP4)
    box(12, TOP4, 24, "Evaluation",
        ["AUC · PR-AUC · Brier", "DeLong significance test", "calibration curves",
         "precision at top 5/10/20%"], h=h4)
    box(41, TOP4, 24, "Interpretability",
        ["SHAP feature importance", "for XGBoost and MLP", "check the top-ranked",
         "places are recognisable"], h=h4)
    box(70, TOP4, 24, "Delivery",
        ["cli.py risk scorer", "17 pytest unit tests", "resumable run_pipeline.py",
         "fixed seed, runs on CPU"], h=h4)
    band(TOP4 - h4 / 2, "OUTPUTS")

    ax.set_ylim(TOP4 - h4 - 2.5, 100)
    fig.savefig(OUT / "architecture.png", dpi=170, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", OUT / "architecture.png")


# --------------------------------------------------------------------------- #
# 2. Work plan Gantt
# --------------------------------------------------------------------------- #
def gantt() -> None:
    """Timeline. Deliberately labelled by month rather than by an exact date,
    so the figure does not go stale depending on the day of submission."""
    D = dt.date
    tasks = [
        ("Project concept and proposal",            D(2026, 5, 4),  D(2026, 5, 17), "done"),
        ("Reading and literature review",           D(2026, 5, 11), D(2026, 6, 14), "done"),
        ("OECD download and streaming adapter",     D(2026, 5, 25), D(2026, 6, 14), "done"),
        ("Feature engineering and EDA",             D(2026, 6, 8),  D(2026, 6, 21), "done"),
        ("Four models built and evaluated",         D(2026, 6, 15), D(2026, 6, 28), "done"),
        ("Interim report",                          D(2026, 6, 22), D(2026, 6, 30), "done"),
        ("M1  Interim report submitted",            D(2026, 6, 30), D(2026, 6, 30), "milestone"),
        ("Report consolidation and rewrite",        D(2026, 8, 3),  D(2026, 8, 18), "active"),
        ("M2  Full report submitted",               D(2026, 8, 18), D(2026, 8, 18), "milestone"),
        ("Label-sensitivity study",                 D(2026, 8, 19), D(2026, 8, 31), "plan"),
        ("Autoencoder and supervised ensemble",     D(2026, 8, 24), D(2026, 9, 6),  "plan"),
        ("Threshold tied to audit capacity",        D(2026, 9, 1),  D(2026, 9, 10), "plan"),
        ("Exam revision",                           D(2026, 8, 24), D(2026, 9, 14), "plan"),
        ("M3  Written exam",                        D(2026, 9, 14), D(2026, 9, 14), "milestone"),
        ("Documentation and test coverage",         D(2026, 9, 7),  D(2026, 9, 18), "plan"),
        ("Final report write-up",                   D(2026, 9, 7),  D(2026, 9, 26), "plan"),
        ("M4  Final report due",                    D(2026, 9, 28), D(2026, 9, 28), "milestone"),
    ]
    colours = {"done": "#2E7D32", "active": "#E0A008", "plan": "#8FA0B5",
               "milestone": "#B03A36"}
    labels = {"done": "Completed", "active": "In progress",
              "plan": "Planned", "milestone": "Milestone / deadline"}

    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    n = len(tasks)
    for i, (name, s, e, status) in enumerate(tasks):
        y = n - i
        if status == "milestone":
            ax.barh(y, 2.4, left=s, height=0.68, color=colours[status])
        else:
            ax.barh(y, (e - s).days, left=s, height=0.56, color=colours[status])

    # Month-level marker rather than a specific day.
    marker = D(2026, 8, 18)
    ax.axvline(marker, color=NAVY, ls="--", lw=1.4, alpha=0.9)
    ax.text(marker, n + 0.9, "position at time of writing (mid-August)",
            ha="center", va="bottom", fontsize=8.5, color=NAVY)

    ax.set_yticks(range(1, n + 1))
    ax.set_yticklabels([t[0] for t in reversed(tasks)], fontsize=8.6)
    for lbl, spec in zip(ax.get_yticklabels(), list(reversed(tasks))):
        if spec[3] == "milestone":
            lbl.set_color(ACCENT)
            lbl.set_fontweight("bold")

    ax.set_xlim(D(2026, 4, 27), D(2026, 10, 6))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0.3, n + 1.8)
    ax.set_title("Project timeline and milestones (May – September 2026)",
                 fontsize=12.5, color=NAVY, pad=16)

    order = ("done", "active", "plan", "milestone")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[k]) for k in order]
    ax.legend(handles, [labels[k] for k in order], loc="upper center",
              bbox_to_anchor=(0.5, -0.075), ncol=4, fontsize=8.6, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "gantt.png", dpi=170, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", OUT / "gantt.png")


# --------------------------------------------------------------------------- #
# 3. Code listings, rendered as images
# --------------------------------------------------------------------------- #
LISTINGS: dict[str, str] = {
    "listing_adapter": '''for chunk in pd.read_csv(raw_path, usecols=USECOLS, chunksize=CHUNK,
                         dtype=str, low_memory=False):
    m = (
        (chunk["AGGREGATION_TYPE"] == "TOTAL")        # totals only
        & (chunk["STATISTICAL_OPERATION"] == "_Z")    # headline series
        & (chunk["MEASURE"].isin(WANTED_MEASURES | {"PROFIT"}))
        & (chunk["COUNTERPART_AREA"].map(is_real_jurisdiction))
    )
    keep.append(chunk[m])
''',
    "listing_features": '''def _safe_div(num, den):
    return num / den.replace(0, np.nan)      # 0-division -> NaN, not inf

def _slog(x):
    """Log that keeps the sign, so a loss stays distinguishable."""
    return np.sign(x) * np.log1p(x.abs())

# the tax rate means nothing on a loss, so blank it rather than letting
# "tax as a fraction of a negative number" into the model
etr = _safe_div(out["income_tax_paid"], out["profit_before_tax"])
etr[out["profit_before_tax"] <= 0] = np.nan
out["effective_tax_rate"] = etr.clip(0, 1)
''',
    "listing_tokenizer": '''class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias   = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x):                      # x: (batch, n_features)
        return x.unsqueeze(-1) * self.weight + self.bias   # -> (B, F, d_token)

# in FTTransformer.forward:
tokens = self.tokenizer(x)                     # one token per feature
cls    = self.cls.expand(x.size(0), -1, -1)    # learnable summary token
seq    = torch.cat([cls, tokens], dim=1)       # [CLS] goes at the front
z      = self.encoder(seq)                     # attention mixes them
return self.head(z[:, 0]).squeeze(-1)          # classify the [CLS] state
''',
    "listing_leakage": '''def test_no_single_feature_is_circular_with_label():
    # The big one: guard against label leakage. If any single feature were
    # nearly perfectly correlated with the target, the model would just be
    # reading the answer off that column and the whole exercise would be
    # meaningless.
    feats = impute_features(_feats())[0]
    for col in FEATURE_COLUMNS:
        c = np.corrcoef(feats[col], feats[TARGET_COLUMN])[0, 1]
        assert abs(c) < 0.95, f"{col} too correlated with label ({c:.2f})"
''',
}


def listings() -> None:
    """Render each code extract to a PNG with syntax highlighting.

    Done as images so the code reads as a figure rather than as body prose.
    Rendered large and then placed small in the document, so it stays sharp
    if the reader zooms in.
    """
    try:
        from pygments import highlight
        from pygments.formatters import ImageFormatter
        from pygments.lexers import PythonLexer
    except ImportError:
        print("pygments not installed - skipping listing images")
        return

    for name, src in LISTINGS.items():
        formatter = ImageFormatter(
            font_name="DejaVu Sans Mono",
            font_size=30,                 # large; scaled down on the page
            line_numbers=False,
            style="friendly",
            image_pad=18,
            line_pad=6,
        )
        png = highlight(src.rstrip("\n"), PythonLexer(), formatter)
        (OUT / f"{name}.png").write_bytes(png)
        print("wrote", OUT / f"{name}.png")


# --------------------------------------------------------------------------- #
# 4. Refresh the EDA figures through the project's own code
# --------------------------------------------------------------------------- #
def refresh_eda() -> None:
    """Regenerate the correlation heatmap (and the distribution grid) using
    src/eda.py, so the report figure and the codebase never drift apart."""
    root = _ROOT
    if not (root / "src").is_dir():
        print("src/ not found - skipping EDA refresh")
        return
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    try:
        import pandas as pd
        from config import DATA_DIR, REAL_DATA_FILENAME
        from reference_data import TAX_HAVENS
        from features import engineer_features
        from eda import plot_correlation_heatmap
    except Exception as exc:                      # pragma: no cover
        print(f"EDA refresh skipped: {exc!r}")
        return
    path = DATA_DIR / REAL_DATA_FILENAME
    if not path.exists():
        print(f"EDA refresh skipped: {path} not found")
        return
    df = pd.read_csv(path)
    df["is_known_haven"] = df["partner_jurisdiction"].isin(TAX_HAVENS).astype(int)
    feats = engineer_features(df)
    plot_correlation_heatmap(feats)


if __name__ == "__main__":
    architecture()
    gantt()
    listings()
    refresh_eda()
