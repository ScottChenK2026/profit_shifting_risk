"""
build_report.py
---------------
Builds the CM3070 final project report: six chapters, strict word caps.

  Introduction   <=1000
  Literature     <=2500
  Design         <=2000
  Implementation <=2000
  Evaluation     <=2500
  Conclusion     <=1000
  total          <=9500   (the per-chapter caps sum to more, on purpose)

Layout: Cambria throughout, justified body text, narrow (0.5in) margins,
generous spacing around headings, figures and tables, and a hyperlinked
table of contents covering heading levels 1 and 2.

Code extracts are placed as images rather than text, so they read as figures.

Body word counts (excluding the title page, contents, headings, captions,
tables and references) are tracked per chapter and printed at the end.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
if (ROOT / "outputs" / "figures").is_dir():
    FIG = ROOT / "outputs" / "figures"
    OUT = ROOT.parent / "CM3070_Preliminary_Report_18082026.docx"
else:
    FIG = ROOT / "figures"
    OUT = ROOT / "CM3070_Preliminary_Report_18082026.docx"

# Cached heading -> page number map for the contents list. Regenerate with
#     python build_report.py --remap
# which builds the file, renders it, reads the real page numbers back, and
# rebuilds. The numbers are also PAGEREF fields, so Word corrects them itself
# when fields refresh on open.
PAGE_MAP_FILE = ROOT / "toc_pages.json"
PAGE_MAP: dict[str, int] = (
    json.loads(PAGE_MAP_FILE.read_text()) if PAGE_MAP_FILE.exists() else {})

FONT = "Cambria"
NAVY = RGBColor(0x1F, 0x38, 0x64)
BLUE = RGBColor(0x2E, 0x54, 0x96)
GREY = RGBColor(0x55, 0x55, 0x55)

doc = Document()

# --------------------------------------------------------------------------- #
# Page setup and styles
# --------------------------------------------------------------------------- #
sec = doc.sections[0]
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Inches(0.5))


def _set_font(style, name=FONT):
    """Set the font on a style for Latin, East-Asian and complex scripts, so
    Word does not silently fall back to Calibri for stray characters."""
    style.font.name = name
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), name)


st = doc.styles["Normal"]
_set_font(st)
st.font.size = Pt(11)
pf = st.paragraph_format
pf.space_after = Pt(8)
pf.line_spacing = 1.30
pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

for hstyle, size, colour, before, after in (
        ("Heading 1", 16, NAVY, 26, 12),
        ("Heading 2", 13, BLUE, 18, 8),
        ("Heading 3", 11.5, RGBColor(0x44, 0x54, 0x6A), 14, 6)):
    s = doc.styles[hstyle]
    _set_font(s)
    s.font.size = Pt(size)
    s.font.color.rgb = colour
    s.font.bold = True
    s.paragraph_format.space_before = Pt(before)
    s.paragraph_format.space_after = Pt(after)
    s.paragraph_format.keep_with_next = True
    s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

for extra in ("List Bullet", "List Paragraph"):
    try:
        _set_font(doc.styles[extra])
    except KeyError:
        pass

counts: dict[str, int] = {}
_cur = {"ch": None}
_fig_n = {"n": 0}
_tab_n = {"n": 0}
_lst_n = {"n": 0}
_bookmarks: list[tuple[int, str, str]] = []   # (level, text, bookmark id)


def set_chapter(name):
    _cur["ch"] = name
    counts.setdefault(name, 0)


def body(text):
    p = doc.add_paragraph()
    p.add_run(text)
    if _cur["ch"]:
        counts[_cur["ch"]] += len(text.split())
    return p


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(4)
    p.add_run(text)
    if _cur["ch"]:
        counts[_cur["ch"]] += len(text.split())
    return p


def heading(text, level):
    """Add a heading and bookmark it so the contents page can link to it."""
    p = doc.add_heading(text, level=level)
    bid = f"_Toc{len(_bookmarks) + 100}"
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), str(len(_bookmarks) + 100))
    start.set(qn("w:name"), bid)
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), str(len(_bookmarks) + 100))
    p._p.insert(0, start)
    p._p.append(end)
    _bookmarks.append((level, text, bid))
    return p


def caption(text, space_after=14):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = GREY
    return p


def _place(fname, width_in):
    doc.add_picture(str(FIG / fname), width=Inches(width_in))
    par = doc.paragraphs[-1]
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(12)
    par.paragraph_format.space_after = Pt(2)
    par.paragraph_format.keep_with_next = True
    return par


def figure(fname, width_in, cap_text, chapter):
    if not (FIG / fname).exists():
        print(f"  !! missing figure {fname}")
        return
    _fig_n["n"] += 1
    _place(fname, width_in)
    caption(f"Figure {chapter}.{_fig_n['n']}  {cap_text}")


def listing(fname, width_in, cap_text, chapter):
    """A code extract, placed as an image so it reads as a figure."""
    if not (FIG / fname).exists():
        print(f"  !! missing listing {fname}")
        return
    _lst_n["n"] += 1
    _place(fname, width_in)
    caption(f"Listing {chapter}.{_lst_n['n']}  {cap_text}")


def reset_counters():
    _fig_n["n"] = 0
    _lst_n["n"] = 0
    _tab_n["n"] = 0


def table(header, rows, cap_text, chapter, font_pt=9.0):
    _tab_n["n"] += 1
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(f"Table {chapter}.{_tab_n['n']}  {cap_text}")
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = GREY

    t = doc.add_table(rows=1, cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Light Grid Accent 1"
    for i, htext in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        par = c.paragraphs[0]
        par.paragraph_format.space_after = Pt(2)
        run = par.add_run(htext)
        run.bold = True
        run.font.size = Pt(font_pt)
        run.font.name = FONT
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            par = cells[i].paragraphs[0]
            par.paragraph_format.space_after = Pt(2)
            par.paragraph_format.line_spacing = 1.05
            run = par.add_run(str(v))
            run.font.size = Pt(font_pt)
            run.font.name = FONT
    for row in t.rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    if len(t.rows) <= 8:
        for row in list(t.rows)[:-1]:
            for cell in row.cells:
                for par in cell.paragraphs:
                    par.paragraph_format.keep_with_next = True
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(10)
    return t


def add_page_numbers():
    footer = doc.sections[0].footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.font.name = FONT
    run.font.size = Pt(9)
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve")
    it.text = "PAGE"
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
    run._r.append(f1); run._r.append(it); run._r.append(f2)


def build_toc(entries, page_map, anchor):
    """Write the contents list as real, clickable entries.

    Each line is an internal hyperlink to the heading's bookmark, followed by a
    dotted leader and a PAGEREF field. The field carries a cached page number
    so the list reads correctly the moment the file is opened, and refreshes to
    Word's own pagination when fields are updated.
    """
    created = []
    for level, text, bid in entries:
        p = doc.add_paragraph()
        created.append(p._p)
        pf_ = p.paragraph_format
        pf_.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf_.space_after = Pt(4 if level == 1 else 2)
        pf_.line_spacing = 1.0
        pf_.left_indent = Inches(0.0 if level == 1 else 0.28)
        # Right-aligned tab with a dot leader, just inside the text column.
        tabs = OxmlElement("w:tabs")
        tab = OxmlElement("w:tab")
        tab.set(qn("w:val"), "right")
        tab.set(qn("w:leader"), "dot")
        tab.set(qn("w:pos"), "10800")       # twentieths of a point
        tabs.append(tab)
        p._p.get_or_add_pPr().append(tabs)

        hyper = OxmlElement("w:hyperlink")
        hyper.set(qn("w:anchor"), bid)
        r = OxmlElement("w:r")
        rPr = OxmlElement("w:rPr")
        if level == 1:
            b = OxmlElement("w:b"); rPr.append(b)
        sz = OxmlElement("w:sz"); sz.set(qn("w:val"), "21" if level == 1 else "20")
        rPr.append(sz)
        col = OxmlElement("w:color"); col.set(qn("w:val"), "1F3864" if level == 1 else "333333")
        rPr.append(col)
        r.append(rPr)
        t = OxmlElement("w:t"); t.set(qn("xml:space"), "preserve"); t.text = text
        r.append(t)
        tab_r = OxmlElement("w:r")
        tab_el = OxmlElement("w:tab")
        tab_r.append(tab_el)
        hyper.append(r)
        hyper.append(tab_r)

        # PAGEREF field with a cached result.
        f1 = OxmlElement("w:r")
        fc1 = OxmlElement("w:fldChar"); fc1.set(qn("w:fldCharType"), "begin")
        f1.append(fc1)
        f2 = OxmlElement("w:r")
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve")
        it.text = f" PAGEREF {bid} \\h "
        f2.append(it)
        f3 = OxmlElement("w:r")
        fc3 = OxmlElement("w:fldChar"); fc3.set(qn("w:fldCharType"), "separate")
        f3.append(fc3)
        f4 = OxmlElement("w:r")
        t4 = OxmlElement("w:t"); t4.text = str(page_map.get(text, 1))
        f4.append(t4)
        f5 = OxmlElement("w:r")
        fc5 = OxmlElement("w:fldChar"); fc5.set(qn("w:fldCharType"), "end")
        f5.append(fc5)
        for el in (f1, f2, f3, f4, f5):
            hyper.append(el)
        p._p.append(hyper)

    # The entries are only known once every chapter has been written, so they
    # are created at the end of the document and then moved up to sit directly
    # under the Contents heading.
    prev = anchor._p
    for el in created:
        prev.addnext(el)
        prev = el


def request_field_update():
    """Ask Word to refresh fields (the TOC) when the document is opened."""
    settings = doc.settings.element
    upd = OxmlElement("w:updateFields")
    upd.set(qn("w:val"), "true")
    settings.append(upd)


# ============================== TITLE PAGE ============================== #
for _ in range(4):
    doc.add_paragraph()


def _centre(text, size, bold=False, colour=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(size)
    r.font.name = FONT
    if colour is not None:
        r.font.color.rgb = colour
    return p


_centre("Final Project Report", 15, bold=True, colour=NAVY)
_centre("CM3070 Computer Science Final Project", 12, colour=GREY)
doc.add_paragraph()
_centre("Detecting Profit Shifting Risk in Multinational Enterprises "
        "Using Neural Networks and OECD Country-by-Country Reporting Data",
        17, bold=True)
doc.add_paragraph()
doc.add_paragraph()
for txt, sz in (("BSc Computer Science, University of London", 12),
                ("Specialisation / Template: Machine Learning and Neural "
                 "Networks (CM3015)", 12),
                ("Scott Chen   ·   Student number: 210106514", 11),
                ("Professor: Matthew Yee-King", 11)):
    _centre(txt, sz, colour=GREY)
doc.add_page_break()

# ============================== CONTENTS ============================== #
_toc_heading = doc.add_heading("Contents", level=1)
_toc_anchor = doc.add_paragraph()
_toc_anchor.paragraph_format.space_after = Pt(2)
_toc_pagebreak = doc.add_paragraph()
_toc_pagebreak.add_run().add_break(WD_BREAK.PAGE)


# ============================== CHAPTER 1 ============================== #
set_chapter("1 Introduction")
reset_counters()
heading("Chapter 1: Introduction", 1)

body("A large multinational is really a group of many smaller companies, in "
     "different countries, all under the same owner. Because those companies "
     "trade with one another, the group has a lot of say over where its profit "
     "ends up on paper, and an obvious incentive to make profit appear where "
     "tax is low even when the real work happens somewhere else. That practice "
     "is called profit shifting, and it costs governments a great deal of tax "
     "revenue every year. This project is about spotting the warning signs of "
     "it automatically.")

body("To make profit shifting easier to see, large groups now have to file a "
     "yearly summary called a Country-by-Country Report, or CbCR. For each "
     "country a group operates in, it lists how much money the group takes in, "
     "how much profit it reports, how much tax it pays, how many people it "
     "employs and what it owns. The reporting standard came out of the "
     "OECD/G20 work on base erosion and profit shifting (OECD, 2015). The "
     "tell-tale sign of shifting is easy to state: a lot of profit reported in "
     "a place with barely any staff or assets. The catch is that the tools used "
     "to flag it are still mostly simple rules, such as “flag any subsidiary "
     "in a country whose headline tax rate is below some cut-off”. Rules like "
     "that miss the cases that matter, where no single number looks wrong but "
     "the combination does: high profit per employee, very little tax actually "
     "paid on that profit, most revenue coming from other companies in the same "
     "group, and a cluster of paper-only holding or finance companies with "
     "hardly any staff.")

body("My motivation is both professional and academic. I hold a Master's in "
     "Accounting and work as a transfer pricing analyst in tax technology, so "
     "cross-border compliance data is my daily material, and I have seen how "
     "much screening is still done by hand. The OECD now publishes the CbCR "
     "statistics openly, and they contain something most machine-learning work "
     "on tax has passed over: a count of how many establishments in each "
     "country do each kind of business, whether that is holding shares, lending "
     "within the group, managing patents and trademarks, or nothing at all. "
     "That mix of activities is, in effect, a fingerprint of how a group is "
     "structured in a place, and it is the signal this project sets out to use.")

body("The project asks one precise question: using only the open OECD "
     "statistics, can a neural network learn to recognise the countries long "
     "identified as tax havens from the reported economics and the activity mix "
     "alone, without ever being told which country it is looking at? It is not "
     "an attempt to prove wrongdoing by any company. The data is far too "
     "aggregated for that, and the output is a screening score meant to help a "
     "person decide where to look first.")

heading("1.1 What this project contributes", 2)

body("The main contribution is a set of inputs rather than a model. This "
     "project shows that the business-activity counts in the OECD's open data, "
     "how many establishments in a country do nothing but hold shares, manage "
     "intellectual property, lend within the group or sit dormant, carry much "
     "of the signal needed to identify tax-haven jurisdictions, and that this "
     "works without any paid, company-level data at all. As far as I can "
     "establish, that block of variables has not been fed to a machine-learning "
     "model for this problem before, and section 5.5 shows it ranks among the "
     "most useful inputs in both the tree model and the neural one.")

body("Two smaller contributions follow from that, and I do not want to "
     "overstate either. The first is a fair test. The research on deep learning "
     "for table-shaped data disagrees with itself about whether attention-based "
     "models beat gradient boosting, so I compare an FT-Transformer against a "
     "tuned XGBoost baseline on a split by year and check whether the "
     "difference is real rather than luck. The second is practical: the whole "
     "pipeline is open, can be rebuilt from a free download, and can be "
     "explained through SHAP, which matters in a field where a score nobody can "
     "explain to a taxpayer is a score nobody can use. What this project does "
     "not claim is record accuracy. The accuracy is good, but the interesting "
     "result is which inputs produce it.")

heading("1.2 Project template", 2)

body("This project follows the Machine Learning and Neural Networks template "
     "(CM3015). Rather than taking one of the two suggested briefs, neural "
     "style transfer or breast-cancer CNNs, it is a self-directed project "
     "within that template's deep-learning remit, applying neural networks to a "
     "real structured-data problem and following the workflow the module "
     "teaches: build a baseline, regularise, and evaluate carefully.")

heading("1.3 Aims and objectives", 2)

body("The aims are to test whether profit-shifting risk can be learned from "
     "open CbCR data; to find out whether the activity mix is the source of "
     "that signal; to compare a modern neural network fairly against the "
     "strongest simple baseline; and to keep the pipeline open, repeatable and "
     "explainable. The objectives are to turn the raw 925,000-row OECD export "
     "into a clean modelling table; to build features in three blocks "
     "(profitability and tax, economic substance, and the activity mix); to "
     "train an FT-Transformer, an MLP, an autoencoder and an XGBoost baseline; "
     "to test them on a later year using AUC, calibration, a significance test, "
     "an audit-budget view and SHAP; and to deliver tested code and a "
     "command-line scorer. Chapter 2 reviews and compares the related work, "
     "Chapter 3 sets out the design and plan, Chapter 4 describes the build, "
     "Chapter 5 evaluates it, and Chapter 6 concludes.")


# ============================== CHAPTER 2 ============================== #
set_chapter("2 Literature review")
reset_counters()
doc.add_page_break()
heading("Chapter 2: Literature Review", 1)

heading("2.1 How profit gets moved", 2)
body("The main lever for moving profit is the price one part of a group charges "
     "another, for a loan, a brand licence or a management fee. This is called "
     "transfer pricing. Because both sides share an owner, no real market sets "
     "the price, so the group has room to set it in a way that moves profit to "
     "a low-tax country. The rule meant to stop this, the arm's length "
     "principle (OECD, 2022), says these internal prices should match what two "
     "unrelated companies would have agreed. That sounds reasonable, but it is "
     "hard to enforce exactly where the largest sums are, on patents and "
     "internal lending, because there is no outside price to compare against. "
     "So a gap opens between where a group's real activity is and where its "
     "profit is reported, and CbCR was introduced (OECD, 2015) to make that gap "
     "visible.")

heading("2.2 Evidence on the scale of the problem", 2)
body("The scale is well established, though estimates vary with method. "
     "Tørsløv, Wier and Zucman (2023) estimate that 36% of multinational "
     "profits are shifted to havens each year. Clausing (2016) reaches broadly "
     "similar conclusions for the United States by a different route, and the "
     "review by Beer, de Mooij and Liu (2020) draws a large literature together "
     "into a repeatable finding: reported profit responds strongly to "
     "differences in tax rates. These studies disagree on the exact size "
     "precisely because their data and assumptions differ, which is a useful "
     "warning that any single number, including a model's risk score, depends "
     "on what went into it. Work using CbCR data directly (Garcia-Bernardo and "
     "Janský, 2024) documents large and persistent gaps between where profit "
     "is booked and where staff and assets sit, which is exactly the contrast "
     "my substance features try to measure. The recurring lesson is that the "
     "signal comes from combinations: a low tax rate on its own does not make a "
     "haven, it is low rates together with secrecy, related-party flows and "
     "shell structures. That is the main argument for a model that can combine "
     "features rather than test each one against a threshold.")

body("It is worth being sceptical about this evidence rather than just quoting "
     "the headline numbers. The estimates rest on assumptions about how to "
     "split profit between countries, and on data sources that are themselves "
     "patchy and unevenly reported, which is part of why credible studies "
     "differ by tens of percent. The move to public CbCR data, which this "
     "project uses, fixes some of that but brings its own problems: the figures "
     "are totals rather than company-level, some country cells are withheld or "
     "rounded, and dividends paid within a group can be counted twice and "
     "inflate apparent profitability. A model built on this data inherits those "
     "flaws, so the sensible goal is a useful screening signal rather than a "
     "precise measurement. That shapes both how I frame the work and which "
     "metrics I choose.")

heading("2.3 What offshore centres look like", 2)
body("The most directly relevant work is Garcia-Bernardo et al. (2017), whose "
     "analysis of global ownership networks separates “sink” centres, which "
     "accumulate holding and intellectual-property structures, from “conduit” "
     "countries such as the Netherlands, the United Kingdom, Switzerland, "
     "Singapore and Ireland, which route money through finance and holding "
     "entities. This matters here for two reasons. First, it means the kind of "
     "entity present in a country tells you about its role, not just its tax "
     "rate. Second, that is precisely what the OECD activity counts measure. As "
     "far as I can tell, no published model for detecting profit shifting has "
     "used those counts as inputs, and that is the gap this project aims at. "
     "One weakness I inherit from them is that the conduit and sink labels are "
     "themselves a judgement rather than ground truth, which I return to when "
     "discussing my own label.")

body("Their work also hints at a richer target than a yes/no haven flag: a "
     "range running from ordinary operating countries through conduits to pure "
     "sinks. The activity counts could in principle support that. I keep to a "
     "binary label here to keep the comparison with earlier work clean, but I "
     "note it as a natural extension.")

heading("2.4 Machine learning for tax risk", 2)
body("Several systems already apply machine learning to tax and customs risk. "
     "They are worth comparing individually, because the differences between "
     "them shaped my own design decisions, and Table 2.1 sets them side by "
     "side.")

body("de Roux et al. (2018) tackle under-reporting in Colombian tax returns. "
     "Because audited outcomes are scarce and expensive, they drop supervised "
     "learning altogether and group taxpayers by their declared figures, "
     "treating small, unusual clusters as suspicious. Flagged taxpayers were "
     "then audited and a high share turned out to be under-reporting. The "
     "strength is that the method needs no labels. The weakness, for my "
     "purposes, is that they never show what a supervised model would have "
     "achieved on the same data, so a reader cannot tell what the label-free "
     "approach costs. I use the idea, my autoencoder exists for the same "
     "reason, but I run it alongside supervised models rather than instead of "
     "them.")

body("Vanhoeyveld, Martens and Peeters (2020) work on Belgian VAT fraud and "
     "compare several anomaly-detection methods within industry sectors, "
     "reporting AUC against known fraud cases. Their contribution is careful "
     "benchmarking on a genuinely large administrative dataset, and judging a "
     "firm against its sector peers rather than in absolute terms is a good "
     "idea that I partly copy by expressing nearly every feature as a ratio. "
     "The limitation is access: the data is confidential, so nobody outside the "
     "tax authority can repeat or extend the result.")

body("Kim et al. (2020) present DATE, a deep model for choosing which customs "
     "declarations to inspect in Nigeria. Two things stand out. It is one of "
     "the few papers here to use a modern deep architecture rather than an "
     "off-the-shelf classifier, and it measures success by how many real cases "
     "appear within a fixed inspection budget rather than by overall accuracy, "
     "because that is the constraint the officers actually face. That second "
     "choice struck me as obviously right and I have taken it directly as my "
     "audit-budget analysis in section 3.6. The difference is that customs "
     "fraud is a transaction-level question with real decided outcomes, while "
     "profit shifting at country level has neither.")

body("Garcia-Bernardo and Janský (2024) is the closest work on my actual data, "
     "using CbCR statistics to measure the mismatch between profit and activity "
     "across countries. It measures and explains but builds no classifier, and "
     "it does not use the activity counts. It is, in effect, the descriptive "
     "half of what I am attempting.")

table(["Study", "Data", "Method", "Success measured by", "Gap this project fills"],
      [["de Roux et al. (2018)", "Colombian returns (closed)",
        "Unsupervised clustering", "Later audit hit-rate",
        "No supervised benchmark; data not open"],
       ["Vanhoeyveld et al. (2020)", "Belgian VAT (closed)",
        "Anomaly detection by sector", "AUC vs known fraud",
        "Closed data; no deep models tested"],
       ["Kim et al. (2020)", "Nigerian customs (closed)",
        "Tree embedding + attention", "Precision within inspection budget",
        "Different task; real labels, mine are a proxy"],
       ["Garcia-Bernardo and Janský (2024)", "OECD CbCR (open)",
        "Economic measurement", "Descriptive only",
        "No model; ignores the activity counts"],
       ["This project", "OECD CbCR (open)",
        "FT-Transformer, MLP, autoencoder, XGBoost",
        "AUC, PR-AUC, significance, calibration, precision@k, SHAP", "—"]],
      "Related tax-risk systems and the gap each leaves.", 2, font_pt=8.5)

body("Three limitations recur. Almost every system runs on confidential data, "
     "so nobody outside the issuing authority can reproduce or build on the "
     "results; this project uses only a free public download, trading detail "
     "for openness. Most use standard classifiers, with gradient boosting the "
     "repeated strong performer (Chen and Guestrin, 2016), and the one deep "
     "exception addresses a different task, so whether modern deep models help "
     "on profit-shifting data is simply untested. And none of them uses "
     "information about what the entities in a country actually do, even though "
     "the offshore-finance work in section 2.3 says that is where the signal "
     "lives.")

body("There is a further problem with how these systems are judged. Several "
     "report a single AUC or accuracy figure from a random split, with little "
     "attention to the rarity of the positive class, to whether the predicted "
     "probabilities mean anything, to the threshold an analyst would really "
     "use, or to whether a gap between two models is meaningful at all. For a "
     "screening tool that matters: a model can post a flattering AUC and still "
     "be useless at the only threshold anyone will apply, and two models can "
     "differ by a couple of points purely by chance. The evaluation plan in "
     "section 3.6 is built to close that gap.")

body("One tension is worth naming because it shaped my priorities. Tax "
     "authorities have preferred transparent, rule-based screening, because a "
     "rule can be explained to a taxpayer and defended on appeal while a learned "
     "score cannot. That is about whether a decision can be justified, not "
     "about technical taste, and it is why explainability is treated here as a "
     "goal in its own right rather than a closing paragraph.")

heading("2.5 Deep learning on table-shaped data", 2)
body("Whether deep learning helps on table-shaped data at all is genuinely "
     "disputed, and this project is built around that dispute rather than "
     "assuming an answer. Grinsztajn, Oyallon and Varoquaux (2022) show that "
     "tree ensembles still tend to beat neural networks on typical tabular "
     "problems, because trees cope better with uninformative features and with "
     "targets that change abruptly. Shwartz-Ziv and Armon (2022) reach the same "
     "conclusion from another angle, finding that deep models which beat "
     "boosting on their authors' chosen datasets often fail to do so on fresh "
     "ones. Against this, Gorishniy et al. (2021) introduce the FT-Transformer, "
     "which turns each feature into an embedding and applies self-attention, "
     "and report it competitive with gradient boosting. The disagreement is "
     "unresolved and depends on the dataset, which is exactly why I build a "
     "compact FT-Transformer and test it against XGBoost on real data with a "
     "significance test, treating “does the neural model win here?” as a "
     "question to answer rather than assume.")

body("Other designs occupy the same space. TabNet (Arik and Pfister, 2021) "
     "picks a small subset of features at each step, which buys some built-in "
     "explainability, and NODE (Popov, Morozov and Babenko, 2020) makes "
     "tree-like decisions trainable by gradient descent. The survey by Borisov "
     "et al. (2022) covers both and concludes that no deep design reliably "
     "beats a well-tuned gradient-boosting baseline, and that where an "
     "advantage appears it tends to be on very large datasets. With roughly "
     "fourteen thousand rows my problem sits firmly where the literature "
     "expects trees to be hard to beat. The realistic expectation, then, is "
     "that XGBoost will lead, and that the neural models earn their place "
     "elsewhere: in the autoencoder's ability to score without a label, and in "
     "what attention and SHAP together reveal about which features matter.")

heading("2.6 Anomaly detection and explainability", 2)
body("Because confirmed cases of profit shifting are not available as labels, "
     "the problem is partly one of anomaly detection: havens are an unusual "
     "minority. Pang et al. (2021) review the deep options, of which the "
     "reconstruction autoencoder is the most established. Trained only on "
     "ordinary countries, it gives a high error to those it cannot reproduce, "
     "producing a score that never touches the imperfect labels. Separately, "
     "any score used in a regulatory setting has to be explainable, because an "
     "analyst must be able to say why a case was flagged. SHAP (Lundberg and "
     "Lee, 2017) attributes a prediction to its inputs in a model-agnostic way, "
     "and I use it both to justify predictions and, more importantly, to check "
     "that the models rely on sensible signals rather than quirks of the data.")

body("SHAP should not be taken on trust either. Its attributions can be "
     "unstable when features are correlated, and I have several that are, since "
     "many ratios share a denominator. A feature ranking explains the model, "
     "not the world. So I use SHAP as a sanity check and a way of communicating "
     "results, read against the economic expectations from sections 2.2 and "
     "2.3, rather than as evidence about why any particular country attracts "
     "shifted profit.")

heading("2.7 Where this project sits", 2)
body("To summarise: the economic literature shows shifting is large and driven "
     "by combinations of structural factors; the machine-learning work on tax "
     "risk relies on closed data, rarely tests modern deep models against "
     "strong baselines, and often evaluates in ways that would not survive "
     "contact with a real audit budget; and the offshore-finance literature "
     "points to an activity-mix signal no classifier has used. This project "
     "sits at that intersection. Its claim is not that a transformer beats a "
     "tree, but that a freely available and previously unused part of the OECD "
     "data is where the predictive signal actually sits, shown with an "
     "evaluation designed to be hard to fool.")


# ============================== CHAPTER 3 ============================== #
set_chapter("3 Design")
reset_counters()
doc.add_page_break()
heading("Chapter 3: Design", 1)

heading("3.1 Domain, users and why the concept is justified", 2)
body("The project sits where international tax policy meets applied machine "
     "learning. Three users motivate it. A tax-authority analyst has to triage "
     "a large volume of filings under a fixed audit budget and needs a "
     "data-driven way to decide where to look first. A researcher studying "
     "profit shifting needs an open, repeatable scoring tool, which, as section "
     "2.4 showed, does not currently exist. A transfer-pricing practitioner, my "
     "own context, may want a risk indicator for internal review before a "
     "filing goes out. In every case the deliverable is a ranked warning "
     "signal, not a verdict, and that is what justifies a model producing "
     "meaningful probabilities that can be explained, rather than a hard "
     "yes/no classifier.")

heading("3.2 System architecture", 2)
body("The system is a four-layer pipeline, shown in Figure 3.1, and the layers "
     "are deliberately independent so each can be tested and replaced on its "
     "own. The data layer holds the adapter, the only component that ever "
     "touches the raw OECD download; it produces a tidy table, and nothing "
     "downstream needs to know about file formats or file sizes. The feature "
     "layer turns that table into the eighteen inputs, grouped into the blocks "
     "in section 3.4, and is where the anti-cheating rules are enforced. The "
     "model layer holds the four models behind a common interface, so adding a "
     "fifth would not mean touching the evaluation code. The last layer "
     "computes every metric and figure and exposes the trained model through a "
     "command-line tool.")

body("Two decisions in that structure are worth defending. First, the split "
     "into training, tuning and test years is defined once, at the top, and "
     "everything fitted downstream, the scaler, the fill-in values, the model "
     "parameters, comes from the training years alone. Putting that boundary in "
     "one place rather than in each model is what makes it checkable. Second, "
     "the pipeline saves each model's predictions to disk, so an interrupted "
     "run resumes instead of restarting. That sounds like a convenience, but on "
     "a laptop it is what made it practical to keep improving the evaluation "
     "without retraining everything each time.")

figure("architecture.png", 6.5,
       "System architecture. The split by year is enforced once, at the top.", 3)

heading("3.3 Data source and adapter", 2)
body("The source is the OECD's CbCR Table I statistics, downloaded as the full "
     "unfiltered export of about 925,000 rows. The adapter reads the roughly "
     "350 MB file in chunks, keeps the overall totals, drops regional groupings "
     "such as “Other Europe” and “Stateless”, and reshapes the financial "
     "figures, the activity counts and the profit and loss breakdowns into one "
     "row per reporting country, partner country and year. That gives 14,137 "
     "rows covering 56 reporting and 228 partner countries for 2016 to 2021.")

body("Three choices there are worth justifying, because each could have gone "
     "the other way. Reading in chunks is partly about memory, but mostly about "
     "not tying the project to the machine it runs on; the same code works "
     "whether the OECD publishes 925,000 rows or three million. Keeping only "
     "the overall totals avoids counting the same money twice, since the "
     "profit-making and loss-making breakdowns add up to the total. And the "
     "unit of analysis is the reporting-partner-year combination rather than "
     "the partner country alone, which keeps far more rows and lets the same "
     "country look different depending on who is reporting it, at the cost that "
     "rows are not fully independent, a point I return to in section 5.7.")

heading("3.4 Features and the label", 2)
body("Eighteen features are built in four blocks, summarised in Table 3.1. Each "
     "block comes from a specific finding in Chapter 2 rather than from "
     "convenience. The profitability and tax ratios capture the headline "
     "symptom, profit detached from activity and lightly taxed. The substance "
     "ratios measure the profit-versus-people-and-assets gap that "
     "Garcia-Bernardo and Janský (2024) document. The activity shares capture "
     "the conduit and sink structure of Garcia-Bernardo et al. (2017), and the "
     "loss ratio reflects the deliberate parking of losses.")

table(["Block", "Features", "What it captures", "From"],
      [["Profitability and tax",
        "profit margin, related-party revenue share, tax rate paid and accrued",
        "Profit detached from activity, lightly taxed", "§2.1, §2.2"],
       ["Economic substance",
        "profit, revenue, capital and assets per employee; employees per entity; entities per group",
        "Whether booked profit matches real people and assets", "§2.2"],
       ["Business-activity mix",
        "holding, IP, internal-finance and dormant shares; combined shifting share; real-activity share",
        "The structural fingerprint of a conduit or sink", "§2.3"],
       ["Loss shifting", "loss-side profit as a share of total absolute profit",
        "Concentration of loss-making sub-groups", "§2.2"]],
      "The four feature blocks and the work that motivates each.", 3,
      font_pt=8.5)

body("The label is whether the partner country appears on a combined tax-haven "
     "list, drawn from the EU's non-cooperative lists, the Tax Justice Network "
     "rankings and the conduit and sink centres of Garcia-Bernardo et al. "
     "(2017). It is decided entirely separately from every feature. The "
     "country's identity is deliberately kept out of the inputs: if the model "
     "could see that a partner was “Cayman”, it would simply memorise the "
     "answer. I also dropped headline tax-rate features, so the question stays "
     "clean: can the reported economics and the activity mix alone reveal the "
     "havens? A unit test enforces this by checking that no single feature is "
     "almost perfectly correlated with the target.")

body("Missing values are handled deliberately rather than dropped. The tax rate "
     "is undefined where profit is zero or negative, about a quarter of rows, "
     "and is filled with the training-set median, while a missing activity "
     "count means there are no establishments of that type and becomes zero. "
     "Every fill-in value and the scaler are computed on the training years "
     "alone and then applied to later years, so nothing from the test period "
     "can leak backwards.")

heading("3.5 The four models and why these four", 2)
body("The four models are not all expected to win. They are chosen to cover the "
     "range of plausible approaches so that the comparison actually settles "
     "something. Table 3.2 sets out each choice against the alternative I "
     "rejected.")

table(["Model", "Setup", "Why this one", "Alternative rejected"],
      [["FT-Transformer",
        "18 tokens of size 32, 4 heads, 3 blocks, learnable [CLS], AdamW, early stopping",
        "§2.2 says the signal comes from combinations; attention models feature interactions directly",
        "TabNet and NODE target the same regime, but Borisov et al. (2022) find neither reliably better"],
       ["MLP", "128–64–32, ReLU, dropout 0.3, class-weighted loss",
        "Shows how much of any neural gain comes from attention rather than flexibility alone",
        "Leaving it out, which would make the transformer result impossible to interpret"],
       ["Autoencoder",
        "32–16–8 encoder, mirrored decoder, trained on non-havens only",
        "The label is a proxy; a model that never sees it checks the supervised results",
        "One-class SVM or isolation forest, which give no per-feature account of why a row is odd"],
       ["XGBoost", "Grid over depth, learning rate and trees; 3-fold stratified CV",
        "The consistent strong performer on this kind of data (§2.5), so it is the bar to clear",
        "Logistic regression, too weak to make a neural win meaningful"]],
      "The four models, the reasoning behind each, and the alternative rejected.",
      3, font_pt=8.0)

body("Class imbalance is handled the same way across all four, by weighting the "
     "rare class in the loss rather than by duplicating rows, so the test set "
     "keeps a realistic mix. The FT-Transformer is kept small so it trains on a "
     "CPU in minutes, which matters because anyone should be able to re-run "
     "the whole pipeline on a laptop.")

heading("3.6 Evaluation plan", 2)
body("The evaluation follows machine-learning norms rather than, say, the user "
     "testing suited to a web project. The principle is that each test should "
     "answer a question I could otherwise get wrong, and Table 3.3 states those "
     "questions alongside the metric and what I would treat as a good result.")

body("The split is the foundation. Models are trained on 2016 to 2019, tuned on "
     "2020 and tested on 2021. Splitting by year rather than at random matters "
     "because the same country pair recurs every year with almost identical "
     "figures, so a random split would put near-duplicates on both sides and "
     "flatter the score. It also matches how the tool would really be used, "
     "predicting a later year from earlier ones.")

table(["Test", "Question", "Metric", "Good result"],
      [["Test on a later year", "Does it generalise to a year never seen?",
        "AUC-ROC on 2021", "Clearly above 0.80"],
       ["Ranking under imbalance", "Is performance real with only ~19% positives?",
        "PR-AUC vs the 0.19 base rate", "Several times the base rate"],
       ["Model comparison", "Is the gap between models more than noise?",
        "DeLong test", "A stated z and p, either way"],
       ["Calibration", "When it says 0.7, is it right about 70% of the time?",
        "Brier score, reliability curves", "Points near the diagonal"],
       ["Audit budget", "Under real review capacity, how clean is the top slice?",
        "Precision and recall at top 5/10/20%", "Top-decile precision well above base rate"],
       ["Error analysis", "What kind of mistakes does each model make?",
        "Confusion matrices", "Errors that make sense"],
       ["Feature attribution", "Is it using sensible signals?", "SHAP importance",
        "Substance and activity features prominent"],
       ["Recognisability", "Are the top-ranked countries the expected ones?",
        "Known-haven rate in the top 50", "Overwhelmingly known havens"],
       ["Correctness", "Does the code do what it claims?",
        "Unit tests, including a leakage guard", "All tests pass"]],
      "Each test, the question it answers, the metric, and what would satisfy me.",
      3, font_pt=8.2)

body("Two of these need a note. The audit-budget analysis is taken from Kim et "
     "al. (2020) and is the number I would put in front of a practitioner, "
     "because a fixed inspection capacity, not headline AUC, is what a tax "
     "authority actually experiences. The DeLong test is there because of the "
     "criticism I made in section 2.4, and I committed to reporting its result "
     "whichever way it fell, including the case where my neural model loses.")

body("It is worth saying what is deliberately not tested. There is no user "
     "study. A tool like this would eventually need one, since whether an "
     "analyst trusts and acts on a score is a question about people rather than "
     "models, but I have no access to tax-authority analysts and a study with "
     "fellow students would tell me nothing useful. Nor is there a comparison "
     "against a live rule-based system, because the thresholds real "
     "administrations use are not published. The audit-budget view stands in "
     "for both: it at least expresses performance in the units the intended "
     "user works in.")

heading("3.7 Technology", 2)
body("Everything is Python and open source, chosen so the pipeline can be "
     "rebuilt from a free download with no licences and no GPU. Table 3.4 "
     "lists each component and why it was chosen.")

table(["Component", "Library", "Version", "Why"],
      [["Data handling", "pandas, numpy", "≥1.5, ≥1.24",
        "Chunked reading makes the adapter possible"],
       ["Metrics and classical ML", "scikit-learn, scipy", "≥1.3, ≥1.10",
        "Metrics, calibration curves, stratified splits"],
       ["Gradient boosting", "xgboost", "≥2.0", "The baseline; fast on CPU"],
       ["Neural networks", "torch", "≥2.2",
        "The FT-Transformer needs a custom tokeniser"],
       ["Explainability", "shap", "≥0.44",
        "Explains the tree and the networks on the same basis"],
       ["Figures", "matplotlib, seaborn", "≥3.7, ≥0.12",
        "Figures regenerate with the pipeline"],
       ["Testing", "pytest", "≥7.4", "Hosts the leakage guard (§4.5)"]],
      "Technology used, and the reason for each choice.", 3, font_pt=8.5)

heading("3.8 Work plan", 2)
body("Figure 3.2 shows the schedule and Table 3.5 breaks the remaining work "
     "into tasks with deliverables and milestones. The project runs from early "
     "May to the end of September 2026. Everything up to and including the "
     "four-model build and this report is done; what remains is the "
     "label-sensitivity study, the combined score, the threshold work, and the "
     "documentation and write-up, alongside revision for the written exam in "
     "mid-September.")

figure("gantt.png", 6.3,
       "Project timeline with task breakdown and milestones.", 3)

table(["ID", "Task", "Deliverable", "Window", "Milestone"],
      [["T1", "Consolidate results into the full six-chapter report",
        "This report", "Early–mid Aug", "M2 Full report"],
       ["T2", "Label-sensitivity study: drop contested conduits, try a positive-unlabelled setup",
        "Sensitivity table", "Mid–late Aug", "—"],
       ["T3", "Combine the autoencoder score with the supervised score",
        "Combined-score results, or a negative finding", "Late Aug–early Sep", "—"],
       ["T4", "Threshold set from a stated audit capacity", "Threshold analysis",
        "Early Sep", "—"],
       ["T5", "Exam revision", "—", "Late Aug–mid Sep", "M3 Written exam"],
       ["T6", "Documentation and test coverage for third-party reproduction",
        "Reproducible repository", "Early–mid Sep", "—"],
       ["T7", "Final write-up", "Final report", "Sep", "M4 Final report due"]],
      "Task breakdown, deliverables and milestones.", 3, font_pt=8.2)

body("The plan is realistic mainly because the riskiest parts are behind me: "
     "getting the data into shape and proving the models train and generalise. "
     "What is left is analysis and writing. The remaining risks are about the "
     "analysis rather than the engineering, and each has a fallback. If the "
     "model turns out to lean too heavily on my imperfect haven list, T2 and "
     "the label-free autoencoder will expose it. If the neural networks still "
     "do not beat XGBoost, that is a reportable result rather than a failure, "
     "so nothing depends on getting a particular answer. And because the whole "
     "pipeline reruns on a laptop in minutes, each extra experiment is cheap. "
     "Table 3.6 records the main risks and how each is handled.")

table(["Risk", "Likelihood", "Impact", "Mitigation"],
      [["The proxy label biases the model, so it learns my list rather than the pattern",
        "Medium", "High", "Sensitivity study (T2); label-free autoencoder; discussed in §5.7"],
       ["Neural models under-perform the baseline", "High", "Low",
        "Already the case, and reported as a fair test rather than a failure (§5.2)"],
       ["Activity counts missing for some countries", "Medium", "Medium",
        "A missing count means no establishments, so zero is the truthful reading"],
       ["Remaining analysis collides with exam revision", "Medium", "Medium",
        "Plan is light late Aug to mid Sep; T2–T4 are independent so any one can slip"],
       ["Raw file too large for the machine", "Low", "Medium",
        "Chunked adapter; synthetic data generator as a fallback for tests"]],
      "Risk register for the remainder of the project.", 3, font_pt=8.5)

heading("3.9 Ethical considerations", 2)
body("The data is aggregated and anonymised, so no company is identifiable and "
     "no personal data is processed. The output is a screening score, not a "
     "finding of avoidance. The clearest misuse would be to treat a high score "
     "as an accusation against a legitimate structure, so any real deployment "
     "would need a person in the loop. The proxy nature of the label is an "
     "ethical as much as a technical limitation, because a model trained on a "
     "proxy inherits the assumptions of whoever drew up the list, and those "
     "lists are politically contested.")


# ============================== CHAPTER 4 ============================== #
set_chapter("4 Implementation")
reset_counters()
doc.add_page_break()
heading("Chapter 4: Implementation", 1)

body("The pipeline runs end to end on the real OECD data. Every number and "
     "figure in this chapter and the next comes from one command, "
     "python run_pipeline.py, with a fixed random seed, on CPU, in a few "
     "minutes. This chapter describes how it is built; Chapter 5 evaluates what "
     "it produces.")

heading("4.1 From 925,000 rows to a model table", 2)
body("The raw export holds 924,811 rows, one per reporting country, partner "
     "country, measure, breakdown and year. It is both too large to load "
     "comfortably and the wrong shape for modelling. The adapter reads it in "
     "200,000-row chunks and throws away what it does not need as it goes, so "
     "memory use stays modest however large the download becomes. The filter in "
     "Listing 4.1 is the heart of it, and each of its four conditions removes a "
     "specific way the data could mislead the model.")

listing("listing_adapter.png", 5.9,
        "The streaming filter in src/oecd_adapter.py.", 4)

body("The check for a genuine country deserves a mention, because it caused the "
     "one bug that genuinely surprised me. The export mixes real country codes "
     "with groupings such as “Stateless”, “Rest of world” and regional "
     "totals. Left in, these produce rows whose economics look extraordinary "
     "because they are sums over dozens of countries, and the model happily "
     "learns them. Excluding anything that is not a three-letter country code, "
     "and is not on an explicit exclusion list, removes them cleanly.")

body("After filtering, the adapter reshapes the measures into columns, attaches "
     "the profit-making and loss-making breakdowns as separate fields, and "
     "drops rows with neither revenue nor profit. The result is 14,137 rows "
     "across 56 reporting and 228 partner countries for 2016 to 2021. The "
     "label, partner country is a known haven, is true for 2,665 rows, or "
     "18.9%, which is a realistic minority rather than an extreme one.")

heading("4.2 Building the features", 2)
body("Raw revenue or profit means little on its own; the ratios carry the "
     "signal. Two details matter more than they look. The first is that "
     "division goes through a helper that turns division by zero into a missing "
     "value rather than infinity, because a single infinite value will quietly "
     "wreck a scaler. The second is a log that keeps the sign, used for "
     "profit-based ratios: profit can be negative, an ordinary log would fail, "
     "but whether a figure was a profit or a loss carries real meaning, so I "
     "take the log of the size and put the sign back. Listing 4.2 shows both.")

listing("listing_features.png", 5.9,
        "Numerical guards in src/features.py.", 4)

body("The activity block turns raw establishment counts into shares of all "
     "establishments in that country, so a large economy and a small one are "
     "compared on the same footing. Missing values are handled by meaning "
     "rather than by rule: a missing activity count means there are none of "
     "that type, so it becomes zero, whereas a missing tax rate is genuinely "
     "unknown and is filled with the training-year median. The tax rate is "
     "missing for 23.6% of rows and the loss ratio for 41.7%, both for "
     "structural reasons rather than poor data quality. Figure 4.2 confirms the "
     "features are not merely restating one another.")

body("The features separate the two classes in the way Chapter 2 predicts, and "
     "strikingly so for the activity block. Table 4.1 contrasts the class means "
     "on the real data.")

table(["Feature", "Non-haven mean", "Haven mean", "Ratio"],
      [["Effective tax rate", "0.282", "0.109", "0.39×"],
       ["Related-party revenue share", "0.220", "0.382", "1.74×"],
       ["Holding-company share", "0.043", "0.209", "4.86×"],
       ["Internal-group-finance share", "0.010", "0.033", "3.30×"],
       ["Shifting-activity share", "0.149", "0.363", "2.44×"],
       ["Real-activity share", "0.633", "0.336", "0.53×"],
       ["Employees per entity", "152.0", "38.4", "0.25×"]],
      "Class means for the most telling features.", 4, font_pt=9.0)

body("None of these is decisive on its own, and Figure 4.1 shows real overlap in "
     "every one of them. That overlap is why a threshold rule fails and a model "
     "that can combine them is worth building.")

figure("feature_distributions.png", 6.3,
       "Feature distributions by class (havens orange, non-havens blue). The "
       "activity and tax-rate features separate best, but all overlap.", 4)

figure("correlation_heatmap.png", 6.2,
       "Correlation between features, lower triangle only. The strongest pair "
       "is about −0.89. No feature was dropped, for the reason given in the "
       "text.", 4)

heading("4.3 The four models", 2)
body("All four were trained on 2016 to 2019 and tuned on 2020. The "
     "FT-Transformer is the technical centrepiece and took most of the "
     "implementation effort. A transformer normally reads a sequence of word "
     "embeddings; here there are no words, so each of the eighteen numbers must "
     "first become its own vector. Listing 4.3 is the part that does it: every "
     "feature gets its own learned weight and bias, so the value 1.2 in the "
     "holding-share column produces a different vector from 1.2 in the tax-rate "
     "column.")

listing("listing_tokenizer.png", 6.1,
        "The feature tokeniser and forward pass of the FT-Transformer "
        "(src/models/ft_transformer.py), following Gorishniy et al. (2021).", 4)

body("A learnable summary token is added in front of the feature tokens and the "
     "whole sequence passes through three transformer blocks with four "
     "attention heads. Attention lets the model relate any feature to any "
     "other directly, which is what section 2.2 argued the problem needs. The "
     "final state of the summary token feeds a small output layer. I chose the "
     "pre-normalisation arrangement after an initial version trained unstably "
     "at this small size, and kept the token size at 32 with three layers so "
     "the model stays trainable on a CPU.")

body("All three neural models share a training recipe: a loss that weights the "
     "rare class by the ratio of the two classes, roughly 4.3 to 1, AdamW with "
     "weight decay, and early stopping that restores the best snapshot so an "
     "over-trained model is never returned. Weighting the loss rather than "
     "duplicating rows was deliberate, because oversampling would change the "
     "mix the model sees and the audit-budget analysis in section 5.3 depends "
     "on that mix staying realistic.")

body("The MLP is three layers of 128, 64 and 32 units with dropout at 0.3, "
     "trained in batches of 256 for up to 200 epochs. It exists as a control: "
     "if the FT-Transformer beat XGBoost but the MLP did not, the gain would be "
     "down to attention specifically, whereas if both behaved alike the "
     "attention machinery is adding nothing. As section 5.2 reports, they "
     "behave almost identically.")

body("The autoencoder needs different treatment throughout. It compresses "
     "eighteen features through 32, 16 and 8 units and expands them back, and "
     "it is trained only on non-haven rows, so it never sees a positive "
     "example. How badly it rebuilds a row becomes the anomaly score. The final "
     "activation is dropped from both halves so the compressed code and the "
     "rebuilt row can take negative values, which matters because several "
     "features are signed logs. Its cut-off is set at the 95th percentile of "
     "training error rather than learned, and section 5.7 explains why that "
     "choice mattered more than I expected.")

body("XGBoost is grid-searched over depth (4, 6, 8), learning rate (0.03, 0.1) "
     "and number of trees (300, 600) using three-fold cross-validation on the "
     "training years, scored on AUC, with the same class weighting. Keeping the "
     "class mix constant across folds matters: with 19% positives, an uneven "
     "fold could contain very few havens and give a misleading score. Figures "
     "4.3 and 4.4 show both networks trained smoothly, with validation loss "
     "tracking training loss and no runaway over-fitting.")

figure("ft-transformer_training_curve.png", 4.2,
       "FT-Transformer training and validation loss. The curves stay close.", 4)

figure("mlp_training_curve.png", 4.2,
       "MLP training and validation loss; the same pattern.", 4)

heading("4.4 Running it repeatably", 2)
body("A single script wires the stages together, and two properties of it "
     "shaped how the project could be worked on. The random seed is fixed and "
     "set before any model is built, so a rerun reproduces the reported numbers "
     "exactly rather than approximately. And each model's predictions are "
     "cached to disk, with the script loading the cache instead of retraining "
     "if it finds one. That began as a convenience after an interrupted run, "
     "but it changed how I worked: because re-running the evaluation no longer "
     "meant retraining four models, I revised the metrics and figures far more "
     "often, and several observations in Chapter 5 came out of iterations I "
     "would otherwise not have bothered with.")

heading("4.5 Command-line tool and testing", 2)
body("The trained model, the scaler and the fill-in values are saved together, "
     "so new records get exactly the treatment training data did. A "
     "command-line tool loads them and scores either a single record typed as "
     "arguments or a whole file, returning a probability and a HIGH, MEDIUM or "
     "LOW band. As a sanity check, a made-up haven-like record (many holding "
     "and internal-finance establishments, very low tax, almost no staff) "
     "scores 0.95; a substance-heavy record scores 0.34; and scoring the full "
     "dataset puts Cayman, Luxembourg, Bermuda and the British Virgin Islands "
     "at the top.")

body("Seventeen unit tests cover the data, features, models and evaluation, and "
     "all pass. Most are ordinary checks: activity shares lie between zero and "
     "one, the label is binary, filling in leaves no gaps, the models learn a "
     "planted signal. The one I care most about is the guard in Listing 4.4, "
     "which fails the build if any single feature becomes almost perfectly "
     "correlated with the label. That is the circular-reasoning failure this "
     "project is most exposed to, and I wanted it caught automatically rather "
     "than by my own vigilance.")

listing("listing_leakage.png", 6.0,
        "The leakage guard in tests/test_features.py.", 4)


# ============================== CHAPTER 5 ============================== #
set_chapter("5 Evaluation")
reset_counters()
doc.add_page_break()
heading("Chapter 5: Evaluation", 1)

body("This chapter works through the plan in Table 3.3. All results are on the "
     "held-out 2021 data, 2,932 observations the models never saw during "
     "training or tuning.")

heading("5.1 Headline results", 2)
body("Table 5.1 gives the full set of metrics. The three supervised models are "
     "directly comparable; the autoencoder is unsupervised and is reported "
     "mainly as a ranker, for reasons section 5.7 explains.")

table(["Model", "AUC-ROC", "PR-AUC", "F1", "Precision", "Recall", "Brier"],
      [["XGBoost (baseline)", "0.941", "0.842", "0.765", "0.776", "0.755", "0.071"],
       ["FT-Transformer", "0.922", "0.780", "0.687", "0.604", "0.795", "0.102"],
       ["MLP", "0.920", "0.782", "0.675", "0.573", "0.822", "0.108"],
       ["Autoencoder (unsupervised)", "0.767", "0.432", "0.011", "1.000", "0.005", "0.174"]],
      "Performance on the held-out 2021 data. Threshold-based figures use a "
      "0.5 cut-off; the base rate is 18.8%.", 5, font_pt=9.0)

body("All three supervised models clear the bar set in Table 3.3 comfortably. "
     "The PR-AUC figures matter more given the imbalance, and at roughly four "
     "times the base rate they show the ranking is genuinely informative rather "
     "than an artefact of getting the majority class right. Figures 5.1 and 5.2 "
     "show the corresponding curves.")

figure("roc_curves.png", 4.3,
       "ROC curves on the held-out test year. XGBoost leads throughout.", 5)

figure("pr_curves.png", 4.3,
       "Precision-recall curves, which matter more than ROC at this base rate.", 5)

heading("5.2 Is the difference real, and do the probabilities mean anything?", 2)
body("The most important finding is that XGBoost is the best model and the gap "
     "is real rather than noise. DeLong's test rejects equality decisively: "
     "XGBoost against the MLP gives z = −4.81, p ≈ 1.5×10⁻⁶, and against the "
     "FT-Transformer z = −4.30, p ≈ 1.7×10⁻⁵. On this problem the "
     "gradient-boosted baseline beats both neural networks with statistical "
     "significance, which is what Grinsztajn et al. (2022) and Borisov et al. "
     "(2022) would predict for a dataset this size.")

body("I would rather report that plainly than dress it up. The attention model "
     "is competitive and was interesting to build, but it does not win here, "
     "and having committed in section 3.6 to reporting the test whichever way "
     "it fell, the fair comparison is itself one of the results. It also tells "
     "me where not to spend the remaining time.")

body("XGBoost is also much better calibrated, with a Brier score of 0.071 "
     "against roughly 0.10 for both neural models. Figure 5.3 shows why that "
     "matters in practice: the neural models are consistently over-confident in "
     "the middle of the range, so a score of 0.6 from the MLP does not mean "
     "what a user would reasonably assume. For a tool whose output is meant to "
     "be read as a probability, that is a real defect rather than a cosmetic "
     "one.")

figure("calibration.png", 4.3,
       "Calibration curves. Both neural models sit above the diagonal in the "
       "mid-range, so they overstate risk there.", 5)

heading("5.3 The audit-budget view", 2)
body("Headline AUC is not what a tax authority experiences. With a limited "
     "budget it can review only the highest-scoring cases, so how clean the top "
     "slice is matters far more than the area under any curve. Table 5.2 gives "
     "that view for the three supervised models.")

table(["Review budget", "XGBoost precision", "XGBoost recall",
       "FT-Transformer precision", "MLP precision"],
      [["Top 5% (147 cases)", "98.0%", "26.1%", "93.2%", "93.2%"],
       ["Top 10% (293 cases)", "92.5%", "49.2%", "89.8%", "88.4%"],
       ["Top 20% (586 cases)", "72.9%", "77.5%", "68.9%", "68.9%"]],
      "Precision and recall under realistic review capacities.", 5, font_pt=9.0)

body("An analyst reviewing only the top tenth of flagged country pairs would "
     "find that more than nine in ten are known havens, while still catching "
     "about half of all of them. That is a concrete, operationally meaningful "
     "result, and the one I would lead with in front of a practitioner. It is "
     "also where the models differ least: at the top 5% the three are within "
     "five points of each other, which suggests that for triage the choice "
     "between them matters less than the features they share.")

heading("5.4 What kind of mistakes each model makes", 2)
body("The confusion matrices in Figure 5.4 show the models fail differently at "
     "the default threshold. XGBoost is balanced, with 120 false alarms against "
     "135 missed havens. Both neural models trade precision for recall heavily: "
     "the MLP catches 453 of 551 havens, more than XGBoost's 416, but at the "
     "cost of 338 false alarms, nearly three times as many, as Figure 5.5 "
     "shows. The FT-Transformer sits between them.")

body("This follows directly from the class weighting meeting a fixed 0.5 "
     "threshold, and it is partly an artefact of using one cut-off across "
     "models whose calibration differs. It matters practically, because a tool "
     "that nearly triples the false-alarm load to gain about seven percentage "
     "points of recall would be rejected by whoever has to work the queue. It "
     "also points at a fix I have not yet made: the threshold should come from "
     "a stated audit capacity rather than sit at 0.5, which is task T4.")

figure("confusion_xgboost.png", 2.7,
       "Confusion matrix, XGBoost. Errors are roughly balanced.", 5)

figure("confusion_mlp.png", 2.7,
       "Confusion matrix, MLP. Higher recall, far more false alarms.", 5)

body("Looking at which cases are missed is more instructive than counting them. "
     "The misses fall into two groups. The first is small conduit countries in "
     "years where the reporting country files little activity detail, so the "
     "activity features are mostly filled in and the model is working from the "
     "tax and profitability block alone. The second is countries such as "
     "Ireland and the Netherlands that combine real operations with conduit "
     "structures; they have genuine employees and assets, so the substance "
     "ratios look ordinary even though the label says haven. That second group "
     "is not really a model failure at all. It is the label being blunt about a "
     "genuinely mixed case, and it is the strongest practical argument for the "
     "sensitivity study in T2.")

heading("5.5 What the models learned", 2)
body("This section carries the project's central claim, so it is worth being "
     "clear about what SHAP does and does not show. It attributes the model's "
     "output to its inputs. It does not establish that those inputs cause "
     "profit shifting in the world. What it can tell me is whether the model is "
     "leaning on the block of features Chapter 2 argued would matter.")

body("It is. For XGBoost the leading features are employees per entity (mean "
     "|SHAP| 1.50), the holding-company share (0.89) and the real-activity "
     "share (0.85), with capital and revenue per employee following. For the "
     "MLP the top three are the real-activity share, the dormant share and the "
     "combined shifting share, so the activity block takes the whole podium. "
     "Across both models, the part of the OECD data earlier work has ignored is "
     "among the most useful available, which is the empirical core of the claim "
     "in section 1.1. Figures 5.6 and 5.7 give the full rankings.")

body("The two models agree on substance but weight it differently, and that "
     "difference is informative rather than noise. The tree leans on employees "
     "per entity, a single sharp variable of the kind trees exploit well, while "
     "the neural models spread weight across the activity shares, closer to the "
     "combination story that motivated using attention. That the transformer "
     "still loses on accuracy suggests the combinations it can represent are "
     "not worth much beyond what the tree already captures with splits, at "
     "least at this sample size.")

figure("shap_xgboost.png", 4.4,
       "SHAP feature importance, XGBoost.", 5)

figure("shap_mlp.png", 4.4,
       "SHAP feature importance, MLP.", 5)

heading("5.6 Are the top-ranked countries recognisable?", 2)
body("Of the 50 highest-risk observations in 2021, every one is a known haven. "
     "The most frequently flagged partners are the Cayman Islands, Luxembourg, "
     "Bermuda, the British Virgin Islands, Hong Kong, Mauritius, the "
     "Netherlands and Switzerland, which are exactly the sinks and conduits the "
     "offshore-centre literature identifies. The model has learned something "
     "recognisable rather than an artefact. The obvious caveat is that since "
     "those countries define my label, this check confirms internal "
     "consistency, not external truth.")

heading("5.7 Critical assessment and limitations", 2)
body("The autoencoder result needs qualifying rather than reporting at face "
     "value. Its precision of 1.000 in Table 5.1 looks impressive, but it comes "
     "from making just three positive predictions out of 551 actual havens, a "
     "recall of 0.5%. A precision computed on three cases is not evidence of "
     "anything, and it is a direct consequence of fixing the cut-off at the "
     "95th percentile of training error rather than choosing it against a "
     "target. The fair view is the ranking one, where its AUC of 0.767 and "
     "top-5% precision of 54.4% sit far below every supervised model. The "
     "honest conclusion is that the label-free approach is much weaker here. "
     "That is worth knowing, and it means the case for combining it with the "
     "supervised score has to be tested rather than assumed, which is how T3 is "
     "now framed.")

bullet("Proxy label. The target is membership of a haven list, not confirmed "
       "shifting, so the model predicts the proxy and inherits its assumptions. "
       "This is the deepest limitation and T2 is the main response to it.")
bullet("Aggregated data. The statistics are totals at country-pair level, so "
       "the model speaks to country patterns and can say nothing about an "
       "individual firm.")
bullet("The neural network does not win. On this problem XGBoost is "
       "significantly better and better calibrated; the FT-Transformer's "
       "contribution is to establish that carefully, not to top the table.")
bullet("Contested conduits. Treating the Netherlands or Switzerland as havens "
       "is defensible but arguable, and how much the results depend on that is "
       "untested until T2 is done.")
bullet("Fixed threshold. Section 5.4 shows the 0.5 cut-off flatters some models "
       "and penalises others; a threshold derived from capacity would make the "
       "comparison fairer.")
bullet("Rows are not independent. Because the unit is the "
       "reporting-partner-year combination, the same partner country appears "
       "many times across training and test years. Splitting by year stops a "
       "row leaking into its own future, but it does not make rows independent, "
       "so the DeLong p-values should be read as strong rather than exact.")

body("Set against those caveats, the work does what this stage requires. There "
     "is a complete, tested, repeatable pipeline on real data; an evaluation "
     "that answers each question it set out to answer, including one where the "
     "answer was not the one I wanted; and a clear finding about which signals "
     "carry the information.")


# ============================== CHAPTER 6 ============================== #
set_chapter("6 Conclusion")
reset_counters()
doc.add_page_break()
heading("Chapter 6: Conclusion and Further Work", 1)

body("This project set out to test whether profit-shifting risk can be learned "
     "from open Country-by-Country Reporting data alone. On the evidence so far "
     "it can. A pipeline that turns a 925,000-row public download into 14,137 "
     "modelling rows, builds eighteen economically grounded features and trains "
     "four models reaches an AUC of 0.94 on a year it never saw, and would give "
     "an analyst reviewing the top tenth of cases a better than nine-in-ten hit "
     "rate.")

body("The more interesting answer is where that signal comes from. The "
     "business-activity counts, the number of entities in a country that merely "
     "hold shares, manage intellectual property, lend within the group or do "
     "nothing at all, are among the most useful inputs available, taking the "
     "top three positions for the neural model and two of the top three for the "
     "tree. That block sits in a free public dataset and, as far as I can "
     "establish, has not been used this way before. If the project contributes "
     "one thing, it is that.")

body("Two secondary findings are worth carrying forward. The deep-learning "
     "comparison came out against the neural models: XGBoost beat both with "
     "statistical significance and was substantially better calibrated. Having "
     "spent most of the implementation effort on the FT-Transformer I would "
     "have preferred a different result, but the literature predicted it for a "
     "dataset this size, and reporting it plainly is more useful than "
     "explaining it away. Separately, the autoencoder's apparent perfect "
     "precision turned out to be an artefact of a cut-off that made only three "
     "predictions. Both are reminders that how the evaluation is designed "
     "mattered more than the modelling.")

body("There is a broader theme here that I did not expect when I started. The "
     "tax-risk work reviewed in section 2.4 is largely built on confidential "
     "data, which makes it authoritative and almost impossible to build on. "
     "Working from a public dataset costs real detail, I can say something "
     "about country pairs but nothing about individual firms, and yet it buys "
     "something the closed studies cannot offer: anyone can download the same "
     "file, run the same pipeline and disagree with me. For a field where the "
     "conclusions are contested and consequential, that seems like a trade "
     "worth making more often.")

body("Four things remain, set out as tasks T2 to T7 in section 3.8. The "
     "label-sensitivity study is the most important, because everything here "
     "rests on a proxy label, and dropping the contested conduit countries or "
     "moving to a positive-unlabelled setup will show how much the conclusions "
     "depend on my particular list. Second, combining the autoencoder with the "
     "supervised score will be tested rather than assumed. Third, a threshold "
     "derived from a stated audit capacity should replace the 0.5 cut-off that "
     "section 5.4 showed to be distorting the comparison. Fourth, the "
     "documentation and tests need tightening so a third party can reproduce "
     "everything from the raw download.")

body("Beyond that, the extension I find most tempting is to move from a yes/no "
     "haven flag to the operating-conduit-sink range that Garcia-Bernardo et "
     "al. (2017) describe and that the activity counts could support. It would "
     "make the output more informative for a policy user and, as far as I know, "
     "has not been tried on this dataset. It is out of scope for the time "
     "remaining, but it is where I would go next.")


# ============================== REFERENCES ============================== #
doc.add_page_break()
heading("References", 1)
_cur["ch"] = None

REFS = [
    "Arik, S.Ö. and Pfister, T. (2021) ‘TabNet: attentive interpretable tabular "
    "learning’, Proceedings of the AAAI Conference on Artificial Intelligence, "
    "35(8), pp. 6679–6687. doi: 10.1609/aaai.v35i8.16826.",

    "Beer, S., de Mooij, R. and Liu, L. (2020) ‘International corporate tax "
    "avoidance: a review of the channels, magnitudes, and blind spots’, Journal "
    "of Economic Surveys, 34(3), pp. 660–688. doi: 10.1111/joes.12305.",

    "Borisov, V., Leemann, T., Seßler, K., Haug, J., Pawelczyk, M. and Kasneci, "
    "G. (2022) ‘Deep neural networks and tabular data: a survey’, IEEE "
    "Transactions on Neural Networks and Learning Systems. "
    "doi: 10.1109/TNNLS.2022.3229161.",

    "Chen, T. and Guestrin, C. (2016) ‘XGBoost: a scalable tree boosting "
    "system’, Proceedings of the 22nd ACM SIGKDD International Conference on "
    "Knowledge Discovery and Data Mining, pp. 785–794. "
    "doi: 10.1145/2939672.2939785.",

    "Clausing, K.A. (2016) ‘The effect of profit shifting on the corporate tax "
    "base in the United States and beyond’, National Tax Journal, 69(4), "
    "pp. 905–934. doi: 10.17310/ntj.2016.4.09.",

    "de Roux, D., Pérez, B., Moreno, A., Villamil, M.d.P. and Figueroa, C. "
    "(2018) ‘Tax fraud detection for under-reporting declarations using an "
    "unsupervised machine learning approach’, Proceedings of the 24th ACM SIGKDD "
    "International Conference on Knowledge Discovery and Data Mining, "
    "pp. 215–222. doi: 10.1145/3219819.3219878.",

    "Garcia-Bernardo, J., Fichtner, J., Takes, F.W. and Heemskerk, E.M. (2017) "
    "‘Uncovering offshore financial centers: conduits and sinks in the global "
    "corporate ownership network’, Scientific Reports, 7, 6246. "
    "doi: 10.1038/s41598-017-06322-9.",

    "Garcia-Bernardo, J. and Janský, P. (2024) ‘Profit shifting of multinational "
    "corporations worldwide’, World Development, 177, 106527. "
    "doi: 10.1016/j.worlddev.2023.106527.",

    "Gorishniy, Y., Rubachev, I., Khrulkov, V. and Babenko, A. (2021) "
    "‘Revisiting deep learning models for tabular data’, Advances in Neural "
    "Information Processing Systems, 34. Available at: "
    "https://arxiv.org/abs/2106.11959.",

    "Grinsztajn, L., Oyallon, E. and Varoquaux, G. (2022) ‘Why do tree-based "
    "models still outperform deep learning on typical tabular data?’, Advances "
    "in Neural Information Processing Systems, 35. Available at: "
    "https://arxiv.org/abs/2207.08815.",

    "Kim, S., Tsai, Y.-C., Singh, K., Choi, Y., Ibok, E., Li, C.-T. and Cha, M. "
    "(2020) ‘DATE: dual attentive tree-aware embedding for customs fraud "
    "detection’, Proceedings of the 26th ACM SIGKDD International Conference on "
    "Knowledge Discovery and Data Mining, pp. 2880–2890. "
    "doi: 10.1145/3394486.3403339.",

    "Lundberg, S.M. and Lee, S.-I. (2017) ‘A unified approach to interpreting "
    "model predictions’, Advances in Neural Information Processing Systems, 30. "
    "Available at: https://arxiv.org/abs/1705.07874.",

    "OECD (2015) Transfer Pricing Documentation and Country-by-Country "
    "Reporting, Action 13 – 2015 Final Report. Paris: OECD Publishing. "
    "doi: 10.1787/9789264241480-en.",

    "OECD (2022) OECD Transfer Pricing Guidelines for Multinational Enterprises "
    "and Tax Administrations 2022. Paris: OECD Publishing. "
    "doi: 10.1787/0e655865-en.",

    "Pang, G., Shen, C., Cao, L. and van den Hengel, A. (2021) ‘Deep learning "
    "for anomaly detection: a review’, ACM Computing Surveys, 54(2), article 38. "
    "doi: 10.1145/3439950.",

    "Popov, S., Morozov, S. and Babenko, A. (2020) ‘Neural oblivious decision "
    "ensembles for deep learning on tabular data’, International Conference on "
    "Learning Representations. Available at: https://arxiv.org/abs/1909.06312.",

    "Shwartz-Ziv, R. and Armon, A. (2022) ‘Tabular data: deep learning is not "
    "all you need’, Information Fusion, 81, pp. 84–90. "
    "doi: 10.1016/j.inffus.2021.11.011.",

    "Tørsløv, T., Wier, L. and Zucman, G. (2023) ‘The missing profits of "
    "nations’, The Review of Economic Studies, 90(3), pp. 1499–1534. "
    "doi: 10.1093/restud/rdac049.",

    "Vanhoeyveld, J., Martens, D. and Peeters, B. (2020) ‘Value-added tax fraud "
    "detection with scalable anomaly detection techniques’, Applied Soft "
    "Computing, 86, 105895. doi: 10.1016/j.asoc.2019.105895.",
]
for ref in REFS:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(7)
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.first_line_indent = Inches(-0.3)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.add_run(ref)


build_toc(_bookmarks, PAGE_MAP, _toc_anchor)
add_page_numbers()
request_field_update()
doc.save(OUT)

CAPS = {"1 Introduction": 1000, "2 Literature review": 2500, "3 Design": 2000,
        "4 Implementation": 2000, "5 Evaluation": 2500, "6 Conclusion": 1000}
print(f"\nSaved -> {OUT}\n")
print(f"{'Chapter':<22}{'words':>8}{'cap':>8}  status")
print("-" * 52)
total = 0
for ch, cap in CAPS.items():
    n = counts.get(ch, 0)
    total += n
    print(f"{ch:<22}{n:>8}{cap:>8}  {'OK' if n <= cap else f'OVER by {n-cap}'}")
print("-" * 52)
print(f"{'TOTAL body':<22}{total:>8}{9500:>8}  "
      f"{'OK' if total <= 9500 else f'OVER by {total-9500}'}")


# --------------------------------------------------------------------------- #
# Optional: rebuild the contents page numbers from an actual render
# --------------------------------------------------------------------------- #
def _remap() -> None:
    """Render the document, read which page each heading lands on, cache that,
    and rebuild so the contents list shows real page numbers."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        print("LibreOffice not found - keeping the existing page map.")
        return
    subprocess.run([soffice, "--headless", "--convert-to", "pdf",
                    str(OUT), "--outdir", str(OUT.parent)],
                   check=True, capture_output=True)
    pdf = OUT.with_suffix(".pdf")
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                         check=True, capture_output=True, text=True).stdout
    pages = txt.split("\f")
    mapping: dict[str, int] = {}
    for level, text, _bid in _bookmarks:
        needle = re.sub(r"\s+", " ", text).strip()
        for i, page in enumerate(pages, start=1):
            flat = re.sub(r"\s+", " ", page)
            if needle in flat:
                mapping[text] = i
                break
    PAGE_MAP_FILE.write_text(json.dumps(mapping, indent=2))
    print(f"\nCached {len(mapping)} contents entries -> {PAGE_MAP_FILE.name}")
    print("Rebuilding with real page numbers...")
    subprocess.run([sys.executable, __file__], check=True)


if "--remap" in sys.argv:
    _remap()
