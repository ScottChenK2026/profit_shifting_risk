"""
reference_data.py
-----------------
This is the "lookup tables" file. It holds two kinds of outside knowledge:
the list of countries I'm treating as tax havens (used to build the answer the
model tries to predict), and the mapping from OECD's cryptic measure codes to
readable column names. Crucially, none of this comes from the companies' own
reported numbers, so using it to define the answer doesn't accidentally hand
the model the answer through the inputs (that would be leakage).

About the tax-haven list
------------------------
A "tax haven" here just means a country widely used to book profits where very
little tax gets paid. There's no single official list everyone agrees on, so
what I use is a best-effort stand-in (a proxy): a country counts as a haven if
it shows up on the well-known lists. I combined a few sources to be reasonably
comprehensive:
  * the EU's own lists of "non-cooperative" countries (the blacklist and the
    watch/"grey" list);
  * the countries near the top of the Tax Justice Network's secrecy and
    corporate-tax-haven rankings;
  * the offshore financial centres flagged by Garcia-Bernardo et al. (2017),
    "Uncovering Offshore Financial Centers" - both the end destinations
    ("sinks") and the big pass-through hubs ("conduits").

It's an approximation, not gospel, and I talk about where it falls short in the
write-up. The codes are ISO-3166 three-letter country codes (e.g. "LUX" for
Luxembourg) so they line up with how the OECD data labels countries.
"""

from __future__ import annotations

# --- The haven list itself (three-letter country codes) ------------------- #
# Grouped roughly by region just to make it easier to scan and check by eye.
TAX_HAVENS: set[str] = {
    # Caribbean / Atlantic
    "AIA", "ATG", "ABW", "BHS", "BRB", "BLZ", "BMU", "VGB", "CYM", "CUW",
    "DMA", "GRD", "MSR", "KNA", "LCA", "VCT", "SXM", "TCA", "ANT",
    # Europe (incl. major conduits)
    "AND", "CYP", "GIB", "GGY", "IRL", "IMN", "JEY", "LIE", "LUX", "MLT",
    "MCO", "NLD", "SMR", "CHE",
    # Asia / Pacific / Middle East / Africa / Indian Ocean
    "BHR", "HKG", "LBN", "MAC", "MDV", "MHL", "MUS", "NRU", "PAN", "WSM",
    "SYC", "SGP", "VUT", "ARE", "LBR", "COK", "PLW",
}

# --- What the local offices actually do ----------------------------------- #
# The OECD also reports how many establishments (offices/entities) in each
# country mainly do each kind of activity. This is gold for us: some types are
# classic "paper office" setups - a holding company that just owns shares, an
# entity that only lends money around the group, one that just holds patents
# and collects royalties, or a dormant shell that does nothing - and these tend
# to cluster where profit is being shifted. Others (factories, sales, R&D) are
# signs of real business happening on the ground. The left side is OECD's code,
# the right is the friendlier name I use everywhere else.
ACTIVITY_CODES: dict[str, str] = {
    "ACT_RD": "research_development",
    "ACT_HOLDING": "holding_equity",
    "ACT_IP": "ip_management",
    "ACT_PUR": "purchasing",
    "ACT_MAN": "manufacturing",
    "ACT_SALES": "sales_marketing",
    "ACT_ADMIN": "admin_support",
    "ACT_FIN": "regulated_finance",
    "ACT_IGF": "internal_group_finance",
    "ACT_INS": "insurance",
    "ACT_SER": "services_unrelated",
    "ACT_DOR": "dormant",
    "ACT_OTHER": "other_activity",
}

# Splitting the activity types into the two buckets above: the "paper office"
# kinds that often go hand-in-hand with shifting, versus the ones that mean
# actual operations are happening locally.
SHIFTING_PRONE_ACTIVITIES = [
    "holding_equity", "ip_management", "internal_group_finance", "dormant",
]
REAL_ACTIVITIES = [
    "research_development", "manufacturing", "sales_marketing",
    "services_unrelated", "purchasing",
]

# --- The money and headcount figures -------------------------------------- #
# Same idea as the activity map above, but for the financial numbers and counts
# (revenue, profit, tax, employees, and so on). OECD's code on the left, my
# column name on the right.
FINANCIAL_CODES: dict[str, str] = {
    "TOT_REV": "total_revenues",
    "RPR": "related_party_revenues",
    "UPR": "unrelated_party_revenues",
    "PROFIT": "profit_before_tax",
    "PROFIT_ADJ": "adjusted_profit_before_tax",
    "TAX_PAID": "income_tax_paid",
    "TAX_ACCRUED": "income_tax_accrued",
    "STATED_CAPITAL": "stated_capital",
    "EARNINGS": "accumulated_earnings",
    "ASSETS": "tangible_assets",
    "EMPLOYEES": "num_employees",
    "CBCR_COUNT": "num_mne_groups",
    "SUBGROUPS_COUNT": "num_subgroups",
    "ENTITIES_COUNT": "num_entities",
}

# Some "counterpart" codes in the data aren't actual countries - they're
# roll-ups like "rest of world", regional totals, or stateless/unknown. We need
# to drop these so we don't compare a real country against a regional bucket.
COUNTERPART_AGGREGATES: set[str] = {
    "W", "E", "A", "S", "F", "W_O", "E_O", "A_O", "S_O", "F_O",
    "STLS", "ANT_F", "FJT", "WLD", "_T", "ZZZ",
}


def is_real_jurisdiction(code: str) -> bool:
    """Quick check for whether a counterpart code looks like a genuine single
    country rather than one of those aggregate buckets. A real one is three
    capital letters (like "FRA") and isn't in the exclude list above."""
    return (isinstance(code, str) and len(code) == 3 and code.isalpha()
            and code.upper() == code and code not in COUNTERPART_AGGREGATES)
