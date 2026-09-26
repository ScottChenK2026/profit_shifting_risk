"""
reference_data.py
-----------------
The lookup tables. Two kinds of outside knowledge live here: the list of countries I treat as tax
havens, which is used to build the answer the model tries to predict, and the mapping from OECD's
rather cryptic measure codes to readable column names. None of it comes from the companies' own
reported numbers, which is the point - defining the answer this way cannot accidentally hand the
model that answer through the inputs.

About the haven list
--------------------
A tax haven here just means a country widely used to book profits where very little tax is paid.
No single official list exists that everyone agrees on, so what I use is a best-effort proxy: a
country counts as a haven if it turns up on the well-known lists. I combined three sources to get
reasonable coverage:

1. the EU's lists of non-cooperative jurisdictions, both the blacklist and the grey watch list;
2. the countries near the top of the Tax Justice Network's secrecy and corporate-tax-haven
    rankings;
3. the offshore financial centres identified by Garcia-Bernardo et al. (2017), "Uncovering
    Offshore Financial Centers", covering both the end destinations they call sinks and the big
    pass-through conduits.

It is an approximation and I am upfront about that in the write-up, where I also discuss what it
gets wrong. Codes are ISO-3166 three-letter country codes (LUX for Luxembourg, and so on) so they
line up with the way the OECD data labels countries.
"""

from __future__ import annotations

# --- The haven list itself, as three-letter country codes ----------------- #
# Grouped roughly by region, only so it is easier to scan and check by eye.
TAX_HAVENS: set[str] = {
    # Caribbean / Atlantic
    "AIA", "ATG", "ABW", "BHS", "BRB", "BLZ", "BMU", "VGB", "CYM", "CUW",
    "DMA", "GRD", "MSR", "KNA", "LCA", "VCT", "SXM", "TCA", "ANT",
    # Europe, including the major conduits
    "AND", "CYP", "GIB", "GGY", "IRL", "IMN", "JEY", "LIE", "LUX", "MLT",
    "MCO", "NLD", "SMR", "CHE",
    # Asia / Pacific / Middle East / Africa / Indian Ocean
    "BHR", "HKG", "LBN", "MAC", "MDV", "MHL", "MUS", "NRU", "PAN", "WSM",
    "SYC", "SGP", "VUT", "ARE", "LBR", "COK", "PLW",
}

# --- What the local offices actually do ----------------------------------- #
# The OECD also reports how many establishments in each country mainly do each kind of activity,
# and for this project that is gold. Some types are the classic paper-office setup: a holding
# company that only owns shares, an entity that only lends money around the group, one that holds
# patents and collects royalties, a dormant shell that does nothing at all. Those cluster where
# profit is being shifted. Others - factories, sales, R&D - mean real business is happening on the
# ground. OECD's code on the left, the friendlier name I use everywhere else on the right.
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

# The two buckets those activity types fall into: the paper-office kinds that tend to go with
# shifting, and the ones that mean actual operations are happening locally.
SHIFTING_PRONE_ACTIVITIES = [
    "holding_equity", "ip_management", "internal_group_finance", "dormant",
]
REAL_ACTIVITIES = [
    "research_development", "manufacturing", "sales_marketing",
    "services_unrelated", "purchasing",
]

# --- The money and headcount figures -------------------------------------- #
# Same idea as the activity map, but for the financial numbers and the counts: revenue, profit,
# tax, employees. OECD's code on the left, my column name on the right.
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

# Some counterpart codes in the data are not countries at all - they are roll-ups like "rest of
# world", regional totals, or stateless and unknown. They have to go, otherwise we would end up
# comparing a real country against a regional bucket.
COUNTERPART_AGGREGATES: set[str] = {
    "W", "E", "A", "S", "F", "W_O", "E_O", "A_O", "S_O", "F_O",
    "STLS", "ANT_F", "FJT", "WLD", "_T", "ZZZ",
}


def is_real_jurisdiction(code: str) -> bool:
    """Does this counterpart code look like a genuine single country rather than one of the
    aggregate buckets? A real one is three capital letters, like "FRA", and is not on the exclude
    list above."""
    return (isinstance(code, str) and len(code) == 3 and code.isalpha()
            and code.upper() == code and code not in COUNTERPART_AGGREGATES)


# --- Alternative haven definitions, for the sensitivity study --------------- #
# The list above is a judgement call, and the write-up is candid that the whole project rests on
# it. Two of the arguments against it are worth testing rather than just conceding.
#
# The first is that several countries on the list are contested. The Netherlands, Switzerland,
# Ireland and Singapore are conduits in Garcia-Bernardo et al. (2017) rather than end destinations:
# real businesses operate there, and calling them havens is defensible but arguable.
# CONTESTED_CONDUITS names them so src/label_sensitivity.py can drop them and see what changes.
#
# The second is that the list mixes two different things - the sinks where profit comes to rest and
# the conduits it travels through. STRICT_SINKS keeps only the former, which is the narrowest
# reading anyone would accept. It was expected to be the harder task, since the best-known conduits
# are gone; in the event it scored almost exactly what the full list did.
CONTESTED_CONDUITS: set[str] = {"NLD", "CHE", "IRL", "SGP", "GBR"}

STRICT_SINKS: set[str] = TAX_HAVENS - CONTESTED_CONDUITS


def haven_set(variant: str = "baseline") -> set[str]:
    """Return the set of jurisdiction codes counted as havens under a given labelling.

    "baseline" is the combined list used throughout the main results. "strict_sinks" drops the
    contested conduits. "conduits_only" keeps just those contested cases that appear on the
    baseline list, which is a deliberately awkward variant: if the model can still separate them
    from ordinary countries, the activity mix is picking up something about structure rather than
    merely re-learning the obvious cases.
    """
    if variant == "baseline":
        return set(TAX_HAVENS)
    if variant == "strict_sinks":
        return set(STRICT_SINKS)
    if variant == "conduits_only":
        return set(CONTESTED_CONDUITS) & set(TAX_HAVENS)
    raise ValueError(f"unknown label variant: {variant!r}")
