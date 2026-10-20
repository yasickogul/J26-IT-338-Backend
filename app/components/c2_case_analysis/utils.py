"""Shared helpers (notebook cells 3 and 16: safe_list, keyword lists, is_civil)."""
import re


def safe_list(x):
    """Safely convert a possibly-numpy-array / None field into a plain list."""
    if x is None:
        return []
    if isinstance(x, float):  # handles NaN
        return []
    try:
        return list(x)
    except TypeError:
        return []


CRIMINAL_KEYWORDS = [
    "penal code", "criminal procedure code", "prosecution", "accused",
    "convicted", "conviction", "acquitted", "sentence of imprisonment",
    "bail act", "indictment", "excise ordinance", "dangerous drugs",
    "offence", "charge sheet",
]

CIVIL_KEYWORDS = [
    "plaintiff", "defendant", "appellant", "respondent", "damages",
    "contract", "breach", "mortgage", "partition", "trust", "property",
    "succession", "tenancy", "rent act", "company", "partnership",
    "negligence", "tort", "specific performance", "injunction",
    "declaration of title", "prescription", "thesawalamai",
]


def is_civil(row):
    laws_list = safe_list(row["laws_referred"])
    laws = " ".join([str(l.get("law", "") or "") for l in laws_list])
    text = " ".join([
        str(row.get("case_title", "")),
        str(row.get("short_summary", "")),
        str(row.get("full_summary", "")),
        laws,
    ]).lower()
    civil_score = sum(k in text for k in CIVIL_KEYWORDS)
    criminal_score = sum(k in text for k in CRIMINAL_KEYWORDS)
    return civil_score > criminal_score


def strip_json_fences(raw):
    """Remove a leading ```json and trailing ``` fence if the model added one."""
    return re.sub(r"^```json\s*|\s*```$", "", (raw or "").strip()).strip()
