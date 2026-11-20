"""Shared GeoNews-IN constants (aligned with Submission-1 pilots)."""

from __future__ import annotations

CORE_THEMES: dict[str, set[str]] = {
    "CONFLICT": {"ARMEDCONFLICT", "MILITARY", "CEASEFIRE"},
    "TERROR": {"TERROR"},
    "SANCTION": {"SANCTIONS"},
    "TRADE": {"ECON_TRADE_DISPUTE", "EPU_CATS_TRADE_POLICY", "ECON_FREETRADE"},
    "ENERGY": {"ECON_OILPRICE"},
    "SHIPPING": {"MARITIME_INCIDENT", "MARITIME_PIRACY", "BLOCKADE"},
    "DIPLO": {"EPU_CATS_NATIONAL_SECURITY"},
    "POLITICS": {"ELECTION"},
    "MONETARY": {"ECON_CENTRALBANK", "ECON_INTEREST_RATES"},
    "TECH": {"CYBER_ATTACK"},
}

ECON_THEMES: set[str] = {
    "ECON_STOCKMARKET",
    "EPU_ECONOMY",
    "EPU_ECONOMY_HISTORIC",
    "ECON_OILPRICE",
    "ECON_CENTRALBANK",
    "ECON_INTEREST_RATES",
    "ECON_TRADE_DISPUTE",
    "EPU_CATS_TRADE_POLICY",
}

WHITELIST_SOURCES: set[str] = {
    "indiatimes.com",
    "business-standard.com",
    "livemint.com",
    "moneycontrol.com",
    "thehindubusinessline.com",
    "financialexpress.com",
    "reuters.com",
    "bloomberg.com",
    "cnbc.com",
    "ft.com",
    "nytimes.com",
    "economist.com",
    "theguardian.com",
    "bbc.com",
    "bbc.co.uk",
    "dw.com",
    "aljazeera.com",
    "scmp.com",
    "oilprice.com",
    "eia.gov",
    "thehindu.com",
    "hindustantimes.com",
}

SECTORS: tuple[str, ...] = (
    "IT",
    "Bank",
    "Pharma",
    "Auto",
    "FMCG",
    "Metal",
    "Realty",
    "Energy",
    "PSUBank",
    "PvtBank",
    "Media",
    "Infra",
)

# Same 17 RSS domains as GeoTrade DEFAULT_ADAPTERS (cross-check layer).
RSS_CROSSCHECK_DOMAINS: tuple[str, ...] = (
    "reuters.com",
    "bbc.com",
    "bbc.co.uk",
    "ft.com",
    "bloomberg.com",
    "cnbc.com",
    "aljazeera.com",
    "theguardian.com",
    "nytimes.com",
    "economist.com",
    "dw.com",
    "scmp.com",
    "indiatimes.com",
    "thehindu.com",
    "livemint.com",
    "moneycontrol.com",
    "oilprice.com",
)

SESSION_CLOSE_MINUTES = 15 * 60 + 30  # 15:30 IST
GTI_LAMBDA = 0.7
EVT_PERCENTILE = 0.75
GTI_HIGH_PERCENTILE = 0.80
PRICE_EPS_FRAC = 0.1
MIN_SUPPORT = 0.08
MIN_CONFIDENCE = 0.55
MIN_LIFT = 1.2
MAX_ANTECEDENT = 3
BLOCK_LEN = 10
N_PERM = 1000  # default; raise to 10_000 for final report runs
FDR_Q = 0.10
STABILITY_DELTA = 0.10
