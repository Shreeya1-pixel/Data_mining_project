"""GKG → GeoNews-IN filtering (T1–T3) and theme tagging."""

from __future__ import annotations

import hashlib
import re

import pandas as pd

from research.dm_pipeline.constants import CORE_THEMES, ECON_THEMES, WHITELIST_SOURCES

_TITLE_NORM = re.compile(r"[^a-z0-9 ]")


def parse_themes(themes: str | float | None) -> set[str]:
    if themes is None or (isinstance(themes, float) and pd.isna(themes)):
        return set()
    return {y.split(",")[0] for y in str(themes).split(";") if y}


def core_types(theme_set: set[str]) -> list[str]:
    return sorted(k for k, v in CORE_THEMES.items() if theme_set & v)


def normalize_title(title: str | float | None) -> str:
    if title is None or (isinstance(title, float) and pd.isna(title)):
        return ""
    t = str(title).lower()
    t = _TITLE_NORM.sub("", t)
    return " ".join(t.split())


def title_url_hash(title: str, url: str | None = None) -> str:
    key = normalize_title(title) + "|" + (url or "")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def prepare_gkg(df: pd.DataFrame) -> pd.DataFrame:
    """Add theme flags used by tier filters. Expects columns: themes, india, source, title, date."""
    d = df.copy()
    d["th"] = d["themes"].map(parse_themes)
    d["core"] = d["th"].map(core_types)
    d["has_core"] = d["core"].map(bool)
    d["econ"] = d["th"].map(lambda s: bool(s & ECON_THEMES))
    d["wl"] = d["source"].isin(WHITELIST_SOURCES)
    d["day"] = d["date"].astype(str).str[:8]
    d["ntitle"] = d["title"].map(normalize_title)
    d["row_hash"] = [title_url_hash(t, u) for t, u in zip(d["title"], d.get("url", pd.Series([None] * len(d))))]
    return d


def apply_tier(d: pd.DataFrame, tier: str = "T3") -> pd.Series:
    india = d["india"].astype(bool)
    if tier == "T1":
        return india
    if tier == "T2":
        return d["has_core"] & (india | d["econ"])
    if tier == "T3":
        return d["has_core"] & (india | d["econ"]) & d["wl"]
    if tier == "T4":
        return d["has_core"] & india & d["econ"]
    raise ValueError(f"unknown tier {tier}")


def filter_geonews(df: pd.DataFrame, tier: str = "T3", dedup: bool = True) -> pd.DataFrame:
    d = prepare_gkg(df)
    mask = apply_tier(d, tier)
    out = d.loc[mask].copy()
    if dedup:
        out = out.drop_duplicates(["day", "ntitle"])
    return out.reset_index(drop=True)


def tier_summary(df: pd.DataFrame, scale: float = 1.0) -> dict:
    d = prepare_gkg(df)
    out = {}
    for name, tier in [
        ("T1 India + any broad theme", "T1"),
        ("T2 core geopolitical theme, (India or econ)", "T2"),
        ("T3 = T2 and whitelisted source", "T3"),
        ("T4 = T2 and (India and econ)", "T4"),
    ]:
        sub = d.loc[apply_tier(d, tier)]
        dedup = sub.drop_duplicates(["day", "ntitle"])
        out[name] = {
            "pilot_rows": int(len(sub)),
            "month_est": int(len(sub) * scale),
            "month_est_after_title_dedup": int(len(dedup) * scale),
            "title_dup_rate": round(1 - len(dedup) / max(len(sub), 1), 3),
        }
    return out
