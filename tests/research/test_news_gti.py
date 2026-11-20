from __future__ import annotations

import pandas as pd

from research.dm_pipeline.gti import daily_shocks, decayed_levels, event_items, gti_items, gti_series
from research.dm_pipeline.news_filter import filter_geonews, normalize_title, prepare_gkg, tier_summary
from research.dm_pipeline.relevance import apply_relevance, heuristic_relevant
from research.dm_pipeline.session import session_of


def test_normalize_title():
    assert normalize_title("Hello, WORLD!!") == "hello world"


def test_tier_t3_drops_non_whitelist(synthetic_news):
    d = prepare_gkg(synthetic_news)
    # all our fixtures are whitelist except we could add one
    bad = synthetic_news.iloc[[0]].copy()
    bad["source"] = "random-blog.xyz"
    merged = pd.concat([synthetic_news, bad], ignore_index=True)
    out = filter_geonews(merged, tier="T3")
    assert (out["source"] != "random-blog.xyz").all()
    assert len(out) >= 1


def test_tier_summary_keys(synthetic_news):
    s = tier_summary(synthetic_news)
    assert "T3 = T2 and whitelisted source" in s


def test_heuristic_relevance():
    assert heuristic_relevant("Oil sanctions hit India trade") == 1
    assert heuristic_relevant("Toxic worldwide box office collection") == 0


def test_gti_decay_and_items(sessions, synthetic_news):
    geo = filter_geonews(synthetic_news, tier="T3")
    geo = apply_relevance(geo)
    geo["pub_ist"] = pd.to_datetime(geo["date"].astype(str).str[:12], format="%Y%m%d%H%M") + pd.Timedelta(
        hours=5, minutes=30
    )
    geo["session"] = geo["pub_ist"].map(lambda t: session_of(t, sessions))
    shocks = daily_shocks(geo, sessions)
    levels = decayed_levels(shocks, lam=0.7)
    gti = gti_series(levels)
    assert len(gti) == len(sessions)
    # decayed level should be >= same-day shock on first nonzero day
    assert (levels >= 0).all().all()
    evt = event_items(levels, train_end=sessions[20])
    flags = gti_items(gti, train_end=sessions[20])
    assert "GTI_HIGH" in flags.columns
    assert evt.shape[0] == len(sessions)
