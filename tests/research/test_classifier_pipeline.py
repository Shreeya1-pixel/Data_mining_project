from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.dm_pipeline.baskets import basket_matrix, build_baskets
from research.dm_pipeline.classifier import GeoRuleClassifier, compare_classifiers, target_labels
from research.dm_pipeline.gti import daily_shocks, decayed_levels, event_items, gti_items, gti_series
from research.dm_pipeline.mining import Rule, filter_lagged_news_rules, mine_rules
from research.dm_pipeline.news_filter import filter_geonews
from research.dm_pipeline.pipeline import run_experiment
from research.dm_pipeline.prices import load_closes, price_states, returns, train_sigma
from research.dm_pipeline.relevance import apply_relevance
from research.dm_pipeline.session import session_of


def test_georule_predicts_from_match():
    rules = [
        Rule(
            antecedents=frozenset({"CONFLICT_EVT_L1"}),
            consequents=frozenset({"Bank_DOWN"}),
            support=0.2,
            confidence=0.8,
            lift=2.0,
        )
    ]
    clf = GeoRuleClassifier(rules, "Bank", default_label="FLAT")
    clf.fit_majority(["UP", "UP", "DOWN"])
    assert clf.predict_one({"CONFLICT_EVT_L1", "IT_UP"}) == "DOWN"
    assert clf.predict_one({"IT_UP_L1"}) in {"UP", "DOWN", "FLAT"}


def test_compare_classifiers_runs():
    rng = np.random.default_rng(0)
    train, test = [], []
    for i in range(80):
        b = []
        if rng.random() < 0.4:
            b.append("CONFLICT_EVT_L1")
            b.append("Bank_DOWN" if rng.random() < 0.7 else "Bank_UP")
        else:
            b.append("Bank_UP" if rng.random() < 0.5 else "FLAT_PLACEHOLDER")
            if "FLAT_PLACEHOLDER" in b:
                b.remove("FLAT_PLACEHOLDER")
        b.append("IT_UP_L1")
        train.append(b)
    for i in range(40):
        b = ["IT_UP_L1"]
        if rng.random() < 0.4:
            b.append("CONFLICT_EVT_L1")
            b.append("Bank_DOWN" if rng.random() < 0.7 else "Bank_UP")
        else:
            if rng.random() < 0.5:
                b.append("Bank_UP")
        test.append(b)
    rules = filter_lagged_news_rules(mine_rules(basket_matrix(train), min_support=0.05, min_confidence=0.5, min_lift=1.0))
    table = compare_classifiers(train, test, rules, "Bank")
    assert "georule" in set(table["name"])
    assert len(table) >= 5


def test_end_to_end_synthetic(tmp_path, sessions, synthetic_closes, synthetic_news):
    closes = tmp_path / "closes.csv"
    synthetic_closes.to_csv(closes)
    news_path = tmp_path / "news.parquet"
    synthetic_news.to_parquet(news_path)

    summary = run_experiment(
        closes_path=closes,
        news_paths=[news_path],
        artifacts_dir=tmp_path / "art",
        train_end=str(sessions[25].date()),
        holdout_start=str(sessions[30].date()),
        n_perm=20,
        sectors=("Bank", "IT"),
    )
    assert summary["n_baskets"] > 0
    assert (tmp_path / "art" / "summary.json").exists()
    assert (tmp_path / "art" / "classifier_comparison.csv").exists()


def test_build_baskets_includes_lags(sessions, synthetic_closes, synthetic_news):
    ret = returns(synthetic_closes)
    sigma = train_sigma(ret, sessions[15])
    states = price_states(ret, sigma)
    geo = apply_relevance(filter_geonews(synthetic_news))
    geo["pub_ist"] = pd.to_datetime(geo["date"].astype(str).str[:12], format="%Y%m%d%H%M") + pd.Timedelta(
        hours=5, minutes=30
    )
    geo["session"] = geo["pub_ist"].map(lambda t: session_of(t, sessions))
    shocks = daily_shocks(geo, sessions)
    levels = decayed_levels(shocks)
    evt = event_items(levels, sessions[15])
    gflags = gti_items(gti_series(levels), sessions[15])
    baskets = build_baskets(sessions, states, evt, gflags)
    assert any(any(x.endswith("_L1") for x in b) for b in baskets)
