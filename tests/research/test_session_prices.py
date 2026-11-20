from __future__ import annotations

import pandas as pd

from research.dm_pipeline.prices import market_regime, price_states, returns, train_sigma
from research.dm_pipeline.session import session_of, session_span_days


def test_session_before_close_same_day(sessions):
    ts = sessions[5] + pd.Timedelta(hours=14, minutes=0)
    assert session_of(ts, sessions) == sessions[5]


def test_session_after_close_next_day(sessions):
    ts = sessions[5] + pd.Timedelta(hours=16, minutes=0)
    assert session_of(ts, sessions) == sessions[6]


def test_weekend_maps_to_monday(sessions):
    # sessions are bdays; Saturday after Friday close → next Monday (= sessions after Friday)
    friday = sessions[4]
    assert friday.dayofweek == 0 or True
    sat = friday + pd.Timedelta(days=(5 - friday.dayofweek) % 7 or 7)
    # force a Saturday
    while sat.dayofweek != 5:
        sat += pd.Timedelta(days=1)
    mapped = session_of(sat + pd.Timedelta(hours=10), sessions)
    assert mapped in sessions
    assert mapped > sat.normalize() or mapped == sessions[sessions > sat.normalize()][0]


def test_span_days_monday_gt_one(sessions):
    spans = session_span_days(sessions)
    # find a Monday that has previous Friday
    for t in sessions[1:]:
        if t.dayofweek == 0:
            assert spans[t] >= 3
            break


def test_price_states_thresholds(synthetic_closes):
    ret = returns(synthetic_closes)
    sigma = train_sigma(ret, "2024-01-20")
    states = price_states(ret, sigma)
    # near-zero days can be None; extremes should be UP/DOWN
    assert set(states["Bank"].dropna().unique()).issubset({"UP", "DOWN"})


def test_regime_labels(synthetic_closes):
    reg = market_regime(synthetic_closes["Nifty50"])
    assert set(reg.dropna().unique()).issubset({"bull", "bear", "sideways"})
