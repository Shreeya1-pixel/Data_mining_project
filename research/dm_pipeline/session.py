"""NSE session mapping (IST close 15:30)."""

from __future__ import annotations

import pandas as pd


def session_of(ts: pd.Timestamp, sessions: pd.DatetimeIndex, close_minutes: int = 15 * 60 + 30) -> pd.Timestamp:
    """Map a publication timestamp to the trading session that can still react to it.

    Same-day if on/before close on a session day; otherwise the next session.
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("Asia/Kolkata").tz_localize(None)
    day = ts.normalize()
    mins = ts.hour * 60 + ts.minute
    if day in sessions and mins <= close_minutes:
        return day
    later = sessions[sessions > day]
    if len(later) == 0:
        return pd.NaT
    return later[0]


def session_span_days(sessions: pd.DatetimeIndex) -> pd.Series:
    """Calendar days since previous session (Δ_t), used to normalise weekend news piles."""
    idx = list(sessions)
    spans = {}
    for i, t in enumerate(idx):
        if i == 0:
            spans[t] = 1
        else:
            spans[t] = max(1, (t - idx[i - 1]).days)
    return pd.Series(spans)
