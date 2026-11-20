"""Daily event-typed Geopolitical Tension Index (session grain)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.dm_pipeline.constants import EVT_PERCENTILE, GTI_HIGH_PERCENTILE, GTI_LAMBDA
from research.dm_pipeline.session import session_span_days


def article_severity(tone: float | None, n_types: int = 1) -> float:
    """Map GKG tone (−100..100) + multi-type boost into severity in [0, 1]."""
    if tone is None or (isinstance(tone, float) and np.isnan(tone)):
        base = 0.4
    else:
        # stronger |tone| → higher severity
        base = float(np.clip(abs(float(tone)) / 10.0, 0.05, 1.0))
    boost = 1.0 + 0.1 * max(0, n_types - 1)
    return float(np.clip(base * boost, 0.0, 1.0))


def daily_shocks(
    articles: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    event_col: str = "core",
    session_col: str = "session",
    tone_col: str = "tone",
) -> pd.DataFrame:
    """Build G_{k,t} = (1/Δ_t) Σ severity for event type k on session t."""
    spans = session_span_days(sessions)
    rows = articles.dropna(subset=[session_col]).copy()
    if rows.empty:
        return pd.DataFrame(index=sessions)

    n_types = rows[event_col].map(lambda c: len(c) if isinstance(c, (list, tuple, set)) else 1)
    rows = rows.copy()
    rows["_sev"] = [article_severity(t, int(n)) for t, n in zip(rows[tone_col], n_types)]

    exploded = rows.explode(event_col).dropna(subset=[event_col])
    raw = exploded.groupby([session_col, event_col])["_sev"].sum().unstack(fill_value=0.0)
    raw = raw.reindex(sessions, fill_value=0.0)
    delta = spans.reindex(sessions).fillna(1).clip(lower=1)
    return raw.div(delta, axis=0)


def decayed_levels(shocks: pd.DataFrame, lam: float = GTI_LAMBDA) -> pd.DataFrame:
    """E_{k,t} = λ E_{k,t-1} + G_{k,t}."""
    levels = shocks.astype(float).copy()
    if levels.empty:
        return levels
    arr = np.array(levels.to_numpy(dtype=float), copy=True)
    for i in range(1, len(arr)):
        arr[i] = lam * arr[i - 1] + arr[i]
    return pd.DataFrame(arr, index=levels.index, columns=levels.columns)


def gti_series(levels: pd.DataFrame) -> pd.Series:
    return levels.sum(axis=1)


def event_items(
    levels: pd.DataFrame,
    train_end: str | pd.Timestamp,
    evt_q: float = EVT_PERCENTILE,
) -> pd.DataFrame:
    """Boolean matrix: k_EVT when E_{k,t} > training q-percentile."""
    train_end = pd.Timestamp(train_end)
    thr = levels.loc[levels.index <= train_end].quantile(evt_q)
    return levels.gt(thr)


def gti_items(
    gti: pd.Series,
    train_end: str | pd.Timestamp,
    high_q: float = GTI_HIGH_PERCENTILE,
) -> pd.DataFrame:
    train_end = pd.Timestamp(train_end)
    thr = float(gti.loc[gti.index <= train_end].quantile(high_q))
    out = pd.DataFrame(index=gti.index)
    out["GTI_HIGH"] = gti > thr
    out["GTI_RISING"] = gti > gti.shift(1)
    return out
