"""Nifty sector price states and regimes."""

from __future__ import annotations

import pandas as pd

from research.dm_pipeline.constants import PRICE_EPS_FRAC, SECTORS


def load_closes(path: str) -> pd.DataFrame:
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    px.index = pd.DatetimeIndex(px.index).normalize()
    return px


def returns(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change()


def train_sigma(ret: pd.DataFrame, train_end: str | pd.Timestamp) -> pd.Series:
    return ret[ret.index <= pd.Timestamp(train_end)].std()


def price_states(
    ret: pd.DataFrame,
    sigma: pd.Series,
    sectors: tuple[str, ...] = SECTORS,
    eps_frac: float = PRICE_EPS_FRAC,
) -> pd.DataFrame:
    """Return DataFrame of UP/DOWN/None per sector."""
    out = pd.DataFrame(index=ret.index, columns=list(sectors), dtype=object)
    for s in sectors:
        if s not in ret.columns:
            continue
        eps = eps_frac * float(sigma.get(s, ret[s].std()))
        r = ret[s]
        out[s] = None
        out.loc[r > eps, s] = "UP"
        out.loc[r < -eps, s] = "DOWN"
    return out


def market_regime(nifty: pd.Series, sma_window: int = 50, slope_window: int = 5) -> pd.Series:
    """Bull / bear / sideways from Nifty50 vs SMA50 and short slope."""
    sma = nifty.rolling(sma_window, min_periods=sma_window).mean()
    slope = nifty.diff(slope_window)
    regime = pd.Series("sideways", index=nifty.index, dtype=object)
    above = nifty > sma
    below = nifty < sma
    regime.loc[above & (slope >= 0)] = "bull"
    regime.loc[below & (slope <= 0)] = "bear"
    regime.loc[sma.isna()] = None
    return regime
