from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sessions() -> pd.DatetimeIndex:
    # 40 weekdays
    return pd.bdate_range("2024-01-02", periods=40)


@pytest.fixture
def synthetic_closes(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = ["Nifty50", "IT", "Bank", "Pharma", "Auto", "FMCG", "Metal", "Realty", "Energy", "PSUBank", "PvtBank", "Media", "Infra"]
    data = 1000 + np.cumsum(rng.normal(0, 5, size=(len(sessions), len(cols))), axis=0)
    return pd.DataFrame(data, index=sessions, columns=cols)


@pytest.fixture
def synthetic_news(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Minimal GKG-like rows covering a few sessions."""
    rows = []
    themes = {
        "CONFLICT": "ARMEDCONFLICT;MILITARY",
        "ENERGY": "ECON_OILPRICE;ECON_STOCKMARKET",
        "TRADE": "ECON_TRADE_DISPUTE;EPU_CATS_TRADE_POLICY",
    }
    sources = ["reuters.com", "livemint.com", "bbc.com"]
    for i, day in enumerate(sessions[:30]):
        for k, th in themes.items():
            if i % 3 == 0 or k == "ENERGY":
                stamp = day.strftime("%Y%m%d") + "100000"
                rows.append(
                    {
                        "date": stamp,
                        "source": sources[i % 3],
                        "url": f"https://{sources[i % 3]}/a/{i}-{k}",
                        "title": f"India {k.lower()} oil sanction conflict update {i}",
                        "themes": th,
                        "tone": float((-1) ** i * (i % 7)),
                        "india": True,
                    }
                )
    # inject one off-topic
    rows.append(
        {
            "date": sessions[5].strftime("%Y%m%d") + "120000",
            "source": "hindustantimes.com",
            "url": "https://hindustantimes.com/box",
            "title": "Toxic worldwide box office collection day 6",
            "themes": "ELECTION",
            "tone": 0.0,
            "india": True,
        }
    )
    return pd.DataFrame(rows)
