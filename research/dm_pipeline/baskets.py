"""Daily one-hot baskets: news + price items (and lagged _L1 copies)."""

from __future__ import annotations

import pandas as pd

from research.dm_pipeline.constants import SECTORS


def price_items_for_day(states: pd.DataFrame, day: pd.Timestamp, lag: bool = False) -> list[str]:
    suffix = "_L1" if lag else ""
    items: list[str] = []
    if day not in states.index:
        return items
    row = states.loc[day]
    for s in states.columns:
        v = row[s]
        if v in ("UP", "DOWN"):
            items.append(f"{s}_{v}{suffix}")
    return items


def news_items_for_day(
    evt_active: pd.DataFrame,
    gti_flags: pd.DataFrame,
    day: pd.Timestamp,
    lag: bool = False,
) -> list[str]:
    suffix = "_L1" if lag else ""
    items: list[str] = []
    if day in evt_active.index:
        for k, on in evt_active.loc[day].items():
            if bool(on):
                items.append(f"{k}_EVT{suffix}")
    if day in gti_flags.index:
        for k, on in gti_flags.loc[day].items():
            if bool(on):
                items.append(f"{k}{suffix}")
    return items


def build_baskets(
    sessions: pd.DatetimeIndex,
    states: pd.DataFrame,
    evt_active: pd.DataFrame,
    gti_flags: pd.DataFrame,
    sectors: tuple[str, ...] = SECTORS,
) -> list[list[str]]:
    """One basket per session (skip first day: no lag available)."""
    sessions = pd.DatetimeIndex(sessions).sort_values()
    baskets: list[list[str]] = []
    for i, day in enumerate(sessions):
        if i == 0:
            continue
        prev = sessions[i - 1]
        items = []
        items += price_items_for_day(states, day, lag=False)
        items += news_items_for_day(evt_active, gti_flags, day, lag=False)
        items += news_items_for_day(evt_active, gti_flags, prev, lag=True)
        items += price_items_for_day(states, prev, lag=True)
        baskets.append(sorted(set(items)))
    return baskets


def basket_matrix(baskets: list[list[str]]) -> pd.DataFrame:
    """One-hot encode baskets for mlxtend."""
    from mlxtend.preprocessing import TransactionEncoder

    te = TransactionEncoder()
    arr = te.fit(baskets).transform(baskets)
    return pd.DataFrame(arr, columns=te.columns_)


def is_lagged_predictive(antecedent: frozenset[str], consequent: frozenset[str]) -> bool:
    """Lagged rules: only _L1 on the left; day-t price items on the right."""
    if not antecedent or not consequent:
        return False
    if any(not a.endswith("_L1") for a in antecedent):
        return False
    if any(not (c.endswith("_UP") or c.endswith("_DOWN")) or c.endswith("_L1") for c in consequent):
        return False
    # consequent must be same-day price (no _L1, ends with UP/DOWN)
    return all("_EVT" not in c and not c.startswith("GTI_") for c in consequent)


def has_news_antecedent(antecedent: frozenset[str]) -> bool:
    return any(a.endswith("_EVT_L1") or a.startswith("GTI_") for a in antecedent)


def news_gain(conf_full: float, conf_price_only: float | None) -> float:
    """gain = conf(X_P ∪ X_N → Y) − conf(X_P → Y); if X_P empty, second term is supp(Y) handled by caller."""
    if conf_price_only is None:
        return float("nan")
    return conf_full - conf_price_only
