"""Apriori GeoRule mining with lag / news filters."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from research.dm_pipeline.baskets import has_news_antecedent, is_lagged_predictive
from research.dm_pipeline.constants import MAX_ANTECEDENT, MIN_CONFIDENCE, MIN_LIFT, MIN_SUPPORT


@dataclass
class Rule:
    antecedents: frozenset[str]
    consequents: frozenset[str]
    support: float
    confidence: float
    lift: float
    conviction: float | None = None

    @property
    def key(self) -> tuple:
        return (self.antecedents, self.consequents)


def mine_rules(
    onehot: pd.DataFrame,
    min_support: float = MIN_SUPPORT,
    min_confidence: float = MIN_CONFIDENCE,
    min_lift: float = MIN_LIFT,
    max_len: int = MAX_ANTECEDENT + 1,
) -> list[Rule]:
    if onehot.empty or onehot.shape[0] < 5:
        return []
    # drop ultra-rare columns to keep Apriori tractable
    freq = onehot.mean()
    keep = freq[freq >= min_support * 0.5].index
    mat = onehot[keep]
    if mat.shape[1] == 0:
        return []
    freq_itemsets = apriori(mat, min_support=min_support, use_colnames=True, max_len=max_len)
    if freq_itemsets.empty:
        return []
    rules_df = association_rules(freq_itemsets, metric="confidence", min_threshold=min_confidence)
    if rules_df.empty:
        return []
    rules_df = rules_df[rules_df["lift"] >= min_lift]
    out: list[Rule] = []
    for _, row in rules_df.iterrows():
        ant = frozenset(row["antecedents"])
        cons = frozenset(row["consequents"])
        if len(ant) > MAX_ANTECEDENT:
            continue
        out.append(
            Rule(
                antecedents=ant,
                consequents=cons,
                support=float(row["support"]),
                confidence=float(row["confidence"]),
                lift=float(row["lift"]),
                conviction=float(row["conviction"]) if "conviction" in row and pd.notna(row["conviction"]) else None,
            )
        )
    return out


def filter_lagged_news_rules(rules: list[Rule], max_antecedent: int = 2) -> list[Rule]:
    """Family tested for significance: lagged, ≥1 news antecedent, ≤2 antecedents."""
    kept = []
    for r in rules:
        if len(r.antecedents) > max_antecedent:
            continue
        if not is_lagged_predictive(r.antecedents, r.consequents):
            continue
        if not has_news_antecedent(r.antecedents):
            continue
        kept.append(r)
    return kept


def minsupp_for_cell(n_days: int, floor: float = MIN_SUPPORT, min_cooccur: int = 20) -> float | None:
    if n_days < 120:
        return None
    return max(floor, min_cooccur / n_days)


def rules_to_frame(rules: list[Rule]) -> pd.DataFrame:
    rows = []
    for r in rules:
        rows.append(
            {
                "antecedents": ",".join(sorted(r.antecedents)),
                "consequents": ",".join(sorted(r.consequents)),
                "support": r.support,
                "confidence": r.confidence,
                "lift": r.lift,
            }
        )
    return pd.DataFrame(rows)
