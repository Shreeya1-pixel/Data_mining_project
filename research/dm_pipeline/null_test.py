"""News-column block-permutation null (prices fixed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.dm_pipeline.constants import BLOCK_LEN, N_PERM
from research.dm_pipeline.mining import Rule, filter_lagged_news_rules, mine_rules


def _is_news_col(name: str) -> bool:
    return (
        name.endswith("_EVT")
        or name.endswith("_EVT_L1")
        or name.startswith("GTI_")
        or name.endswith("_NEWS_POS")
        or name.endswith("_NEWS_NEG")
        or name.endswith("_NEWS_POS_L1")
        or name.endswith("_NEWS_NEG_L1")
    )


def news_column_mask(columns: pd.Index) -> np.ndarray:
    return np.array([_is_news_col(c) for c in columns], dtype=bool)


def block_permute_rows(mat: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Shuffle contiguous blocks of rows (news columns only applied by caller)."""
    n = mat.shape[0]
    if n == 0:
        return mat.copy()
    n_blocks = int(np.ceil(n / block_len))
    blocks = [mat[i * block_len : min((i + 1) * block_len, n)] for i in range(n_blocks)]
    order = rng.permutation(len(blocks))
    return np.vstack([blocks[i] for i in order])[:n]


def permute_news_columns(
    onehot: pd.DataFrame,
    block_len: int = BLOCK_LEN,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    rng = rng or np.random.default_rng(0)
    out = onehot.copy()
    mask = news_column_mask(onehot.columns)
    if not mask.any():
        return out
    news = np.array(onehot.to_numpy()[:, mask], copy=True)
    permuted = block_permute_rows(news, block_len, rng)
    vals = np.array(out.to_numpy(), copy=True)
    vals[:, mask] = permuted
    return pd.DataFrame(vals, columns=onehot.columns, index=onehot.index)


def empirical_p_value(obs_lift: float, null_lifts: list[float]) -> float:
    b = len(null_lifts)
    return (1 + sum(1 for x in null_lifts if x >= obs_lift)) / (b + 1)


def permutation_test_rules(
    onehot: pd.DataFrame,
    rules: list[Rule],
    n_perm: int = N_PERM,
    block_len: int = BLOCK_LEN,
    seed: int = 42,
    min_support: float = 0.08,
) -> pd.DataFrame:
    """For each rule, permute news blocks and recompute lift of the same itemset if present.

    Uses co-occurrence counts on the one-hot matrix for speed (exact for support/conf/lift).
    """
    if not rules:
        return pd.DataFrame(columns=["antecedents", "consequents", "lift", "p_value", "null_mean_lift"])

    mat = onehot.to_numpy(dtype=bool)
    cols = {c: i for i, c in enumerate(onehot.columns)}
    rng = np.random.default_rng(seed)
    news_mask = news_column_mask(onehot.columns)

    def lift_of(rule: Rule, data: np.ndarray) -> float | None:
        try:
            ai = [cols[a] for a in rule.antecedents]
            ci = [cols[c] for c in rule.consequents]
        except KeyError:
            return None
        ant = data[:, ai].all(axis=1)
        cons = data[:, ci].all(axis=1)
        both = ant & cons
        n = len(data)
        supp_x = ant.mean()
        supp_y = cons.mean()
        supp_xy = both.mean()
        if supp_x <= 0 or supp_y <= 0:
            return 0.0
        conf = supp_xy / supp_x
        return conf / supp_y

    null_lifts: dict[int, list[float]] = {i: [] for i in range(len(rules))}
    base = mat.copy()
    for _ in range(n_perm):
        perm = base.copy()
        if news_mask.any():
            news = np.array(perm[:, news_mask], copy=True)
            perm[:, news_mask] = block_permute_rows(news, block_len, rng)
        for i, rule in enumerate(rules):
            lv = lift_of(rule, perm)
            null_lifts[i].append(0.0 if lv is None else float(lv))

    rows = []
    for i, rule in enumerate(rules):
        p = empirical_p_value(rule.lift, null_lifts[i])
        rows.append(
            {
                "antecedents": ",".join(sorted(rule.antecedents)),
                "consequents": ",".join(sorted(rule.consequents)),
                "support": rule.support,
                "confidence": rule.confidence,
                "lift": rule.lift,
                "p_value": p,
                "null_mean_lift": float(np.mean(null_lifts[i])) if null_lifts[i] else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def mine_and_test(
    onehot: pd.DataFrame,
    n_perm: int = 200,
    block_len: int = BLOCK_LEN,
    seed: int = 42,
) -> pd.DataFrame:
    rules = filter_lagged_news_rules(mine_rules(onehot))
    return permutation_test_rules(onehot, rules, n_perm=n_perm, block_len=block_len, seed=seed)
