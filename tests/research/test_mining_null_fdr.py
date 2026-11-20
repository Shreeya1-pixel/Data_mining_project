from __future__ import annotations

import numpy as np
import pandas as pd

from research.dm_pipeline.baskets import basket_matrix, has_news_antecedent, is_lagged_predictive
from research.dm_pipeline.fdr import apply_fdr, benjamini_hochberg, k_min
from research.dm_pipeline.mining import Rule, filter_lagged_news_rules, mine_rules
from research.dm_pipeline.null_test import block_permute_rows, empirical_p_value, news_column_mask, permute_news_columns


def _planted_baskets(n: int = 120, seed: int = 1) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    baskets = []
    for i in range(n):
        items = []
        # planted: CONFLICT_EVT_L1 → Bank_DOWN with high probability
        conflict = rng.random() < 0.35
        if conflict:
            items.append("CONFLICT_EVT_L1")
            if rng.random() < 0.8:
                items.append("Bank_DOWN")
            elif rng.random() < 0.5:
                items.append("Bank_UP")
        else:
            if rng.random() < 0.3:
                items.append("Bank_UP")
            if rng.random() < 0.3:
                items.append("Bank_DOWN")
        if rng.random() < 0.2:
            items.append("GTI_HIGH_L1")
        if rng.random() < 0.4:
            items.append("IT_UP_L1")
        if rng.random() < 0.3:
            items.append("Energy_DOWN")
        # same-day noise events (should not appear as lagged predictors alone)
        if rng.random() < 0.2:
            items.append("ENERGY_EVT")
        baskets.append(sorted(set(items)))
    return baskets


def test_lagged_predictive_predicate():
    assert is_lagged_predictive(frozenset({"CONFLICT_EVT_L1"}), frozenset({"Bank_DOWN"}))
    assert not is_lagged_predictive(frozenset({"CONFLICT_EVT"}), frozenset({"Bank_DOWN"}))
    assert not is_lagged_predictive(frozenset({"Bank_UP_L1"}), frozenset({"ENERGY_EVT"}))
    assert has_news_antecedent(frozenset({"CONFLICT_EVT_L1", "IT_UP_L1"}))


def test_mine_planted_rule():
    baskets = _planted_baskets()
    oh = basket_matrix(baskets)
    rules = filter_lagged_news_rules(mine_rules(oh, min_support=0.05, min_confidence=0.5, min_lift=1.05))
    ants = [",".join(sorted(r.antecedents)) for r in rules]
    assert any("CONFLICT_EVT_L1" in a for a in ants)


def test_block_permute_preserves_shape():
    mat = np.arange(50).reshape(10, 5)
    rng = np.random.default_rng(0)
    out = block_permute_rows(mat, block_len=3, rng=rng)
    assert out.shape == mat.shape
    assert sorted(out.ravel().tolist()) == sorted(mat.ravel().tolist())


def test_permute_news_keeps_price_columns():
    oh = pd.DataFrame(
        {
            "Bank_DOWN": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "CONFLICT_EVT_L1": [1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
            "GTI_HIGH_L1": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    perm = permute_news_columns(oh, block_len=2, rng=np.random.default_rng(1))
    assert (perm["Bank_DOWN"].to_numpy() == oh["Bank_DOWN"].to_numpy()).all()
    assert news_column_mask(oh.columns).tolist() == [False, True, True]


def test_empirical_p_value_bounds():
    p = empirical_p_value(2.0, [0.5, 1.0, 1.5, 2.5])
    assert 0 < p <= 1


def test_bh_rejects_strong_signals():
    p = np.array([0.001, 0.002, 0.5, 0.8])
    rej = benjamini_hochberg(p, q=0.1)
    assert rej[0] and rej[1]
    assert not rej[3]
    assert k_min(2000, 0.1, 10000) == 2000 / (0.1 * 10001)


def test_apply_fdr_frame():
    df = pd.DataFrame(
        {
            "antecedents": ["A", "B"],
            "consequents": ["Y", "Y"],
            "p_value": [0.001, 0.9],
            "lift": [2.0, 1.1],
        }
    )
    out = apply_fdr(df, q=0.1, b=1000)
    assert "reject_bh" in out.columns
    assert bool(out.loc[0, "reject_bh"])
