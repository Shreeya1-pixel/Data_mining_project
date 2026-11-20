"""Benjamini–Hochberg and Benjamini–Yekutieli FDR control."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.dm_pipeline.constants import FDR_Q


def benjamini_hochberg(p_values: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """Return boolean reject mask (same order as p_values)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, m + 1) / m) * q
    below = ranked <= thresh
    reject = np.zeros(m, dtype=bool)
    if below.any():
        kstar = np.max(np.where(below)[0])
        reject[order[: kstar + 1]] = True
    return reject


def benjamini_yekutieli(p_values: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool)
    c_m = np.sum(1.0 / np.arange(1, m + 1))
    return benjamini_hochberg(p, q=q / c_m)


def k_min(m: int, q: float, b: int) -> float:
    """Minimum number of floor p-values needed for BH to reject any."""
    return m / (q * (b + 1))


def apply_fdr(df: pd.DataFrame, q: float = FDR_Q, b: int = 1000) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["reject_bh"] = []
        out["reject_by"] = []
        return out
    p = out["p_value"].to_numpy()
    out["reject_bh"] = benjamini_hochberg(p, q=q)
    out["reject_by"] = benjamini_yekutieli(p, q=q)
    out["k_min"] = k_min(len(out), q, b)
    return out
