"""End-to-end GeoNews-IN → GeoRule experiment runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.dm_pipeline.baskets import basket_matrix, build_baskets
from research.dm_pipeline.classifier import compare_classifiers
from research.dm_pipeline.constants import BLOCK_LEN, GTI_LAMBDA, N_PERM, SECTORS
from research.dm_pipeline.fdr import apply_fdr
from research.dm_pipeline.gti import daily_shocks, decayed_levels, event_items, gti_items, gti_series
from research.dm_pipeline.mining import filter_lagged_news_rules, mine_rules, rules_to_frame
from research.dm_pipeline.news_filter import filter_geonews, tier_summary
from research.dm_pipeline.null_test import permutation_test_rules
from research.dm_pipeline.prices import load_closes, market_regime, price_states, returns, train_sigma
from research.dm_pipeline.relevance import apply_relevance
from research.dm_pipeline.session import session_of


def _parse_article_times(df: pd.DataFrame) -> pd.Series:
    # GKG DATE column often YYYYMMDDHHMMSS or ISO-like
    raw = df["date"].astype(str)
    ts = pd.to_datetime(raw.str[:12], format="%Y%m%d%H%M", errors="coerce")
    # convert assumed UTC batch time → IST (+5:30) as in pilot
    return ts + pd.Timedelta(hours=5, minutes=30)


def load_pilot_news(paths: list[str | Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def attach_sessions(news: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    out = news.copy()
    out["pub_ist"] = _parse_article_times(out)
    out["session"] = out["pub_ist"].map(lambda t: session_of(t, sessions))
    return out


def run_experiment(
    closes_path: str | Path,
    news_paths: list[str | Path],
    artifacts_dir: str | Path,
    train_end: str = "2024-03-31",
    holdout_start: str = "2025-01-01",
    n_perm: int = 200,
    sectors: tuple[str, ...] = ("Bank", "IT", "Energy", "Metal"),
    seed: int = 42,
    news_months_only: bool = False,
) -> dict:
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    px = load_closes(str(closes_path))
    sessions = px.index
    ret = returns(px)
    sigma = train_sigma(ret, train_end)
    states = price_states(ret, sigma)
    regime = market_regime(px["Nifty50"])

    raw = load_pilot_news(news_paths)
    tiers = tier_summary(raw, scale=(30 * 96) / (30 * 4))
    geo = filter_geonews(raw, tier="T3")
    geo = apply_relevance(geo)  # heuristic until labels filled
    geo = attach_sessions(geo, sessions)

    if news_months_only:
        months = {
            (pd.Timestamp(s).year, pd.Timestamp(s).month)
            for s in geo["session"].dropna()
        }
        sessions = sessions[[(t.year, t.month) in months for t in sessions]]
        if len(sessions) < 5:
            sessions = px.index  # fallback
        states = states.reindex(sessions)
        regime = regime.reindex(sessions)

    shocks = daily_shocks(geo, sessions)
    levels = decayed_levels(shocks, lam=GTI_LAMBDA)
    gti = gti_series(levels)
    evt = event_items(levels, train_end)
    gflags = gti_items(gti, train_end)

    baskets = build_baskets(sessions, states, evt, gflags)
    basket_days = list(sessions[1:])  # mirrors build_baskets skip-first

    onehot = basket_matrix(baskets)
    train_mask = np.array([d <= pd.Timestamp(train_end) for d in basket_days])
    test_mask = np.array([d >= pd.Timestamp(holdout_start) for d in basket_days])

    train_oh = onehot.loc[train_mask].reset_index(drop=True)
    test_oh = onehot.loc[test_mask].reset_index(drop=True)
    train_b = [baskets[i] for i, m in enumerate(train_mask) if m]
    test_b = [baskets[i] for i, m in enumerate(test_mask) if m]

    # Raise support until lagged-news family is ≤ 2000 (report: trim families > 5000)
    min_sup = 0.08
    all_rules = mine_rules(train_oh, min_support=min_sup)
    lagged = filter_lagged_news_rules(all_rules)
    while len(lagged) > 2000 and min_sup < 0.25:
        min_sup += 0.02
        all_rules = mine_rules(train_oh, min_support=min_sup)
        lagged = filter_lagged_news_rules(all_rules)
    # Cap permutation family by lift (keeps BH power with finite B)
    lagged = sorted(lagged, key=lambda r: (-r.lift, -r.confidence, -r.support))[:1500]
    tested = permutation_test_rules(train_oh, lagged, n_perm=n_perm, block_len=BLOCK_LEN, seed=seed)
    tested = apply_fdr(tested, b=n_perm)
    survivors = tested[tested["reject_bh"]].copy() if not tested.empty else tested
    summary_min_sup = min_sup

    # Stability: rolling-origin 4 folds inside train
    fold_reports = []
    train_days = [d for d, m in zip(basket_days, train_mask) if m]
    if len(train_days) >= 200:
        cuts = np.linspace(0.4, 0.85, 4)
        for cut in cuts:
            k = int(len(train_b) * cut)
            sub = basket_matrix(train_b[:k])
            rules_f = filter_lagged_news_rules(mine_rules(sub))
            fold_reports.append({"cut": float(cut), "n_days": k, "n_lagged_rules": len(rules_f)})

    clf_tables = {}
    for sector in sectors:
        if sector not in px.columns:
            continue
        clf_tables[sector] = compare_classifiers(train_b, test_b, lagged, sector).to_dict(orient="records")

    # Exploratory summaries
    corr = ret[[c for c in SECTORS if c in ret.columns]].corr()
    regime_counts = regime.value_counts(dropna=True).to_dict()
    weekday_evt = (
        evt.mean(axis=1)
        .groupby(evt.index.day_of_week)
        .mean()
        .rename(index={0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})
        .to_dict()
    )

    summary = {
        "n_sessions": int(len(sessions)),
        "n_articles_raw": int(len(raw)),
        "n_articles_t3_relevant": int(len(geo)),
        "n_baskets": int(len(baskets)),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_rules_all": int(len(all_rules)),
        "n_rules_lagged_news": int(len(lagged)),
        "n_survivors_bh": int(len(survivors)),
        "tiers": tiers,
        "regime_counts": {str(k): int(v) for k, v in regime_counts.items()},
        "weekday_evt_rate": weekday_evt,
        "fold_reports": fold_reports,
        "classifiers": clf_tables,
        "n_perm": n_perm,
        "min_support_used": summary_min_sup,
        "gti_lambda": GTI_LAMBDA,
        "novelty_note": (
            "News-only block permutation + event-typed session GTI + Nifty sector GeoRules; "
            "not claiming causality or trading PnL."
        ),
    }

    rules_to_frame(all_rules).to_csv(artifacts / "rules_all.csv", index=False)
    rules_to_frame(lagged).to_csv(artifacts / "rules_lagged_news.csv", index=False)
    tested.to_csv(artifacts / "rules_tested.csv", index=False)
    survivors.to_csv(artifacts / "rules_survivors_bh.csv", index=False)
    corr.to_csv(artifacts / "sector_corr.csv")
    gti.to_csv(artifacts / "gti_daily.csv", header=["GTI"])
    pd.Series(regime, name="regime").to_csv(artifacts / "regimes.csv")
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # save a compact classifier table
    clf_rows = []
    for sector, rows in clf_tables.items():
        for r in rows:
            clf_rows.append({"sector": sector, **r})
    pd.DataFrame(clf_rows).to_csv(artifacts / "classifier_comparison.csv", index=False)

    return summary
