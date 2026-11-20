"""CLI: python -m research.dm_pipeline.cli run ..."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "research" / "data" / "pilot"
DEFAULT_ART = ROOT / "research" / "artifacts"


def cmd_run(args: argparse.Namespace) -> None:
    from research.dm_pipeline.pipeline import run_experiment

    news = list(Path(args.news_glob).parent.glob(Path(args.news_glob).name)) if "*" in args.news_glob else [Path(p) for p in args.news]
    if not news:
        news = sorted((DEFAULT_DATA).glob("gkg_pilot_rows_*.parquet"))
    summary = run_experiment(
        closes_path=args.closes,
        news_paths=news,
        artifacts_dir=args.out,
        train_end=args.train_end,
        holdout_start=args.holdout_start,
        n_perm=args.n_perm,
        sectors=tuple(args.sectors.split(",")),
        news_months_only=args.news_months_only,
    )
    print(json.dumps({k: summary[k] for k in summary if k not in ("tiers", "classifiers")}, indent=2))
    print(f"artifacts → {args.out}")


def cmd_tiers(args: argparse.Namespace) -> None:
    import pandas as pd

    from research.dm_pipeline.news_filter import tier_summary

    df = pd.read_parquet(args.parquet)
    print(json.dumps(tier_summary(df, scale=args.scale), indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dm_pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run full GeoRule experiment")
    r.add_argument("--closes", default=str(DEFAULT_DATA / "closes_niftyindices.csv"))
    r.add_argument("--news", nargs="*", default=[])
    r.add_argument("--news-glob", default=str(DEFAULT_DATA / "gkg_pilot_rows_*.parquet"))
    r.add_argument("--out", default=str(DEFAULT_ART))
    r.add_argument("--train-end", default="2024-03-31")
    r.add_argument("--holdout-start", default="2025-01-01")
    r.add_argument("--n-perm", type=int, default=200)
    r.add_argument("--sectors", default="Bank,IT,Energy,Metal")
    r.add_argument("--news-months-only", action="store_true", help="Restrict to months with GKG coverage")
    r.set_defaults(func=cmd_run)

    t = sub.add_parser("tiers", help="Summarise GeoNews-IN filter tiers")
    t.add_argument("parquet")
    t.add_argument("--scale", type=float, default=8.0)
    t.set_defaults(func=cmd_tiers)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
