#!/usr/bin/env python3
"""Headless NBA backtest CLI for tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.backtest import BacktestRunner
from core.elo import EloConfig
from sports.nba.adapter import NbaAdapter, prior_season


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NBA team W/L backtest")
    parser.add_argument("--season", default="2024-25")
    parser.add_argument("--mode", choices=["preseason", "rolling", "both"], default="both")
    parser.add_argument("--n-sims", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch from nba_api")
    parser.add_argument("--json", action="store_true", help="Print full JSON result(s)")
    args = parser.parse_args()

    adapter = NbaAdapter()
    if args.refresh:
        for s in [prior_season(args.season), args.season]:
            print(f"Refreshing {s}…", file=sys.stderr)
            print(adapter.refresh(s), file=sys.stderr)

    season_sched = adapter.fetch_results(args.season, use_cache=not args.refresh, force_refresh=args.refresh)
    prior = adapter.fetch_results(
        prior_season(args.season), use_cache=not args.refresh, force_refresh=args.refresh
    )
    runner = BacktestRunner(elo_config=EloConfig())

    modes = ["preseason", "rolling"] if args.mode == "both" else [args.mode]
    outputs = []
    for mode in modes:
        if mode == "preseason":
            result = runner.run_preseason(prior, season_sched, n_sims=args.n_sims, seed=args.seed)
        else:
            result = runner.run_rolling(prior, season_sched, n_sims=args.n_sims, seed=args.seed)
        outputs.append(result)
        print(f"\n=== {mode} @ {args.season} (n_sims={args.n_sims}) ===")
        print("metrics:", json.dumps(result.metrics, indent=2))
        print(f"{'Team':<32} {'ProjW':>7} {'ActW':>6} {'Err':>7}")
        for t in result.teams[:10]:
            print(f"{t.team:<32} {t.expected_wins:>7.1f} {t.actual_wins:>6} {t.error:>+7.1f}")
        print(f"… {len(result.teams)} teams total")

    if args.json:
        print(json.dumps([o.to_dict() for o in outputs], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
