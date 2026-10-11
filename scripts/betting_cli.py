#!/usr/bin/env python3
"""Walk-forward betting backtest CLI (moneyline + spread)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.betting import BettingConfig, BettingEngine, tune_min_edge
from core.elo import EloConfig
from sports.nba.adapter import NbaAdapter, prior_season
from sports.nba.odds import load_odds_for_season, match_rate


def main() -> int:
    p = argparse.ArgumentParser(description="NBA betting backtest vs historical odds")
    p.add_argument("--season", default="2024-25", help="Holdout / evaluation season")
    p.add_argument("--tune-season", default="2023-24", help="Season used to tune min_edge")
    p.add_argument("--min-edge", type=float, default=None, help="Override tuned edge")
    p.add_argument("--markets", default="moneyline,spread")
    p.add_argument("--bankroll", type=float, default=10000.0)
    p.add_argument("--kelly", type=float, default=0.25)
    p.add_argument("--refresh-odds", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    markets = tuple(m.strip() for m in args.markets.split(",") if m.strip())  # type: ignore
    adapter = NbaAdapter()
    engine = BettingEngine(elo_config=EloConfig())

    tune_season = args.tune_season
    holdout = args.season

    print(f"Loading schedules…", file=sys.stderr)
    tune_sched = adapter.fetch_results(tune_season)
    holdout_sched = adapter.fetch_results(holdout)
    prior_for_tune = adapter.fetch_results(prior_season(tune_season))
    prior_for_holdout = adapter.fetch_results(prior_season(holdout))

    print(f"Loading odds…", file=sys.stderr)
    tune_odds = load_odds_for_season(tune_season, force_refresh=args.refresh_odds)
    holdout_odds = load_odds_for_season(holdout, force_refresh=args.refresh_odds)

    tune_keys = {(g.date, g.home_team_id, g.away_team_id) for g in tune_sched.completed_games()}
    hold_keys = {(g.date, g.home_team_id, g.away_team_id) for g in holdout_sched.completed_games()}
    print(
        f"Odds match rate tune={match_rate(tune_odds, tune_keys):.1%} "
        f"holdout={match_rate(holdout_odds, hold_keys):.1%}",
        file=sys.stderr,
    )

    base = BettingConfig(
        min_edge=0.03,
        markets=markets,  # type: ignore[arg-type]
        kelly_fraction=args.kelly,
        starting_bankroll=args.bankroll,
    )

    if args.min_edge is None:
        best_edge, grid, fitted = tune_min_edge(
            engine, prior_for_tune, tune_sched, tune_odds, base=base
        )
        print("\n=== Edge grid on tune season", tune_season, "===")
        print(f"calibration a={fitted.calib_a:.3f} b={fitted.calib_b:.3f} sigma={fitted.margin_sigma:.2f}")
        for row in grid:
            print(
                f"edge={row['min_edge']:.2f}  n={int(row.get('n_decided',0)):4d}  "
                f"ROI={row.get('roi',0):+.1%}  profit={row.get('profit',0):+.0f}  "
                f"hit={row.get('hit_rate',0):.1%}  score={row.get('score')}"
            )
        print(f"Selected min_edge={best_edge:.2f}")
        cfg = BettingConfig(
            min_edge=best_edge,
            markets=fitted.markets,
            kelly_fraction=fitted.kelly_fraction,
            starting_bankroll=args.bankroll,
            flat_unit_frac=fitted.flat_unit_frac,
            calib_a=fitted.calib_a,
            calib_b=fitted.calib_b,
            margin_sigma=fitted.margin_sigma,
            elo_per_point=fitted.elo_per_point,
        )
    else:
        best_edge = args.min_edge
        grid = []
        a, b, sigma = engine.fit_calibration(
            prior_for_tune, tune_sched, tune_odds, markets=markets  # type: ignore[arg-type]
        )
        cfg = BettingConfig(
            min_edge=best_edge,
            markets=markets,  # type: ignore[arg-type]
            kelly_fraction=args.kelly,
            starting_bankroll=args.bankroll,
            flat_unit_frac=0.01,
            calib_a=a,
            calib_b=b,
            margin_sigma=sigma,
        )

    result = engine.run(
        prior_for_holdout, holdout_sched, holdout_odds, config=cfg, calibrate_sigma=False
    )
    print(f"\n=== Holdout {holdout} @ min_edge={best_edge:.2f} ===")
    print(json.dumps(result.metrics, indent=2))
    print(f"Final bankroll: {result.final_bankroll:,.2f} (start {cfg.starting_bankroll:,.2f})")
    print("Sample bets:")
    for b in result.bets[:8]:
        print(
            f"  {b.date} {b.market:9} {b.side:4} edge={b.edge:+.3f} "
            f"stake={b.stake:.0f} won={b.won} profit={b.profit:+.1f}"
        )

    if args.json:
        payload = result.to_dict()
        payload["tune_grid"] = grid
        payload["selected_min_edge"] = best_edge
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
