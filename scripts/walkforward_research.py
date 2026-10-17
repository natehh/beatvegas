#!/usr/bin/env python3
"""Multi-season walk-forward comparison of betting strategies.

Strategies evaluated on each holdout season (flat 1% units, $10k bankroll):
  1. team model (moneyline+spread) with min_edge tuned on the prior season
  2. public fade (spread only): bet against a side drawing 65-72% of public
     wagers (Yahoo betting splits), no team-model input

The team model is tuned strictly on prior seasons. The public-fade band is a
fixed rule; it is reported on every season for which Yahoo splits exist,
including 2021-22 (which cannot be a model holdout here because tuning it
would require 2020-21 odds).

Usage:
    python scripts/walkforward_research.py [--no-pace]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import replace

from core.betting import BettingConfig, BettingEngine, tune_min_edge
from core.defaults import DEFAULT_BETTING_CONFIG, PUBLIC_FADE_CONFIG
from sports.nba.adapter import NbaAdapter, prior_season
from sports.nba.odds import load_odds_for_season

HOLDOUTS = ["2022-23", "2023-24", "2024-25"]
FADE_SEASONS = ["2021-22"] + HOLDOUTS


def _model_base(**overrides) -> BettingConfig:
    cfg = BettingConfig(
        min_edge=DEFAULT_BETTING_CONFIG.min_edge,
        markets=("moneyline", "spread"),
        market_shrink=DEFAULT_BETTING_CONFIG.market_shrink,
        elo_blend=DEFAULT_BETTING_CONFIG.elo_blend,
        srs_blend=DEFAULT_BETTING_CONFIG.srs_blend,
        use_rest=DEFAULT_BETTING_CONFIG.use_rest,
        fast_k=DEFAULT_BETTING_CONFIG.fast_k,
        slow_k=DEFAULT_BETTING_CONFIG.slow_k,
        b2b_penalty=DEFAULT_BETTING_CONFIG.b2b_penalty,
        flat_unit_frac=0.01,
        use_pace=False,
        use_injuries=False,
        home_advantage_points=DEFAULT_BETTING_CONFIG.home_advantage_points,
    )
    return replace(cfg, **overrides) if overrides else cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    adapter = NbaAdapter()
    engine = BettingEngine()

    seasons_needed = sorted(
        {prior_season(prior_season(s)) for s in HOLDOUTS}
        | set(FADE_SEASONS)
        | {prior_season(s) for s in FADE_SEASONS}
    )
    print(f"Loading schedules for {seasons_needed}…", file=sys.stderr)
    scheds = {s: adapter.fetch_results(s) for s in seasons_needed}
    odds = {s: load_odds_for_season(s) for s in FADE_SEASONS}

    print("\n=== Public fade (spread, 65-72% band, flat 1%) ===")
    tot_profit = tot_stake = 0.0
    for season in FADE_SEASONS:
        res = engine.run(
            scheds[prior_season(season)],
            scheds[season],
            odds[season],
            config=PUBLIC_FADE_CONFIG,
            calibrate_sigma=False,
        )
        m = res.metrics
        tot_profit += m["profit"]
        tot_stake += int(m["n_decided"]) * 100.0
        print(
            f"{season}: n={int(m['n_decided']):4d} hit={m['hit_rate']:.3f} "
            f"ROI={m['roi']:+.2%} profit={m['profit']:+.0f}"
        )
    print(f"ALL: ROI={tot_profit / tot_stake:+.2%} (flat $100 bets)")

    print("\n=== Team model (ML+spread, min_edge tuned on prior season) ===")
    tot_profit = tot_stake = 0.0
    for holdout in HOLDOUTS:
        tune = prior_season(holdout)
        best_edge, _grid, fitted = tune_min_edge(
            engine,
            scheds[prior_season(tune)],
            scheds[tune],
            odds[tune],
            base=_model_base(),
        )
        res = engine.run(
            scheds[tune], scheds[holdout], odds[holdout], config=fitted, calibrate_sigma=False
        )
        m = res.metrics
        tot_profit += m["profit"]
        tot_stake += int(m["n_decided"]) * 100.0
        print(
            f"{holdout} (tuned on {tune}, edge={best_edge:.2f}): "
            f"n={int(m['n_decided']):4d} hit={m['hit_rate']:.3f} "
            f"ROI={m['roi']:+.2%} profit={m['profit']:+.0f} brier={m['brier']}"
        )
    print(f"ALL: ROI={tot_profit / tot_stake:+.2%} (flat $100 bets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
