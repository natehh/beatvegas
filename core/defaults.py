"""Locked-in defaults from walk-forward hillclimb (2023-24 tune / 2024-25 holdout)."""

from __future__ import annotations

from core.betting import BettingConfig

# Selected for volume + least-bad joint performance among ensemble variants.
# Spread-only +1.9% holdout configs were rejected: tune ROI ~-36% (overfit).
# This config was ~flat/slightly positive on holdout with hundreds of bets.
DEFAULT_BETTING_CONFIG = BettingConfig(
    markets=("moneyline", "spread"),
    min_edge=0.03,
    market_shrink=0.0,
    elo_blend=0.70,
    srs_blend=0.30,
    use_rest=True,
    fast_k=36.0,
    slow_k=12.0,
    slow_weight=0.55,
    fast_weight=0.45,
    b2b_penalty=2.2,
    flat_unit_frac=0.01,
    kelly_fraction=0.25,
    max_bet_frac=0.05,
    starting_bankroll=10_000.0,
    one_bet_per_game=True,
)
