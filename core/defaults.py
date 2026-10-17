"""Locked-in defaults from walk-forward hillclimb (2023-24 tune / 2024-25 holdout)."""

from __future__ import annotations

from core.betting import BettingConfig

# Selected for volume + least-bad joint performance among ensemble variants.
# Spread-only +1.9% holdout configs were rejected: tune ROI ~-36% (overfit).
# This config was ~flat/slightly positive on holdout with hundreds of bets.
# Totals hillclimb was clearly worse (~-10% holdout) — keep totals opt-in only.
# Pace hillclimb: alpha=0.15, injuries off → holdout ≈ +0.6% (vs +0.4% baseline).
# Injury MPG adjustments were worse on tune and holdout — keep opt-in only.
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
    # Totals knobs (used when markets includes "total")
    totals_ema_alpha=0.12,
    totals_sigma=11.0,
    totals_market_shrink=0.25,
    totals_home_boost=1.5,
    totals_b2b_penalty=2.0,
    # Pace / injuries
    use_pace=True,
    pace_ema_alpha=0.15,
    use_injuries=False,
    injury_mpg_coeff=4.0,
    injury_total_coeff=2.5,
    injury_questionable_weight=0.25,
    home_advantage_points=2.3,
)

# Public-fade spread strategy — the only rule that was positive in every
# walk-forward holdout season tested (see scripts/walkforward_research.py).
# When one side draws 65-72% of public wagers (Yahoo betting splits), bet the
# other side at the posted spread, flat 1% units. Extremely lopsided splits
# (>=72%) are excluded: they tend to reflect genuine mismatches, and fading
# them was not profitable. Fading *all* >=65% splits was only ~+1% ROI and
# negative in 2024-25; the band was positive in 2021-22 through 2024-25.
# Mixing these bets with team-model bets diluted the return (the model-only
# strategy was negative on the 2022-23 and 2023-24 holdouts), so this preset
# runs fade-only. Treat the edge as modest and regime-sensitive, not a lock.
PUBLIC_FADE_CONFIG = BettingConfig(
    markets=("spread",),
    # Fade candidates bypass the model-edge filter; min_edge is set high so
    # only public-fade candidates (admitted separately) are bet.
    min_edge=0.99,
    flat_unit_frac=0.01,
    kelly_fraction=0.25,
    max_bet_frac=0.05,
    starting_bankroll=10_000.0,
    one_bet_per_game=True,
    use_rest=True,
    use_pace=False,
    use_injuries=False,
    use_public_fade=True,
    public_fade_threshold=65.0,
    public_fade_max_threshold=72.0,
    public_fade_only=True,
    home_advantage_points=2.3,
)
