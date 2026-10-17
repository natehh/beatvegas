"""Tests for public-betting splits, fade band, rest, and home-court fixes."""

from datetime import date

from core.betting import BettingConfig, BettingEngine, OddsLine
from core.ensemble import EnsembleModel
from core.models import Game, SeasonSchedule, Team
from core.rest import RestTracker
from sports.nba.odds import _line_from_dict, _line_to_dict


def test_rest_one_day_off_is_not_back_to_back():
    tr = RestTracker()
    g1 = Game("1", date(2024, 1, 1), "H", "A", 100, 90, "x")
    tr.observe(g1)
    # Play again two calendar days later = 1 full day of rest, not a B2B.
    g2 = Game("2", date(2024, 1, 3), "H", "B", 100, 95, "x")
    ctx = tr.context(g2)
    assert ctx.home_days_rest == 1
    assert not ctx.home_b2b
    # Next-day game is a true back-to-back.
    g3 = Game("3", date(2024, 1, 4), "H", "B", 100, 95, "x")
    tr.observe(g2)
    assert tr.context(g3).home_b2b


def test_ensemble_home_court_applied_once():
    model = EnsembleModel()
    model.slow.ratings = {"H": 1500.0, "A": 1500.0}
    model.fast.ratings = {"H": 1500.0, "A": 1500.0}
    model.srs.ratings = {"H": 0.0, "A": 0.0}
    home_margin = model.expected_margin("H", "A")
    neutral_margin = model.expected_margin("H", "A", is_neutral=True)
    assert abs(home_margin - neutral_margin - model.config.home_advantage_points) < 1e-9


def test_odds_line_public_splits_roundtrip():
    line = OddsLine(
        game_key="g1",
        date=date(2024, 11, 1),
        home_team_id="H",
        away_team_id="A",
        home_ml=-110,
        away_ml=-110,
        home_spread=-2.5,
        away_spread=2.5,
        home_spread_odds=-110,
        away_spread_odds=-110,
        spread_home_wager_pct=68.0,
        spread_away_wager_pct=32.0,
    )
    restored = _line_from_dict(_line_to_dict(line))
    assert restored.spread_home_wager_pct == 68.0
    assert restored.spread_away_wager_pct == 32.0


def _toy_setup(home_wager_pct: float, away_wager_pct: float):
    teams = [Team("H", "Home"), Team("A", "Away")]
    prior = SeasonSchedule(
        sport="toy",
        season="2023-24",
        teams=teams,
        games=[
            Game("p1", date(2023, 11, 1), "H", "A", 110, 100, "2023-24"),
            Game("p2", date(2023, 11, 2), "A", "H", 95, 105, "2023-24"),
            Game("p3", date(2023, 11, 3), "H", "A", 120, 100, "2023-24"),
            Game("p4", date(2023, 11, 4), "A", "H", 90, 108, "2023-24"),
            Game("p5", date(2023, 11, 5), "H", "A", 115, 100, "2023-24"),
        ],
    )
    season = SeasonSchedule(
        sport="toy",
        season="2024-25",
        teams=teams,
        games=[Game("s1", date(2024, 11, 1), "H", "A", 108, 100, "2024-25")],
    )
    odds = [
        OddsLine(
            game_key="o1",
            date=date(2024, 11, 1),
            home_team_id="H",
            away_team_id="A",
            home_ml=-110,
            away_ml=-110,
            home_spread=-1.5,
            away_spread=1.5,
            home_spread_odds=-110,
            away_spread_odds=-110,
            spread_home_wager_pct=home_wager_pct,
            spread_away_wager_pct=away_wager_pct,
        )
    ]
    return prior, season, odds


def _fade_config(**overrides):
    cfg = BettingConfig(
        markets=("spread",),
        min_edge=0.99,
        flat_unit_frac=0.01,
        starting_bankroll=1000.0,
        use_pace=False,
        use_injuries=False,
        use_public_fade=True,
        public_fade_threshold=65.0,
        public_fade_max_threshold=72.0,
        public_fade_only=True,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_public_fade_bets_against_lopsided_home_public():
    prior, season, odds = _toy_setup(68.0, 32.0)
    result = BettingEngine().run(prior, season, odds, config=_fade_config())
    assert len(result.bets) == 1
    assert result.bets[0].market == "spread"
    assert result.bets[0].side == "away"


def test_public_fade_ignores_extreme_public_split():
    # 90% on the home side is outside the [65, 72) fade band.
    prior, season, odds = _toy_setup(90.0, 10.0)
    result = BettingEngine().run(prior, season, odds, config=_fade_config())
    assert result.bets == []


def test_public_fade_ignores_balanced_split():
    prior, season, odds = _toy_setup(55.0, 45.0)
    result = BettingEngine().run(prior, season, odds, config=_fade_config())
    assert result.bets == []
