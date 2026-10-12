"""Totals model and settlement tests."""

from datetime import date

from core.betting import BettingConfig, BettingEngine, OddsLine, _settle_total
from core.models import Game, SeasonSchedule, Team
from core.totals import TotalsModel


def test_settle_total():
    assert _settle_total("over", 220.5, 110, 112) is True
    assert _settle_total("under", 220.5, 110, 112) is False
    assert _settle_total("over", 222.0, 110, 112) is None  # push


def test_totals_model_tracks_scoring():
    model = TotalsModel()
    games = [
        Game("1", date(2024, 1, 1), "H", "A", 120, 110, "x"),
        Game("2", date(2024, 1, 3), "A", "H", 100, 115, "x"),
        Game("3", date(2024, 1, 5), "H", "A", 118, 108, "x"),
    ]
    model.fit(games, ["H", "A"])
    assert model.off["H"] > model.off["A"]
    mu = model.expected_total("H", "A")
    assert 200 < mu < 250
    p_over = model.over_prob("H", "A", mu - 5)
    assert p_over > 0.5


def test_betting_engine_totals_toy():
    teams = [Team("H", "Home"), Team("A", "Away")]
    prior_games = [
        Game(f"p{i}", date(2023, 11, i), "H", "A", 114 + i, 108, "2023-24")
        for i in range(1, 8)
    ] + [
        Game(f"q{i}", date(2023, 11, 10 + i), "A", "H", 105, 112, "2023-24")
        for i in range(1, 6)
    ]
    prior = SeasonSchedule("toy", "2023-24", teams, prior_games)
    season_games = [
        Game("s1", date(2024, 11, 1), "H", "A", 120, 115, "2024-25"),
        Game("s2", date(2024, 11, 3), "A", "H", 108, 110, "2024-25"),
        Game("s3", date(2024, 11, 5), "H", "A", 100, 98, "2024-25"),
    ]
    season = SeasonSchedule("toy", "2024-25", teams, season_games)
    odds = [
        OddsLine(
            "o1", date(2024, 11, 1), "H", "A", -110, -110, -3.5, 3.5, -110, -110,
            total_line=220.5, over_odds=-110, under_odds=-110,
        ),
        OddsLine(
            "o2", date(2024, 11, 3), "A", "H", -110, -110, -1.5, 1.5, -110, -110,
            total_line=218.5, over_odds=-110, under_odds=-110,
        ),
        OddsLine(
            "o3", date(2024, 11, 5), "H", "A", -120, 100, -2.5, 2.5, -110, -110,
            total_line=225.5, over_odds=-105, under_odds=-115,
        ),
    ]
    result = BettingEngine().run(
        prior,
        season,
        odds,
        config=BettingConfig(markets=("total",), min_edge=0.0, flat_unit_frac=0.01, starting_bankroll=1000),
    )
    assert result.metrics["by_total_n"] >= 1
