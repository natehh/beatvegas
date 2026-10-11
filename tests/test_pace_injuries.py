"""Unit tests for pace and injury adjustments."""

from datetime import date

from core.betting import BettingConfig, BettingEngine, OddsLine
from core.injuries import InjuryListing, InjuryModel, normalize_player_name
from core.models import Game, SeasonSchedule, Team
from core.pace import PaceModel, PaceObservation
from core.totals import TotalsModel


def test_normalize_player_name():
    assert normalize_player_name("Porzingis, Kristaps") == "kristaps porzingis"
    assert normalize_player_name("Kristaps Porziņģis") == "kristaps porzingis"
    assert normalize_player_name("Jrue Holiday") == "jrue holiday"


def test_pace_scale_moves_with_fast_teams():
    model = PaceModel()
    obs = [
        PaceObservation(date(2024, 1, 1), "H", "A", 108.0),
        PaceObservation(date(2024, 1, 3), "H", "A", 110.0),
        PaceObservation(date(2024, 1, 5), "H", "A", 109.0),
    ]
    model.fit(obs, ["H", "A", "S"])
    # Slow team S stays near initial; fast pair projects higher than H vs S
    assert model.expected_pace("H", "A") > model.expected_pace("H", "S")
    assert model.pace_scale("H", "A") > model.pace_scale("H", "S")


def test_totals_respects_pace_scale():
    totals = TotalsModel()
    games = [
        Game("1", date(2024, 1, 1), "H", "A", 120, 110, "x"),
        Game("2", date(2024, 1, 3), "A", "H", 105, 115, "x"),
    ]
    totals.fit(games, ["H", "A"])
    base = totals.expected_total("H", "A")
    scaled = totals.expected_total("H", "A", pace_scale=1.05)
    assert scaled > base


def test_injury_margin_favors_healthier_side():
    model = InjuryModel()
    model.minutes.mpg[normalize_player_name("Star, Home")] = 36.0
    model.minutes.mpg[normalize_player_name("Bench, Away")] = 12.0
    listings = [
        InjuryListing(date(2024, 1, 1), "H", "Star, Home", "Out"),
        InjuryListing(date(2024, 1, 1), "A", "Bench, Away", "Out"),
    ]
    impact = model.impact("H", "A", listings)
    assert impact.margin_adj < 0  # home missing the star
    assert impact.total_adj < 0


def test_betting_engine_with_pace_and_injuries_toy():
    teams = [Team("H", "Home"), Team("A", "Away")]
    prior_games = [
        Game(f"p{i}", date(2023, 11, i), "H", "A", 114, 108, "2023-24") for i in range(1, 10)
    ]
    prior = SeasonSchedule("toy", "2023-24", teams, prior_games)
    season_games = [
        Game("s1", date(2024, 11, 1), "H", "A", 110, 108, "2024-25"),
        Game("s2", date(2024, 11, 3), "A", "H", 112, 109, "2024-25"),
    ]
    season = SeasonSchedule("toy", "2024-25", teams, season_games)
    odds = [
        OddsLine("o1", date(2024, 11, 1), "H", "A", -120, 100, -2.5, 2.5, -110, -110),
        OddsLine("o2", date(2024, 11, 3), "A", "H", -110, -110, -1.5, 1.5, -110, -110),
    ]
    pace = [
        PaceObservation(date(2023, 11, i), "H", "A", 100.0 + i) for i in range(1, 10)
    ] + [
        PaceObservation(date(2024, 11, 1), "H", "A", 105.0),
        PaceObservation(date(2024, 11, 3), "A", "H", 104.0),
    ]
    injuries = {
        "2024-11-01": [InjuryListing(date(2024, 11, 1), "A", "Star, Away", "Out")],
    }
    prior_mpg = {"Star, Away": {"player": "Star, Away", "team_id": "A", "mpg": 34.0, "games": 70}}
    result = BettingEngine().run(
        prior,
        season,
        odds,
        config=BettingConfig(
            markets=("moneyline", "spread"),
            min_edge=0.0,
            flat_unit_frac=0.01,
            starting_bankroll=1000,
            use_pace=True,
            use_injuries=True,
        ),
        pace_observations=pace,
        prior_pace_observations=pace[:9],
        injuries_by_date=injuries,
        prior_player_mpg=prior_mpg,
    )
    assert result.metrics["n_bets"] >= 1
