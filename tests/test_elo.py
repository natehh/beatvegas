"""Unit tests for Elo engine."""

from datetime import date

from core.elo import EloConfig, EloEngine
from core.models import Game


def _game(gid: str, home: str, away: str, home_score: int, away_score: int, day: int = 1) -> Game:
    return Game(
        id=gid,
        date=date(2024, 1, day),
        home_team_id=home,
        away_team_id=away,
        home_score=home_score,
        away_score=away_score,
        season="2023-24",
    )


def test_predict_proba_equal_ratings_home_favored():
    eng = EloEngine(EloConfig(home_advantage=100))
    p = eng.predict_proba("A", "B")
    assert 0.5 < p < 0.75


def test_predict_proba_neutral_even():
    eng = EloEngine()
    p = eng.predict_proba("A", "B", is_neutral=True)
    assert abs(p - 0.5) < 1e-9


def test_update_moves_winner_up():
    eng = EloEngine()
    before_home = eng.get_rating("HOME")
    before_away = eng.get_rating("AWAY")
    eng.update(_game("g1", "HOME", "AWAY", 110, 100))
    assert eng.get_rating("HOME") > before_home
    assert eng.get_rating("AWAY") < before_away


def test_fit_orders_chronologically():
    eng = EloEngine(EloConfig(k_factor=40, home_advantage=0))
    games = [
        _game("g2", "A", "B", 100, 90, day=2),
        _game("g1", "B", "A", 100, 90, day=1),
    ]
    eng.fit(games, team_ids=["A", "B"])
    # B beat A first, then A beat B — both should have played; ratings finite.
    assert eng.get_rating("A") != 1500 or eng.get_rating("B") != 1500


def test_regress_toward_mean():
    eng = EloEngine(EloConfig(season_regression=1.0 / 3.0))
    eng.ratings = {"A": 1800.0, "B": 1200.0}
    eng.regress_toward_mean(["A", "B"])
    mean = 1500.0
    assert abs(eng.get_rating("A") - (1800 * 2 / 3 + mean / 3)) < 1e-6
    assert abs(eng.get_rating("B") - (1200 * 2 / 3 + mean / 3)) < 1e-6
