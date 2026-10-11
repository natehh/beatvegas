"""Unit tests for Monte Carlo simulation and backtests."""

from datetime import date

from core.backtest import BacktestRunner
from core.elo import EloConfig, EloEngine
from core.models import Game, SeasonSchedule, Team
from core.simulate import SeasonSimulator


def _toy_schedule() -> SeasonSchedule:
    teams = [
        Team(id="A", name="Alpha", abbreviation="ALP"),
        Team(id="B", name="Beta", abbreviation="BET"),
        Team(id="C", name="Gamma", abbreviation="GAM"),
    ]
    games = [
        Game("1", date(2024, 10, 1), "A", "B", 110, 100, season="2024-25"),
        Game("2", date(2024, 10, 2), "B", "C", 105, 100, season="2024-25"),
        Game("3", date(2024, 10, 3), "C", "A", 98, 100, season="2024-25"),
        Game("4", date(2024, 10, 4), "A", "C", 120, 110, season="2024-25"),
        Game("5", date(2024, 10, 5), "B", "A", 95, 100, season="2024-25"),
        Game("6", date(2024, 10, 6), "C", "B", 101, 99, season="2024-25"),
    ]
    return SeasonSchedule(sport="toy", season="2024-25", teams=teams, games=games)


def _prior_schedule() -> SeasonSchedule:
    teams = [
        Team(id="A", name="Alpha", abbreviation="ALP"),
        Team(id="B", name="Beta", abbreviation="BET"),
        Team(id="C", name="Gamma", abbreviation="GAM"),
    ]
    games = [
        Game("p1", date(2023, 11, 1), "A", "B", 115, 100, season="2023-24"),
        Game("p2", date(2023, 11, 2), "A", "C", 112, 100, season="2023-24"),
        Game("p3", date(2023, 11, 3), "B", "C", 108, 100, season="2023-24"),
        Game("p4", date(2023, 11, 4), "A", "B", 110, 105, season="2023-24"),
    ]
    return SeasonSchedule(sport="toy", season="2023-24", teams=teams, games=games)


def test_simulate_expected_wins_sum_to_games():
    sched = _toy_schedule()
    eng = EloEngine()
    eng.fit(sched.completed_games(), team_ids=[t.id for t in sched.teams])
    sim = SeasonSimulator(eng)
    result = sim.simulate(sched.completed_games(), sched.teams, n_sims=2000, seed=1)
    total_expected = sum(result.expected_wins.values())
    assert abs(total_expected - len(sched.completed_games())) < 0.5


def test_stronger_team_gets_more_wins():
    teams = [Team("A", "A"), Team("B", "B")]
    games = [
        Game("1", date(2024, 1, 1), "A", "B", season="x"),
        Game("2", date(2024, 1, 2), "B", "A", season="x"),
        Game("3", date(2024, 1, 3), "A", "B", season="x"),
        Game("4", date(2024, 1, 4), "B", "A", season="x"),
    ]
    eng = EloEngine(EloConfig(home_advantage=0))
    eng.ratings = {"A": 1700.0, "B": 1300.0}
    result = SeasonSimulator(eng).simulate(games, teams, n_sims=3000, seed=0)
    assert result.expected_wins["A"] > result.expected_wins["B"]


def test_preseason_backtest_runs():
    prior = _prior_schedule()
    season = _toy_schedule()
    result = BacktestRunner().run_preseason(prior, season, n_sims=500, seed=0)
    assert result.mode == "preseason"
    assert len(result.teams) == 3
    assert "mae_wins" in result.metrics
    assert result.metrics["mae_wins"] >= 0


def test_rolling_backtest_game_metrics():
    prior = _prior_schedule()
    season = _toy_schedule()
    result = BacktestRunner().run_rolling(prior, season, n_sims=500, seed=0)
    assert result.mode == "rolling"
    assert result.metrics["n_games"] == 6
    assert 0.0 <= result.metrics["brier"] <= 1.0
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
