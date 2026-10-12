"""Unit tests for odds math and betting settlement."""

from datetime import date

from core.betting import BettingConfig, BettingEngine, OddsLine, _settle_spread
from core.elo import EloConfig, EloEngine
from core.ensemble import EnsembleModel
from core.markets import EloMarketPricer, MarginModel, MarketPricer
from core.models import Game, SeasonSchedule, Team
from core.odds import fair_probs_from_american, kelly_fraction


def test_fair_probs_sum_to_one():
    h, a = fair_probs_from_american(-150, 130)
    assert abs(h + a - 1.0) < 1e-9
    assert h > a


def test_kelly_positive_on_edge():
    f = kelly_fraction(0.55, 2.2, fraction=1.0)
    assert f > 0
    assert kelly_fraction(0.40, 2.2, fraction=1.0) == 0.0


def test_spread_cover_favorite():
    assert _settle_spread("home", -5.5, 110, 100) is True
    assert _settle_spread("home", -5.5, 105, 100) is False
    assert _settle_spread("away", 5.5, 105, 100) is True


def test_elo_market_pricer_ml_edge_signs():
    eng = EloEngine(EloConfig(home_advantage=0))
    eng.ratings = {"H": 1700.0, "A": 1300.0}
    pricer = EloMarketPricer(eng, MarginModel())
    opps = pricer.moneyline_opportunities("H", "A", -120, 100)
    home = next(o for o in opps if o.side == "home")
    assert home.model_prob > 0.7
    assert home.edge > 0


def test_ensemble_favorites_home():
    model = EnsembleModel()
    model.slow.ratings = {"H": 1700.0, "A": 1300.0}
    model.fast.ratings = {"H": 1700.0, "A": 1300.0}
    model.srs.ratings = {"H": 5.0, "A": -5.0}
    p = model.predict_proba("H", "A")
    assert p > 0.6


def test_betting_engine_runs_toy():
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
        games=[
            Game("s1", date(2024, 11, 1), "H", "A", 108, 100, "2024-25"),
            Game("s2", date(2024, 11, 2), "A", "H", 101, 99, "2024-25"),
            Game("s3", date(2024, 11, 4), "H", "A", 112, 105, "2024-25"),
        ],
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
        ),
        OddsLine(
            game_key="o2",
            date=date(2024, 11, 2),
            home_team_id="A",
            away_team_id="H",
            home_ml=-110,
            away_ml=-110,
            home_spread=-1.5,
            away_spread=1.5,
            home_spread_odds=-110,
            away_spread_odds=-110,
        ),
        OddsLine(
            game_key="o3",
            date=date(2024, 11, 4),
            home_team_id="H",
            away_team_id="A",
            home_ml=-150,
            away_ml=130,
            home_spread=-3.5,
            away_spread=3.5,
            home_spread_odds=-110,
            away_spread_odds=-110,
        ),
    ]
    result = BettingEngine().run(
        prior,
        season,
        odds,
        config=BettingConfig(min_edge=0.0, starting_bankroll=1000.0, flat_unit_frac=0.01),
    )
    assert result.metrics["n_bets"] >= 1
    assert result.final_bankroll > 0
