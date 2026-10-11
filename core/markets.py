"""Convert team-strength models into moneyline / spread probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
from scipy.stats import norm

from core.elo import EloEngine
from core.ensemble import EnsembleModel
from core.odds import (
    EdgeOpportunity,
    american_to_decimal,
    fair_probs_from_american,
)
from core.rest import RestContext


class StrengthModel(Protocol):
    def predict_proba(
        self,
        home_team_id: str,
        away_team_id: str,
        *,
        is_neutral: bool = False,
        rest: Optional[RestContext] = None,
    ) -> float: ...

    def expected_margin(
        self,
        home_team_id: str,
        away_team_id: str,
        *,
        is_neutral: bool = False,
        rest: Optional[RestContext] = None,
    ) -> float: ...

    def cover_prob(
        self,
        home_team_id: str,
        away_team_id: str,
        home_spread_line: float,
        *,
        is_neutral: bool = False,
        rest: Optional[RestContext] = None,
    ) -> float: ...

    def shrink_to_market(self, model_p: float, market_p: float) -> float: ...


@dataclass
class MarginModel:
    """Legacy Elo-diff margin helper (kept for tests / simple Elo path)."""

    elo_per_point: float = 28.0
    margin_sigma: float = 12.0

    def expected_margin(self, elo_diff: float) -> float:
        return float(elo_diff) / self.elo_per_point

    def cover_prob(self, elo_diff: float, home_spread_line: float) -> float:
        mu = self.expected_margin(elo_diff)
        threshold = -float(home_spread_line)
        return float(1.0 - norm.cdf(threshold, loc=mu, scale=self.margin_sigma))


@dataclass
class MarketPricer:
    """Price markets from an EnsembleModel (or compatible StrengthModel)."""

    model: EnsembleModel

    def moneyline_opportunities(
        self,
        home_team_id: str,
        away_team_id: str,
        home_ml: float,
        away_ml: float,
        *,
        is_neutral: bool = False,
        rest: Optional[RestContext] = None,
    ) -> list[EdgeOpportunity]:
        raw_home = self.model.predict_proba(
            home_team_id, away_team_id, is_neutral=is_neutral, rest=rest
        )
        fair_home, fair_away = fair_probs_from_american(home_ml, away_ml)
        model_home = self.model.shrink_to_market(raw_home, fair_home)
        model_away = 1.0 - model_home
        return [
            EdgeOpportunity(
                market="moneyline",
                side="home",
                model_prob=model_home,
                market_prob=fair_home,
                edge=model_home - fair_home,
                american_odds=float(home_ml),
                decimal_odds=american_to_decimal(home_ml),
            ),
            EdgeOpportunity(
                market="moneyline",
                side="away",
                model_prob=model_away,
                market_prob=fair_away,
                edge=model_away - fair_away,
                american_odds=float(away_ml),
                decimal_odds=american_to_decimal(away_ml),
            ),
        ]

    def spread_opportunities(
        self,
        home_team_id: str,
        away_team_id: str,
        home_spread: float,
        away_spread: float,
        home_spread_odds: float,
        away_spread_odds: float,
        *,
        is_neutral: bool = False,
        rest: Optional[RestContext] = None,
    ) -> list[EdgeOpportunity]:
        raw_home = self.model.cover_prob(
            home_team_id,
            away_team_id,
            home_spread,
            is_neutral=is_neutral,
            rest=rest,
        )
        fair_home, fair_away = fair_probs_from_american(home_spread_odds, away_spread_odds)
        model_home = self.model.shrink_to_market(raw_home, fair_home)
        model_away = 1.0 - model_home
        return [
            EdgeOpportunity(
                market="spread",
                side="home",
                model_prob=model_home,
                market_prob=fair_home,
                edge=model_home - fair_home,
                american_odds=float(home_spread_odds),
                decimal_odds=american_to_decimal(home_spread_odds),
                line=float(home_spread),
            ),
            EdgeOpportunity(
                market="spread",
                side="away",
                model_prob=model_away,
                market_prob=fair_away,
                edge=model_away - fair_away,
                american_odds=float(away_spread_odds),
                decimal_odds=american_to_decimal(away_spread_odds),
                line=float(away_spread),
            ),
        ]


def calibrate_margin_sigma(
    elo_diffs: np.ndarray,
    actual_margins: np.ndarray,
    *,
    elo_per_point: float = 28.0,
) -> float:
    expected = elo_diffs / elo_per_point
    resid = actual_margins - expected
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 2 else 12.0
    return float(np.clip(sigma, 8.0, 18.0))


# Back-compat helper used by older tests
@dataclass
class EloMarketPricer:
    engine: EloEngine
    margin: MarginModel

    def elo_diff(self, home_team_id: str, away_team_id: str, *, is_neutral: bool = False) -> float:
        home = self.engine.get_rating(home_team_id)
        away = self.engine.get_rating(away_team_id)
        hca = 0.0 if is_neutral else self.engine.config.home_advantage
        return home + hca - away

    def moneyline_opportunities(
        self,
        home_team_id: str,
        away_team_id: str,
        home_ml: float,
        away_ml: float,
        *,
        is_neutral: bool = False,
    ) -> list[EdgeOpportunity]:
        model_home = self.engine.predict_proba(home_team_id, away_team_id, is_neutral=is_neutral)
        fair_home, fair_away = fair_probs_from_american(home_ml, away_ml)
        return [
            EdgeOpportunity(
                market="moneyline",
                side="home",
                model_prob=model_home,
                market_prob=fair_home,
                edge=model_home - fair_home,
                american_odds=float(home_ml),
                decimal_odds=american_to_decimal(home_ml),
            ),
            EdgeOpportunity(
                market="moneyline",
                side="away",
                model_prob=1.0 - model_home,
                market_prob=fair_away,
                edge=(1.0 - model_home) - fair_away,
                american_odds=float(away_ml),
                decimal_odds=american_to_decimal(away_ml),
            ),
        ]
