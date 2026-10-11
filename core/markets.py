"""Convert Elo ratings into moneyline / spread probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from core.elo import EloEngine
from core.odds import (
    EdgeOpportunity,
    american_to_decimal,
    fair_probs_from_american,
)


@dataclass
class MarginModel:
    """
    Map Elo difference (home rating + HCA - away) to expected home margin.
    NBA rule of thumb: ~28 Elo ≈ 1 point. Sigma calibrated from residuals.
    """

    elo_per_point: float = 28.0
    margin_sigma: float = 12.0

    def expected_margin(self, elo_diff: float) -> float:
        return float(elo_diff) / self.elo_per_point

    def cover_prob(self, elo_diff: float, home_spread_line: float) -> float:
        """
        P(home covers) when home_spread_line is the home side line (e.g. -5.5).
        Home covers if home_margin + home_spread_line > 0
        (push when exactly 0 — treated as no-bet / excluded later).
        """
        mu = self.expected_margin(elo_diff)
        # Home covers when margin > -home_spread_line
        threshold = -float(home_spread_line)
        return float(1.0 - norm.cdf(threshold, loc=mu, scale=self.margin_sigma))


@dataclass
class MarketPricer:
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
        out = [
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
        return out

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
    ) -> list[EdgeOpportunity]:
        diff = self.elo_diff(home_team_id, away_team_id, is_neutral=is_neutral)
        model_home_cover = self.margin.cover_prob(diff, home_spread)
        # Away cover ≈ 1 - home cover for non-push continuous case; with half-lines pushes rare.
        model_away_cover = 1.0 - model_home_cover
        fair_home, fair_away = fair_probs_from_american(home_spread_odds, away_spread_odds)
        return [
            EdgeOpportunity(
                market="spread",
                side="home",
                model_prob=model_home_cover,
                market_prob=fair_home,
                edge=model_home_cover - fair_home,
                american_odds=float(home_spread_odds),
                decimal_odds=american_to_decimal(home_spread_odds),
                line=float(home_spread),
            ),
            EdgeOpportunity(
                market="spread",
                side="away",
                model_prob=model_away_cover,
                market_prob=fair_away,
                edge=model_away_cover - fair_away,
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
    """Estimate residual std of (actual margin - expected margin)."""
    expected = elo_diffs / elo_per_point
    resid = actual_margins - expected
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 2 else 12.0
    return float(np.clip(sigma, 8.0, 18.0))
