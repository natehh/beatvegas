"""Ensemble team-strength model: dual Elo + SRS + rest adjustments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
from scipy.stats import norm

from core.elo import EloConfig, EloEngine
from core.models import Game
from core.rest import RestContext, RestTracker
from core.srs import SrsConfig, SrsEngine


@dataclass
class EnsembleConfig:
    # Dual Elo: slow long-term + fast recency
    slow_k: float = 12.0
    fast_k: float = 36.0
    slow_weight: float = 0.55
    fast_weight: float = 0.45
    home_advantage_elo: float = 100.0
    season_regression: float = 1.0 / 3.0
    # Blend strengths in probability / margin space
    elo_blend: float = 0.55  # vs SRS
    srs_blend: float = 0.45
    elo_per_point: float = 28.0
    margin_sigma: float = 12.0
    b2b_penalty: float = 2.2
    rest3_bonus: float = 0.5
    # Shrink model probability toward market when provided (0 = none, 0.35 typical)
    market_shrink: float = 0.35


@dataclass
class EnsembleModel:
    config: EnsembleConfig = field(default_factory=EnsembleConfig)
    slow: EloEngine = field(init=False)
    fast: EloEngine = field(init=False)
    srs: SrsEngine = field(init=False)
    rest: RestTracker = field(default_factory=RestTracker)

    def __post_init__(self) -> None:
        self.slow = EloEngine(
            EloConfig(
                k_factor=self.config.slow_k,
                home_advantage=self.config.home_advantage_elo,
                season_regression=self.config.season_regression,
            )
        )
        self.fast = EloEngine(
            EloConfig(
                k_factor=self.config.fast_k,
                home_advantage=self.config.home_advantage_elo,
                season_regression=self.config.season_regression,
            )
        )
        self.srs = SrsEngine(
            SrsConfig(
                elo_per_point=self.config.elo_per_point,
                season_regression=self.config.season_regression,
            )
        )

    def ensure_teams(self, team_ids: Iterable[str]) -> None:
        for tid in team_ids:
            self.slow.ensure_team(tid)
            self.fast.ensure_team(tid)
            self.srs.ensure_team(tid)

    def fit(self, games: Iterable[Game], team_ids: Optional[Iterable[str]] = None) -> None:
        games_list = sorted((g for g in games if g.is_final), key=lambda g: (g.date, g.id))
        if team_ids is not None:
            self.ensure_teams(team_ids)
        self.slow.fit(games_list, team_ids)
        self.fast.fit(games_list, team_ids)
        self.srs.fit(games_list, team_ids)
        self.rest = RestTracker()
        for g in games_list:
            self.rest.observe(g)

    def regress_toward_mean(self, team_ids: Optional[Iterable[str]] = None) -> None:
        self.slow.regress_toward_mean(team_ids)
        self.fast.regress_toward_mean(team_ids)
        self.srs.regress_toward_mean(team_ids)

    def update(self, game: Game) -> None:
        if not game.is_final:
            return
        self.slow.update(game)
        self.fast.update(game)
        self.srs.update(game)
        self.rest.observe(game)

    def rest_context(self, game: Game) -> RestContext:
        return self.rest.context(game)

    def blended_elo(self, team_id: str) -> float:
        return (
            self.config.slow_weight * self.slow.get_rating(team_id)
            + self.config.fast_weight * self.fast.get_rating(team_id)
        )

    def expected_margin(
        self,
        home_team_id: str,
        away_team_id: str,
        *,
        is_neutral: bool = False,
        rest: Optional[RestContext] = None,
    ) -> float:
        hca_elo = 0.0 if is_neutral else self.config.home_advantage_elo
        elo_diff = self.blended_elo(home_team_id) + hca_elo - self.blended_elo(away_team_id)
        elo_margin = elo_diff / self.config.elo_per_point
        srs_margin = self.srs.expected_margin(home_team_id, away_team_id, is_neutral=is_neutral)
        margin = self.config.elo_blend * elo_margin + self.config.srs_blend * srs_margin
        if rest is not None:
            margin += rest.margin_adjustment(
                b2b_penalty=self.config.b2b_penalty,
                rest3_bonus=self.config.rest3_bonus,
            )
        return float(margin)

    def predict_proba(
        self,
        home_team_id: str,
        away_team_id: str,
        *,
        is_neutral: bool = False,
        rest: Optional[RestContext] = None,
    ) -> float:
        mu = self.expected_margin(
            home_team_id, away_team_id, is_neutral=is_neutral, rest=rest
        )
        # P(home win) ≈ P(margin > 0)
        return float(1.0 - norm.cdf(0.0, loc=mu, scale=self.config.margin_sigma))

    def cover_prob(
        self,
        home_team_id: str,
        away_team_id: str,
        home_spread_line: float,
        *,
        is_neutral: bool = False,
        rest: Optional[RestContext] = None,
    ) -> float:
        mu = self.expected_margin(
            home_team_id, away_team_id, is_neutral=is_neutral, rest=rest
        )
        threshold = -float(home_spread_line)
        return float(1.0 - norm.cdf(threshold, loc=mu, scale=self.config.margin_sigma))

    def shrink_to_market(self, model_p: float, market_p: float) -> float:
        a = self.config.market_shrink
        return float((1.0 - a) * model_p + a * market_p)

    def calibrate_sigma_from_prior(self, games: Iterable[Game]) -> float:
        diffs = []
        margins = []
        # Rebuild chronological expected margins with a throwaway model copy path:
        # Use post-fit ratings statically on prior games (imperfect but stable).
        for g in sorted((x for x in games if x.is_final), key=lambda x: (x.date, x.id)):
            # Approximate with current ratings (after fit) — residual scale estimate.
            mu = self.expected_margin(g.home_team_id, g.away_team_id, is_neutral=g.is_neutral)
            actual = float(g.home_score) - float(g.away_score)  # type: ignore[arg-type]
            diffs.append(mu)
            margins.append(actual)
        if len(margins) < 50:
            return self.config.margin_sigma
        resid = np.array(margins) - np.array(diffs)
        sigma = float(np.clip(np.std(resid, ddof=1), 8.0, 18.0))
        self.config.margin_sigma = sigma
        return sigma
