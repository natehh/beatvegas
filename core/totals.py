"""Expected game totals from team offense/defense ratings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
from scipy.stats import norm

from core.models import Game
from core.rest import RestContext


@dataclass
class TotalsConfig:
    # EMA update rate for offense/defense (higher = more recency)
    ema_alpha: float = 0.12
    initial_off: float = 112.0
    initial_def: float = 112.0
    home_total_boost: float = 1.5  # homes slightly inflate totals
    b2b_total_penalty: float = 2.0  # each B2B side lowers expected total
    rest3_total_boost: float = 0.5
    total_sigma: float = 11.0
    season_regression: float = 0.35  # pull toward league mean between seasons
    market_shrink: float = 0.25


@dataclass
class TotalsModel:
    config: TotalsConfig = field(default_factory=TotalsConfig)
    off: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)  # points allowed
    league_avg: float = 112.0
    _n_obs: int = 0

    def ensure_team(self, team_id: str) -> None:
        if team_id not in self.off:
            self.off[team_id] = self.config.initial_off
            self.defense[team_id] = self.config.initial_def

    def fit(self, games: Iterable[Game], team_ids: Optional[Iterable[str]] = None) -> None:
        if team_ids is not None:
            for tid in team_ids:
                self.ensure_team(tid)
        for g in sorted((x for x in games if x.is_final), key=lambda x: (x.date, x.id)):
            self.update(g)

    def regress_toward_mean(self, team_ids: Optional[Iterable[str]] = None) -> None:
        ids = list(team_ids) if team_ids is not None else list(self.off.keys())
        r = self.config.season_regression
        mean_off = float(np.mean([self.off[t] for t in ids])) if ids else self.config.initial_off
        mean_def = float(np.mean([self.defense[t] for t in ids])) if ids else self.config.initial_def
        for tid in ids:
            self.ensure_team(tid)
            self.off[tid] = self.off[tid] * (1 - r) + mean_off * r
            self.defense[tid] = self.defense[tid] * (1 - r) + mean_def * r
        self.league_avg = 0.5 * (mean_off + mean_def)

    def update(self, game: Game) -> None:
        if not game.is_final:
            return
        self.ensure_team(game.home_team_id)
        self.ensure_team(game.away_team_id)
        hs = float(game.home_score)  # type: ignore[arg-type]
        aws = float(game.away_score)  # type: ignore[arg-type]
        a = self.config.ema_alpha
        # Home offense / away defense see home score; away offense / home defense see away score.
        hid, aid = game.home_team_id, game.away_team_id
        self.off[hid] = (1 - a) * self.off[hid] + a * hs
        self.defense[aid] = (1 - a) * self.defense[aid] + a * hs
        self.off[aid] = (1 - a) * self.off[aid] + a * aws
        self.defense[hid] = (1 - a) * self.defense[hid] + a * aws
        self._n_obs += 1
        # Track league average total / 2
        self.league_avg = (1 - a) * self.league_avg + a * (0.5 * (hs + aws))

    def expected_team_points(
        self,
        team_id: str,
        opp_id: str,
        *,
        is_home: bool,
    ) -> float:
        self.ensure_team(team_id)
        self.ensure_team(opp_id)
        # Blend team's offense with opponent's defense around league average.
        base = 0.5 * (self.off[team_id] + self.defense[opp_id])
        if is_home:
            base += self.config.home_total_boost * 0.5
        return base

    def expected_total(
        self,
        home_team_id: str,
        away_team_id: str,
        *,
        rest: Optional[RestContext] = None,
        pace_scale: float = 1.0,
        total_adj: float = 0.0,
    ) -> float:
        home_pts = self.expected_team_points(home_team_id, away_team_id, is_home=True)
        away_pts = self.expected_team_points(away_team_id, home_team_id, is_home=False)
        total = home_pts + away_pts
        if rest is not None:
            if rest.home_b2b:
                total -= self.config.b2b_total_penalty
            if rest.away_b2b:
                total -= self.config.b2b_total_penalty
            if rest.home_days_rest >= 3:
                total += self.config.rest3_total_boost
            if rest.away_days_rest >= 3:
                total += self.config.rest3_total_boost
        total = total * float(pace_scale) + float(total_adj)
        return float(total)

    def over_prob(
        self,
        home_team_id: str,
        away_team_id: str,
        line: float,
        *,
        rest: Optional[RestContext] = None,
        pace_scale: float = 1.0,
        total_adj: float = 0.0,
    ) -> float:
        mu = self.expected_total(
            home_team_id,
            away_team_id,
            rest=rest,
            pace_scale=pace_scale,
            total_adj=total_adj,
        )
        # P(total > line). For .5 lines pushes are impossible.
        return float(1.0 - norm.cdf(float(line), loc=mu, scale=self.config.total_sigma))

    def shrink_to_market(self, model_p: float, market_p: float) -> float:
        a = self.config.market_shrink
        return float((1.0 - a) * model_p + a * market_p)

    def calibrate_sigma(self, games: Iterable[Game]) -> float:
        errs = []
        # Use current ratings statically on prior games for scale estimate.
        for g in games:
            if not g.is_final:
                continue
            mu = self.expected_total(g.home_team_id, g.away_team_id)
            actual = float(g.home_score) + float(g.away_score)  # type: ignore[arg-type]
            errs.append(actual - mu)
        if len(errs) < 50:
            return self.config.total_sigma
        sigma = float(np.clip(np.std(errs, ddof=1), 8.0, 16.0))
        self.config.total_sigma = sigma
        return sigma
