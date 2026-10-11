"""Simple Rating System (point differential + strength of schedule)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from core.models import Game


@dataclass
class SrsConfig:
    home_advantage_points: float = 2.5
    iterations: int = 20
    season_regression: float = 1.0 / 3.0
    # Map SRS (points) to Elo-like scale for blending: ~28 Elo ≈ 1 point
    elo_per_point: float = 28.0
    initial: float = 0.0


@dataclass
class SrsEngine:
    """
    Iterative SRS: rating ≈ average (margin - HCA + opp rating).
    Ratings are in points relative to league average (0).
    """

    config: SrsConfig = field(default_factory=SrsConfig)
    ratings: dict[str, float] = field(default_factory=dict)
    _games: list[Game] = field(default_factory=list, repr=False)

    def ensure_team(self, team_id: str) -> None:
        if team_id not in self.ratings:
            self.ratings[team_id] = self.config.initial

    def get_rating(self, team_id: str) -> float:
        self.ensure_team(team_id)
        return self.ratings[team_id]

    def to_elo_scale(self, team_id: str, *, center: float = 1500.0) -> float:
        return center + self.get_rating(team_id) * self.config.elo_per_point

    def fit(self, games: Iterable[Game], team_ids: Optional[Iterable[str]] = None) -> None:
        self._games = [g for g in games if g.is_final]
        if team_ids is not None:
            for tid in team_ids:
                self.ensure_team(tid)
        for g in self._games:
            self.ensure_team(g.home_team_id)
            self.ensure_team(g.away_team_id)
        self._recompute()

    def update(self, game: Game) -> None:
        if not game.is_final:
            return
        self.ensure_team(game.home_team_id)
        self.ensure_team(game.away_team_id)
        self._games.append(game)
        # Cheap online update: one iteration toward new game, then occasional full refresh.
        if len(self._games) % 25 == 0:
            self._recompute()
        else:
            self._single_game_nudge(game)

    def _single_game_nudge(self, game: Game) -> None:
        hca = 0.0 if game.is_neutral else self.config.home_advantage_points
        margin = float(game.home_score) - float(game.away_score)  # type: ignore[arg-type]
        # residual contribution for home: margin - hca - (r_home - r_away)
        # nudge each team 1/n_games fraction — approximate
        home = game.home_team_id
        away = game.away_team_id
        err = margin - hca - (self.ratings[home] - self.ratings[away])
        step = 0.05
        self.ratings[home] += step * err
        self.ratings[away] -= step * err

    def _recompute(self) -> None:
        if not self._games:
            return
        teams = list(self.ratings.keys())
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        if n == 0:
            return

        # Accumulate for each team: sum(margin_adj + opp_rating) / gp
        # Iterate ratings.
        ratings = np.zeros(n, dtype=float)
        for _ in range(self.config.iterations):
            sums = np.zeros(n, dtype=float)
            counts = np.zeros(n, dtype=float)
            for g in self._games:
                h = idx[g.home_team_id]
                a = idx[g.away_team_id]
                hca = 0.0 if g.is_neutral else self.config.home_advantage_points
                margin = float(g.home_score) - float(g.away_score)  # type: ignore[arg-type]
                # Home observed performance vs average opponent: (margin - hca) + opp_rating
                sums[h] += (margin - hca) + ratings[a]
                counts[h] += 1.0
                sums[a] += (-margin + hca) + ratings[h]
                counts[a] += 1.0
            new = np.divide(sums, np.maximum(counts, 1.0))
            # Center to mean 0
            new -= new.mean()
            ratings = new

        for t, i in idx.items():
            self.ratings[t] = float(ratings[i])

    def regress_toward_mean(self, team_ids: Optional[Iterable[str]] = None) -> None:
        ids = list(team_ids) if team_ids is not None else list(self.ratings.keys())
        r = self.config.season_regression
        for tid in ids:
            self.ensure_team(tid)
            self.ratings[tid] = self.ratings[tid] * (1.0 - r)

    def expected_margin(self, home_team_id: str, away_team_id: str, *, is_neutral: bool = False) -> float:
        hca = 0.0 if is_neutral else self.config.home_advantage_points
        return self.get_rating(home_team_id) - self.get_rating(away_team_id) + hca

    def predict_proba(self, home_team_id: str, away_team_id: str, *, is_neutral: bool = False, sigma: float = 12.0) -> float:
        from scipy.stats import norm

        mu = self.expected_margin(home_team_id, away_team_id, is_neutral=is_neutral)
        # P(margin > 0)
        return float(1.0 - norm.cdf(0.0, loc=mu, scale=sigma))

    def snapshot(self) -> dict[str, float]:
        return dict(self.ratings)

    def copy(self) -> "SrsEngine":
        eng = SrsEngine(config=self.config, ratings=self.snapshot())
        eng._games = list(self._games)
        return eng
