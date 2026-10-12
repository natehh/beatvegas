"""Team pace (possessions / 48) model for scaling expected totals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional


@dataclass(frozen=True)
class PaceObservation:
    date: date
    home_team_id: str
    away_team_id: str
    pace: float


@dataclass
class PaceConfig:
    ema_alpha: float = 0.15
    initial_pace: float = 100.0
    season_regression: float = 0.35
    min_pace: float = 88.0
    max_pace: float = 112.0


@dataclass
class PaceModel:
    config: PaceConfig = field(default_factory=PaceConfig)
    pace: dict[str, float] = field(default_factory=dict)
    league_pace: float = 100.0
    _n_obs: int = 0

    def ensure_team(self, team_id: str) -> None:
        if team_id not in self.pace:
            self.pace[team_id] = self.config.initial_pace

    def fit(
        self,
        observations: Iterable[PaceObservation],
        team_ids: Optional[Iterable[str]] = None,
    ) -> None:
        if team_ids is not None:
            for tid in team_ids:
                self.ensure_team(tid)
        for obs in sorted(observations, key=lambda x: (x.date, x.home_team_id)):
            self.update(obs)

    def regress_toward_mean(self, team_ids: Optional[Iterable[str]] = None) -> None:
        ids = list(team_ids) if team_ids is not None else list(self.pace.keys())
        if not ids:
            return
        r = self.config.season_regression
        mean = float(sum(self.pace.get(t, self.config.initial_pace) for t in ids) / len(ids))
        for tid in ids:
            self.ensure_team(tid)
            self.pace[tid] = self.pace[tid] * (1 - r) + mean * r
        self.league_pace = mean

    def update(self, obs: PaceObservation) -> None:
        p = float(obs.pace)
        if p < self.config.min_pace or p > self.config.max_pace:
            return
        self.ensure_team(obs.home_team_id)
        self.ensure_team(obs.away_team_id)
        a = self.config.ema_alpha
        self.pace[obs.home_team_id] = (1 - a) * self.pace[obs.home_team_id] + a * p
        self.pace[obs.away_team_id] = (1 - a) * self.pace[obs.away_team_id] + a * p
        self.league_pace = (1 - a) * self.league_pace + a * p
        self._n_obs += 1

    def expected_pace(self, home_team_id: str, away_team_id: str) -> float:
        self.ensure_team(home_team_id)
        self.ensure_team(away_team_id)
        return 0.5 * (self.pace[home_team_id] + self.pace[away_team_id])

    def pace_scale(self, home_team_id: str, away_team_id: str) -> float:
        league = self.league_pace if self.league_pace > 1 else self.config.initial_pace
        return float(self.expected_pace(home_team_id, away_team_id) / league)
