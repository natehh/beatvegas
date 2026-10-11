"""FiveThirtyEight-style Elo rating engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

import numpy as np

from core.models import Game


@dataclass
class EloConfig:
    initial_rating: float = 1500.0
    home_advantage: float = 100.0
    k_factor: float = 20.0
    mov_multiplier_cap: float = 1.75
    season_regression: float = 1.0 / 3.0
    scale: float = 400.0


@dataclass
class EloEngine:
    config: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)

    def ensure_team(self, team_id: str) -> None:
        if team_id not in self.ratings:
            self.ratings[team_id] = self.config.initial_rating

    def get_rating(self, team_id: str) -> float:
        self.ensure_team(team_id)
        return self.ratings[team_id]

    def predict_proba(self, home_team_id: str, away_team_id: str, *, is_neutral: bool = False) -> float:
        """Return P(home win)."""
        home = self.get_rating(home_team_id)
        away = self.get_rating(away_team_id)
        home_edge = 0.0 if is_neutral else self.config.home_advantage
        diff = home + home_edge - away
        return float(1.0 / (1.0 + 10 ** (-diff / self.config.scale)))

    def _mov_multiplier(self, margin: int, elo_diff: float) -> float:
        """538-style margin-of-victory multiplier."""
        abs_margin = abs(margin)
        # Autocorrelation adjustment: reduce credit when favorite wins big as expected.
        multiplier = np.log(abs_margin + 1.0) * (2.2 / (0.001 * abs(elo_diff) + 2.2))
        return float(min(self.config.mov_multiplier_cap, max(0.0, multiplier)))

    def update(self, game: Game) -> None:
        if not game.is_final or game.home_won is None or game.margin is None:
            raise ValueError(f"Cannot update Elo from unfinished game {game.id}")

        home_id, away_id = game.home_team_id, game.away_team_id
        self.ensure_team(home_id)
        self.ensure_team(away_id)

        home_edge = 0.0 if game.is_neutral else self.config.home_advantage
        elo_diff = self.ratings[home_id] + home_edge - self.ratings[away_id]
        expected_home = self.predict_proba(home_id, away_id, is_neutral=game.is_neutral)
        actual_home = 1.0 if game.home_won else 0.0
        mov = self._mov_multiplier(game.margin, elo_diff)
        delta = self.config.k_factor * mov * (actual_home - expected_home)

        self.ratings[home_id] += delta
        self.ratings[away_id] -= delta

    def fit(self, games: Iterable[Game], team_ids: Optional[Iterable[str]] = None) -> None:
        if team_ids is not None:
            for tid in team_ids:
                self.ensure_team(tid)
        ordered = sorted(games, key=lambda g: (g.date, g.id))
        for game in ordered:
            if game.is_final:
                self.update(game)

    def regress_toward_mean(self, team_ids: Optional[Iterable[str]] = None) -> None:
        """Apply between-season regression toward league mean / initial rating."""
        ids = list(team_ids) if team_ids is not None else list(self.ratings.keys())
        if not ids:
            return
        mean = float(np.mean([self.get_rating(t) for t in ids]))
        target = mean  # regress toward current league mean
        r = self.config.season_regression
        for tid in ids:
            current = self.get_rating(tid)
            self.ratings[tid] = current * (1.0 - r) + target * r

    def snapshot(self) -> dict[str, float]:
        return dict(self.ratings)

    def load_ratings(self, ratings: Mapping[str, float]) -> None:
        self.ratings = {k: float(v) for k, v in ratings.items()}

    def copy(self) -> "EloEngine":
        return EloEngine(config=self.config, ratings=self.snapshot())
