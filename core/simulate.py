"""Monte Carlo season simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from core.elo import EloEngine
from core.models import Game, Team


@dataclass
class SimulationResult:
    n_sims: int
    expected_wins: dict[str, float]
    win_samples: dict[str, np.ndarray] = field(repr=False)
    team_names: dict[str, str] = field(default_factory=dict)

    def expected_losses(self, games_played: dict[str, int]) -> dict[str, float]:
        return {
            tid: float(games_played.get(tid, 0) - self.expected_wins.get(tid, 0.0))
            for tid in self.expected_wins
        }

    def to_records(self, games_played: Optional[dict[str, int]] = None) -> list[dict]:
        rows = []
        for tid, exp_w in sorted(self.expected_wins.items(), key=lambda x: -x[1]):
            samples = self.win_samples[tid]
            gp = games_played.get(tid) if games_played else None
            rows.append(
                {
                    "team_id": tid,
                    "team": self.team_names.get(tid, tid),
                    "expected_wins": round(float(exp_w), 2),
                    "expected_losses": round(float(gp - exp_w), 2) if gp is not None else None,
                    "p25_wins": float(np.percentile(samples, 25)),
                    "p50_wins": float(np.percentile(samples, 50)),
                    "p75_wins": float(np.percentile(samples, 75)),
                }
            )
        return rows


@dataclass
class SeasonSimulator:
    """Simulate a list of games given a frozen Elo snapshot (no in-sim updates by default)."""

    engine: EloEngine
    update_during_sim: bool = False

    def simulate(
        self,
        games: Sequence[Game],
        teams: Sequence[Team],
        n_sims: int = 5000,
        *,
        seed: Optional[int] = 42,
        prior_wins: Optional[dict[str, int]] = None,
    ) -> SimulationResult:
        if n_sims < 1:
            raise ValueError("n_sims must be >= 1")

        rng = np.random.default_rng(seed)
        team_ids = [t.id for t in teams]
        team_names = {t.id: t.name for t in teams}
        prior = prior_wins or {tid: 0 for tid in team_ids}

        # Precompute win probabilities for frozen ratings (fast path).
        if not self.update_during_sim:
            probs = np.array(
                [
                    self.engine.predict_proba(g.home_team_id, g.away_team_id, is_neutral=g.is_neutral)
                    for g in games
                ],
                dtype=float,
            )
            # Shape: (n_sims, n_games)
            draws = rng.random((n_sims, len(games)))
            home_wins = draws < probs

            win_counts = {tid: np.full(n_sims, float(prior.get(tid, 0)), dtype=float) for tid in team_ids}
            for i, g in enumerate(games):
                hw = home_wins[:, i]
                win_counts[g.home_team_id] = win_counts.get(
                    g.home_team_id, np.zeros(n_sims)
                ) + hw.astype(float)
                win_counts[g.away_team_id] = win_counts.get(
                    g.away_team_id, np.zeros(n_sims)
                ) + (1.0 - hw.astype(float))
        else:
            win_counts = {tid: np.zeros(n_sims, dtype=float) for tid in team_ids}
            for tid in team_ids:
                win_counts[tid] += float(prior.get(tid, 0))
            base = self.engine.snapshot()
            for s in range(n_sims):
                eng = EloEngine(config=self.engine.config, ratings=dict(base))
                for g in games:
                    p = eng.predict_proba(g.home_team_id, g.away_team_id, is_neutral=g.is_neutral)
                    home_won = bool(rng.random() < p)
                    if home_won:
                        win_counts[g.home_team_id][s] += 1.0
                        margin = max(1, int(round(abs(eng.get_rating(g.home_team_id) - eng.get_rating(g.away_team_id)) / 25)))
                        sim_game = Game(
                            id=g.id,
                            date=g.date,
                            home_team_id=g.home_team_id,
                            away_team_id=g.away_team_id,
                            home_score=100 + margin,
                            away_score=100,
                            season=g.season,
                            is_neutral=g.is_neutral,
                        )
                    else:
                        win_counts[g.away_team_id][s] += 1.0
                        margin = max(1, int(round(abs(eng.get_rating(g.home_team_id) - eng.get_rating(g.away_team_id)) / 25)))
                        sim_game = Game(
                            id=g.id,
                            date=g.date,
                            home_team_id=g.home_team_id,
                            away_team_id=g.away_team_id,
                            home_score=100,
                            away_score=100 + margin,
                            season=g.season,
                            is_neutral=g.is_neutral,
                        )
                    eng.update(sim_game)

        expected = {tid: float(np.mean(samples)) for tid, samples in win_counts.items()}
        return SimulationResult(
            n_sims=n_sims,
            expected_wins=expected,
            win_samples=win_counts,
            team_names=team_names,
        )
