"""Rest / back-to-back features from schedule chronology."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from core.models import Game


@dataclass(frozen=True)
class RestContext:
    home_days_rest: int
    away_days_rest: int

    @property
    def home_b2b(self) -> bool:
        # days_rest counts full days off between games: playing Mon then Wed
        # is 1 day of rest, not a back-to-back. Only 0 days off is a B2B.
        return self.home_days_rest == 0

    @property
    def away_b2b(self) -> bool:
        return self.away_days_rest == 0

    def margin_adjustment(self, *, b2b_penalty: float = 2.2, rest3_bonus: float = 0.5) -> float:
        """
        Points added to expected home margin.
        NBA-ish: B2B is ~1.5–3 pt penalty; slight bonus for 3+ days rest.
        """
        adj = 0.0
        if self.home_b2b:
            adj -= b2b_penalty
        if self.away_b2b:
            adj += b2b_penalty
        if self.home_days_rest >= 3:
            adj += rest3_bonus
        if self.away_days_rest >= 3:
            adj -= rest3_bonus
        return adj


class RestTracker:
    """Track last play date per team while walking a season chronologically."""

    def __init__(self) -> None:
        self.last_played: dict[str, date] = {}

    def context(self, game: Game) -> RestContext:
        home_rest = self._days_rest(game.home_team_id, game.date)
        away_rest = self._days_rest(game.away_team_id, game.date)
        return RestContext(home_days_rest=home_rest, away_days_rest=away_rest)

    def observe(self, game: Game) -> None:
        self.last_played[game.home_team_id] = game.date
        self.last_played[game.away_team_id] = game.date

    def _days_rest(self, team_id: str, on: date) -> int:
        prev = self.last_played.get(team_id)
        if prev is None:
            return 5  # season start / unknown → treat as rested
        delta = (on - prev).days
        # days rest = calendar gap - 1 (play Mon, play Wed => 1 day rest)
        return max(0, delta - 1)


def build_rest_map(games: list[Game]) -> dict[str, RestContext]:
    """Map game.id -> RestContext for a chronological season."""
    tracker = RestTracker()
    out: dict[str, RestContext] = {}
    for g in sorted(games, key=lambda x: (x.date, x.id)):
        ctx = tracker.context(g)
        out[g.id] = ctx
        if g.is_final:
            tracker.observe(g)
    return out
