"""Sport-agnostic domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Protocol, Sequence


@dataclass(frozen=True)
class Team:
    id: str
    name: str
    abbreviation: str = ""


@dataclass(frozen=True)
class Game:
    id: str
    date: date
    home_team_id: str
    away_team_id: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    season: str = ""
    is_neutral: bool = False

    @property
    def is_final(self) -> bool:
        return self.home_score is not None and self.away_score is not None

    @property
    def home_won(self) -> Optional[bool]:
        if not self.is_final:
            return None
        return self.home_score > self.away_score  # type: ignore[operator]

    @property
    def margin(self) -> Optional[int]:
        if not self.is_final:
            return None
        return int(self.home_score) - int(self.away_score)  # type: ignore[arg-type]


@dataclass
class SeasonSchedule:
    sport: str
    season: str
    teams: list[Team] = field(default_factory=list)
    games: list[Game] = field(default_factory=list)

    def team_map(self) -> dict[str, Team]:
        return {t.id: t for t in self.teams}

    def completed_games(self) -> list[Game]:
        return sorted((g for g in self.games if g.is_final), key=lambda g: (g.date, g.id))

    def remaining_games(self, as_of: Optional[date] = None) -> list[Game]:
        if as_of is None:
            return [g for g in self.games if not g.is_final]
        return [
            g
            for g in self.games
            if (not g.is_final) or (g.date > as_of)
        ]

    def games_through(self, as_of: date) -> list[Game]:
        return sorted(
            (g for g in self.games if g.is_final and g.date <= as_of),
            key=lambda g: (g.date, g.id),
        )

    def actual_wins(self) -> dict[str, int]:
        wins = {t.id: 0 for t in self.teams}
        for g in self.completed_games():
            if g.home_won:
                wins[g.home_team_id] = wins.get(g.home_team_id, 0) + 1
            else:
                wins[g.away_team_id] = wins.get(g.away_team_id, 0) + 1
        return wins

    def actual_losses(self) -> dict[str, int]:
        losses = {t.id: 0 for t in self.teams}
        for g in self.completed_games():
            if g.home_won:
                losses[g.away_team_id] = losses.get(g.away_team_id, 0) + 1
            else:
                losses[g.home_team_id] = losses.get(g.home_team_id, 0) + 1
        return losses


class SportAdapter(Protocol):
    """Protocol for sport-specific data access."""

    sport: str

    def list_teams(self, season: str) -> Sequence[Team]: ...

    def fetch_schedule(self, season: str) -> SeasonSchedule: ...

    def fetch_results(self, season: str) -> SeasonSchedule: ...
