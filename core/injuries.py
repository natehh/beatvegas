"""Minutes-weighted injury adjustments for side and total markets."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional


def normalize_player_name(name: str) -> str:
    """Normalize 'Last, First' / 'First Last' / accented names to a match key."""
    text = unicodedata.normalize("NFKD", str(name).strip())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    # Repair common UTF-8→latin1 mojibake leftovers by keeping a-z only tokens later
    text = text.lower().replace(".", " ").replace("*", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if "," in text:
        last, first = text.split(",", 1)
        text = f"{first.strip()} {last.strip()}"
    # Drop generational suffixes for matching
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # ASCII-only key so 'porzingis' matches even if source had encoding glitches
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


STATUS_WEIGHTS = {
    "out": 1.0,
    "doubtful": 0.75,
    "questionable": 0.25,
    "probable": 0.10,
    "available": 0.0,
}


@dataclass(frozen=True)
class InjuryListing:
    date: date
    team_id: str
    player: str
    status: str
    reason: str = ""

    @property
    def player_key(self) -> str:
        return normalize_player_name(self.player)

    @property
    def status_weight(self) -> float:
        return STATUS_WEIGHTS.get(self.status.strip().lower(), 0.0)


@dataclass(frozen=True)
class PlayerMinuteLine:
    date: date
    team_id: str
    player: str
    minutes: float

    @property
    def player_key(self) -> str:
        return normalize_player_name(self.player)


@dataclass
class InjuryConfig:
    mpg_coeff: float = 4.0  # margin points per 36 missing weighted mpg
    total_coeff: float = 2.5  # total points removed per 36 missing weighted mpg (both teams)
    questionable_weight: float = 0.25
    doubtful_weight: float = 0.75
    probable_weight: float = 0.10
    minutes_ema_alpha: float = 0.25
    default_mpg: float = 0.0  # unknown players: ignore unless they play
    min_mpg_to_count: float = 8.0
    lookback_n: int = 8


@dataclass
class MinutesTracker:
    config: InjuryConfig = field(default_factory=InjuryConfig)
    mpg: dict[str, float] = field(default_factory=dict)
    team_of: dict[str, str] = field(default_factory=dict)
    _history: dict[str, list[float]] = field(default_factory=dict)

    def get_mpg(self, player_key: str) -> float:
        return float(self.mpg.get(player_key, self.config.default_mpg))

    def update_line(self, line: PlayerMinuteLine) -> None:
        key = line.player_key
        if not key:
            return
        mins = float(line.minutes)
        if mins < 0:
            return
        hist = self._history.setdefault(key, [])
        hist.append(mins)
        if len(hist) > 40:
            del hist[:-40]
        window = hist[-self.config.lookback_n :]
        self.mpg[key] = float(sum(window) / len(window))
        self.team_of[key] = line.team_id

    def fit(self, lines: Iterable[PlayerMinuteLine]) -> None:
        for line in sorted(lines, key=lambda x: (x.date, x.team_id, x.player_key)):
            self.update_line(line)


@dataclass
class InjuryImpact:
    home_missing_mpg: float
    away_missing_mpg: float
    margin_adj: float  # home perspective
    total_adj: float


@dataclass
class InjuryModel:
    config: InjuryConfig = field(default_factory=InjuryConfig)
    minutes: MinutesTracker = field(default_factory=MinutesTracker)

    def __post_init__(self) -> None:
        self.minutes.config = self.config

    def _weight(self, status: str) -> float:
        s = status.strip().lower()
        if s == "questionable":
            return self.config.questionable_weight
        if s == "doubtful":
            return self.config.doubtful_weight
        if s == "probable":
            return self.config.probable_weight
        return STATUS_WEIGHTS.get(s, 0.0)

    def missing_weighted_mpg(
        self,
        team_id: str,
        listings: Iterable[InjuryListing],
    ) -> float:
        total = 0.0
        seen: set[str] = set()
        for listing in listings:
            if listing.team_id != team_id:
                continue
            key = listing.player_key
            if not key or key in seen:
                continue
            seen.add(key)
            w = self._weight(listing.status)
            if w <= 0:
                continue
            mpg = self.minutes.get_mpg(key)
            if mpg < self.config.min_mpg_to_count:
                continue
            total += w * mpg
        return float(total)

    def impact(
        self,
        home_team_id: str,
        away_team_id: str,
        listings: Iterable[InjuryListing],
    ) -> InjuryImpact:
        rows = list(listings)
        home_m = self.missing_weighted_mpg(home_team_id, rows)
        away_m = self.missing_weighted_mpg(away_team_id, rows)
        # Positive margin_adj favors home (away missing more talent).
        margin_adj = self.config.mpg_coeff * ((away_m - home_m) / 36.0)
        total_adj = -self.config.total_coeff * ((home_m + away_m) / 36.0)
        return InjuryImpact(
            home_missing_mpg=home_m,
            away_missing_mpg=away_m,
            margin_adj=float(margin_adj),
            total_adj=float(total_adj),
        )
