"""Local JSON cache for NBA schedule/results."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from core.models import Game, SeasonSchedule, Team

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def _cache_path(cache_dir: Path, season: str, kind: str) -> Path:
    safe = season.replace("/", "-")
    return cache_dir / f"nba_{safe}_{kind}.json"


def _serialize_game(g: Game) -> dict[str, Any]:
    return {
        "id": g.id,
        "date": g.date.isoformat(),
        "home_team_id": g.home_team_id,
        "away_team_id": g.away_team_id,
        "home_score": g.home_score,
        "away_score": g.away_score,
        "season": g.season,
        "is_neutral": g.is_neutral,
    }


def _deserialize_game(d: dict[str, Any]) -> Game:
    return Game(
        id=str(d["id"]),
        date=date.fromisoformat(d["date"]),
        home_team_id=str(d["home_team_id"]),
        away_team_id=str(d["away_team_id"]),
        home_score=d.get("home_score"),
        away_score=d.get("away_score"),
        season=d.get("season", ""),
        is_neutral=bool(d.get("is_neutral", False)),
    )


def save_schedule(schedule: SeasonSchedule, cache_dir: Optional[Path] = None, kind: str = "results") -> Path:
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, schedule.season, kind)
    payload = {
        "sport": schedule.sport,
        "season": schedule.season,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "teams": [
            {"id": t.id, "name": t.name, "abbreviation": t.abbreviation} for t in schedule.teams
        ],
        "games": [_serialize_game(g) for g in schedule.games],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_schedule(
    season: str, cache_dir: Optional[Path] = None, kind: str = "results"
) -> Optional[SeasonSchedule]:
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    path = _cache_path(cache_dir, season, kind)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    teams = [
        Team(id=str(t["id"]), name=t["name"], abbreviation=t.get("abbreviation", ""))
        for t in payload.get("teams", [])
    ]
    games = [_deserialize_game(g) for g in payload.get("games", [])]
    return SeasonSchedule(
        sport=payload.get("sport", "nba"),
        season=payload.get("season", season),
        teams=teams,
        games=games,
    )


def clear_cache(season: Optional[str] = None, cache_dir: Optional[Path] = None) -> int:
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    if not cache_dir.exists():
        return 0
    removed = 0
    for path in cache_dir.glob("nba_*.json"):
        if season is None or season.replace("/", "-") in path.name:
            path.unlink()
            removed += 1
    return removed
