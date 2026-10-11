"""Official NBA injury reports via nbainjuries (free static PDF snapshots)."""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable, Optional

from core.injuries import InjuryListing
from sports.nba import cache as nba_cache
from sports.nba.teams import team_id_from_name

# Prefer late-afternoon ET snapshot (common pre-tip report).
DEFAULT_REPORT_HOUR = 17
DEFAULT_REPORT_MINUTE = 30


def _cache_path(cache_dir: Path, season: str) -> Path:
    return cache_dir / f"nba_{season.replace('/', '-')}_injuries.json"


def _try_report(dt: datetime):
    from nbainjuries import injury

    try:
        return injury.get_reportdata(dt, return_df=True)
    except Exception:  # noqa: BLE001
        return None


def fetch_report_for_date(
    game_date: date,
    *,
    hour: int = DEFAULT_REPORT_HOUR,
    minute: int = DEFAULT_REPORT_MINUTE,
) -> list[InjuryListing]:
    """Fetch the best available pre-game report for a calendar date."""
    candidates = [
        datetime.combine(game_date, time(hour=hour, minute=minute)),
        datetime.combine(game_date, time(hour=14, minute=0)),
        datetime.combine(game_date, time(hour=12, minute=0)),
        datetime.combine(game_date - timedelta(days=1), time(hour=hour, minute=minute)),
    ]
    df = None
    for dt in candidates:
        df = _try_report(dt)
        if df is not None and len(df) > 0:
            break
    if df is None or len(df) == 0:
        return []

    listings: list[InjuryListing] = []
    for _, row in df.iterrows():
        team_name = str(row.get("Team", "")).strip()
        player = str(row.get("Player Name", "")).strip()
        status = str(row.get("Current Status", "")).strip()
        reason = str(row.get("Reason", "") or "").strip()
        if not team_name or not player or not status:
            continue
        try:
            team_id = team_id_from_name(team_name)
        except Exception:  # noqa: BLE001
            continue
        listings.append(
            InjuryListing(
                date=game_date,
                team_id=team_id,
                player=player,
                status=status,
                reason=reason,
            )
        )
    return listings


def load_injuries_for_dates(
    season: str,
    dates: Iterable[date],
    *,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> dict[str, list[InjuryListing]]:
    cache_dir = cache_dir or nba_cache.DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, season)

    cached: dict[str, list[dict]] = {}
    if path.exists() and not force_refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cached = payload.get("by_date", {})

    uniq_dates = sorted(set(dates))
    by_date: dict[str, list[InjuryListing]] = {}
    dirty = False
    for d in uniq_dates:
        key = d.isoformat()
        if key in cached and not force_refresh:
            by_date[key] = [
                InjuryListing(
                    date=date.fromisoformat(x["date"]),
                    team_id=x["team_id"],
                    player=x["player"],
                    status=x["status"],
                    reason=x.get("reason", ""),
                )
                for x in cached[key]
            ]
            continue
        listings = fetch_report_for_date(d)
        by_date[key] = listings
        cached[key] = [
            {
                "date": x.date.isoformat(),
                "team_id": x.team_id,
                "player": x.player,
                "status": x.status,
                "reason": x.reason,
            }
            for x in listings
        ]
        dirty = True
        # Persist incrementally
        path.write_text(
            json.dumps(
                {
                    "season": season,
                    "source": "nbainjuries / NBA official injury reports",
                    "fetched_at": datetime.utcnow().isoformat() + "Z",
                    "n_dates": len(cached),
                    "by_date": cached,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if dirty or not path.exists():
        path.write_text(
            json.dumps(
                {
                    "season": season,
                    "source": "nbainjuries / NBA official injury reports",
                    "fetched_at": datetime.utcnow().isoformat() + "Z",
                    "n_dates": len(cached),
                    "by_date": cached,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return by_date


def listings_on(by_date: dict[str, list[InjuryListing]], game_date: date) -> list[InjuryListing]:
    return list(by_date.get(game_date.isoformat(), []))
