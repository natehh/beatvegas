"""Fetch per-game pace from Basketball-Reference advanced team gamelogs."""

from __future__ import annotations

import json
import time
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from core.pace import PaceObservation
from sports.nba import cache as nba_cache
from sports.nba.bref import season_end_year
from sports.nba.teams import TEAM_NAME_TO_ABBR, from_bref_abbr, to_bref_abbr

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; beatvegas/0.1; +https://github.com/local/beatvegas)",
    "Accept-Language": "en-US,en;q=0.9",
}

# Stable 30-team set for modern seasons
NBA_TEAMS = sorted({v for v in TEAM_NAME_TO_ABBR.values()})


def _cache_path(cache_dir: Path, season: str) -> Path:
    return cache_dir / f"nba_{season.replace('/', '-')}_pace.json"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(str(x) for x in col if "Unnamed" not in str(x)).strip("_")
            for col in df.columns
        ]
    return df


def fetch_team_pace_gamelog(
    team_id: str,
    season: str,
    *,
    delay_s: float = 0.7,
) -> list[dict]:
    end_year = season_end_year(season)
    bref = to_bref_abbr(team_id)
    url = f"https://www.basketball-reference.com/teams/{bref}/{end_year}/gamelog-advanced/"
    resp = requests.get(url, headers=HEADERS, timeout=45)
    resp.raise_for_status()
    html = resp.text.replace("<!--", "").replace("-->", "")
    tables = pd.read_html(StringIO(html))
    if not tables:
        return []
    df = _flatten_columns(tables[0])
    # Regular season table is first; drop header repeats
    if "Date" not in df.columns:
        raise RuntimeError(f"No Date column in pace gamelog for {team_id}: {list(df.columns)}")
    pace_col = next((c for c in df.columns if str(c).endswith("Pace") or str(c) == "Pace"), None)
    if pace_col is None:
        raise RuntimeError(f"No Pace column for {team_id}: {list(df.columns)}")
    opp_col = "Opp" if "Opp" in df.columns else next(c for c in df.columns if str(c).endswith("Opp"))
    rows = []
    for _, row in df.iterrows():
        date_raw = str(row.get("Date", "")).strip()
        if not re_date(date_raw):
            continue
        try:
            gdate = date.fromisoformat(date_raw[:10])
        except ValueError:
            continue
        try:
            pace = float(row[pace_col])
        except (TypeError, ValueError):
            continue
        opp_raw = str(row.get(opp_col, "")).strip().upper()
        if not opp_raw or opp_raw == "OPP":
            continue
        opp = from_bref_abbr(opp_raw)
        rows.append(
            {
                "date": gdate.isoformat(),
                "team_id": team_id,
                "opp_id": opp,
                "pace": pace,
            }
        )
    if delay_s > 0:
        time.sleep(delay_s)
    return rows


def re_date(text: str) -> bool:
    return bool(text) and text[:4].isdigit() and "-" in text


def load_pace_observations(
    season: str,
    *,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
    delay_s: float = 0.7,
    home_away_lookup: Optional[dict[tuple[date, str, str], tuple[str, str]]] = None,
) -> list[PaceObservation]:
    """
    Load per-game pace.

    home_away_lookup maps (date, team_lo, team_hi) -> (home_id, away_id) when available
    so observations carry true home/away ids.
    """
    cache_dir = cache_dir or nba_cache.DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, season)
    if path.exists() and not force_refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [
            PaceObservation(
                date=date.fromisoformat(d["date"]),
                home_team_id=d["home_team_id"],
                away_team_id=d["away_team_id"],
                pace=float(d["pace"]),
            )
            for d in payload["observations"]
        ]

    team_rows: list[dict] = []
    errors: list[str] = []
    for tid in NBA_TEAMS:
        try:
            team_rows.extend(fetch_team_pace_gamelog(tid, season, delay_s=delay_s))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{tid}: {exc}")
    if len(team_rows) < 500:
        raise RuntimeError(f"Pace fetch incomplete for {season}: {len(team_rows)} rows; {errors[:5]}")

    # Deduplicate games
    games: dict[tuple, dict] = {}
    for r in team_rows:
        gdate = date.fromisoformat(r["date"])
        a, b = sorted([r["team_id"], r["opp_id"]])
        key = (gdate, a, b)
        games[key] = {"date": gdate, "a": a, "b": b, "pace": float(r["pace"])}

    observations: list[PaceObservation] = []
    for (gdate, a, b), g in sorted(games.items()):
        home, away = a, b
        if home_away_lookup is not None:
            resolved = home_away_lookup.get((gdate, a, b))
            if resolved is not None:
                home, away = resolved
        observations.append(
            PaceObservation(date=gdate, home_team_id=home, away_team_id=away, pace=g["pace"])
        )

    path.write_text(
        json.dumps(
            {
                "season": season,
                "source": "basketball-reference advanced gamelog",
                "fetched_at": datetime.utcnow().isoformat() + "Z",
                "n": len(observations),
                "errors": errors,
                "observations": [
                    {
                        "date": o.date.isoformat(),
                        "home_team_id": o.home_team_id,
                        "away_team_id": o.away_team_id,
                        "pace": o.pace,
                    }
                    for o in observations
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return observations


def home_away_lookup_from_schedule(games) -> dict[tuple[date, str, str], tuple[str, str]]:
    out: dict[tuple[date, str, str], tuple[str, str]] = {}
    for g in games:
        a, b = sorted([g.home_team_id, g.away_team_id])
        out[(g.date, a, b)] = (g.home_team_id, g.away_team_id)
    return out
