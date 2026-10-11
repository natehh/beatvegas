"""Basketball-Reference schedule/results fetcher (free, no API key)."""

from __future__ import annotations

import re
import time
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests
from io import StringIO

from core.models import Game, SeasonSchedule, Team
from sports.nba.teams import TEAM_NAME_TO_ABBR, team_id_from_name

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; beatvegas/0.1; +https://github.com/local/beatvegas)",
    "Accept-Language": "en-US,en;q=0.9",
}

MONTHS = ("october", "november", "december", "january", "february", "march", "april", "may", "june")


def season_end_year(season: str) -> int:
    """'2024-25' -> 2025."""
    start = int(season.split("-")[0])
    return start + 1


def regular_season_end_date(end_year: int) -> date:
    """
    Approximate regular-season cutoff (before play-in/playoffs).
    Mid-April; refined by known recent seasons when helpful.
    """
    known = {
        2024: date(2024, 4, 14),
        2025: date(2025, 4, 13),
        2026: date(2026, 4, 12),
    }
    return known.get(end_year, date(end_year, 4, 14))


def fetch_season_from_bref(season: str, *, delay_s: float = 0.8) -> SeasonSchedule:
    end_year = season_end_year(season)
    cutoff = regular_season_end_date(end_year)
    frames: list[pd.DataFrame] = []

    for month in MONTHS:
        url = f"https://www.basketball-reference.com/leagues/NBA_{end_year}_games-{month}.html"
        resp = requests.get(url, headers=HEADERS, timeout=45)
        if resp.status_code == 404:
            continue
        resp.raise_for_status()
        html = resp.text.replace("<!--", "").replace("-->", "")
        try:
            tables = pd.read_html(StringIO(html))
        except ValueError:
            continue
        if not tables:
            continue
        df = tables[0]
        frames.append(df)
        if delay_s > 0:
            time.sleep(delay_s)

    if not frames:
        raise RuntimeError(f"No Basketball-Reference schedule tables for {season}")

    raw = pd.concat(frames, ignore_index=True)
    # Normalize columns
    cols = {c: str(c).strip() for c in raw.columns}
    raw = raw.rename(columns=cols)
    if "Visitor/Neutral" not in raw.columns or "Home/Neutral" not in raw.columns:
        raise RuntimeError(f"Unexpected BRef columns: {list(raw.columns)}")

    pts_vis = "PTS"
    pts_home = "PTS.1" if "PTS.1" in raw.columns else None
    if pts_home is None:
        # sometimes duplicated PTS labels become PTS / PTS.1 after read_html
        pts_cols = [c for c in raw.columns if str(c).startswith("PTS")]
        if len(pts_cols) >= 2:
            pts_vis, pts_home = pts_cols[0], pts_cols[1]
        else:
            raise RuntimeError("Could not find home/away PTS columns")

    teams: dict[str, Team] = {}
    games: list[Game] = []

    for idx, row in raw.iterrows():
        date_raw = str(row.get("Date", "")).strip()
        if not date_raw or not re.search(r"\d{4}", date_raw):
            continue
        try:
            gdate = datetime.strptime(date_raw, "%a, %b %d, %Y").date()
        except ValueError:
            try:
                gdate = pd.to_datetime(date_raw).date()
            except Exception:
                continue

        if gdate > cutoff:
            continue

        away_name = str(row["Visitor/Neutral"]).strip()
        home_name = str(row["Home/Neutral"]).strip()
        if away_name.lower() in TEAM_NAME_TO_ABBR or True:
            pass
        if not away_name or not home_name or away_name == "nan" or home_name == "nan":
            continue

        away_id = team_id_from_name(away_name)
        home_id = team_id_from_name(home_name)
        teams[home_id] = Team(id=home_id, name=home_name, abbreviation=home_id)
        teams[away_id] = Team(id=away_id, name=away_name, abbreviation=away_id)

        def _score(val) -> Optional[int]:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            text = str(val).strip()
            if not text or text.lower() == "nan":
                return None
            try:
                return int(float(text))
            except ValueError:
                return None

        home_score = _score(row[pts_home])
        away_score = _score(row[pts_vis])
        # Skip unplayed
        if home_score is None or away_score is None:
            continue

        notes = str(row.get("Notes", "") or "")
        is_neutral = "neutral" in notes.lower()

        game_id = f"{season}-{gdate.isoformat()}-{away_id}@{home_id}-{idx}"
        games.append(
            Game(
                id=game_id,
                date=gdate,
                home_team_id=home_id,
                away_team_id=away_id,
                home_score=home_score,
                away_score=away_score,
                season=season,
                is_neutral=is_neutral,
            )
        )

    # Deduplicate (month pages shouldn't overlap, but be safe)
    uniq: dict[str, Game] = {}
    for g in games:
        key = f"{g.date}-{g.away_team_id}@{g.home_team_id}"
        uniq[key] = g
    games = sorted(uniq.values(), key=lambda g: (g.date, g.id))

    if len(games) < 100:
        raise RuntimeError(f"Too few BRef games for {season}: {len(games)}")

    return SeasonSchedule(
        sport="nba",
        season=season,
        teams=sorted(teams.values(), key=lambda t: t.name),
        games=games,
    )
