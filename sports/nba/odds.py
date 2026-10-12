"""Free historical NBA odds loader (Yahoo scrapes via csdurfee/scrape_yahoo_odds)."""

from __future__ import annotations

import json
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from core.betting import OddsLine
from sports.nba import cache as nba_cache
from sports.nba.teams import normalize_abbr

YAHOO_ODDS_BASE = (
    "https://raw.githubusercontent.com/csdurfee/scrape_yahoo_odds/main/yahoo_scrapes"
)

# Yahoo short names → our abbreviations
YAHOO_TEAM_TO_ABBR: dict[str, str] = {
    "Atlanta": "ATL",
    "Boston": "BOS",
    "Brooklyn": "BKN",
    "Charlotte": "CHA",
    "Chicago": "CHI",
    "Cleveland": "CLE",
    "Dallas": "DAL",
    "Denver": "DEN",
    "Detroit": "DET",
    "Golden State": "GSW",
    "Houston": "HOU",
    "Indiana": "IND",
    "LA Clippers": "LAC",
    "LA Lakers": "LAL",
    "Memphis": "MEM",
    "Miami": "MIA",
    "Milwaukee": "MIL",
    "Minnesota": "MIN",
    "New Orleans": "NOP",
    "New York": "NYK",
    "Oklahoma City": "OKC",
    "Orlando": "ORL",
    "Philadelphia": "PHI",
    "Phoenix": "PHX",
    "Portland": "POR",
    "Sacramento": "SAC",
    "San Antonio": "SAS",
    "Toronto": "TOR",
    "Utah": "UTA",
    "Washington": "WAS",
}

# Folder year on GitHub → NBA season label (end-year folders cover that season tipoff year)
FOLDER_TO_SEASON = {
    "2021": "2021-22",
    "2022": "2022-23",  # note: file starts mid prior season; we date-filter
    "2023": "2023-24",
    "2024": "2024-25",
}

SEASON_DATE_RANGE = {
    "2021-22": (date(2021, 10, 19), date(2022, 4, 10)),
    "2022-23": (date(2022, 10, 18), date(2023, 4, 9)),
    "2023-24": (date(2023, 10, 24), date(2024, 4, 14)),
    "2024-25": (date(2024, 10, 22), date(2025, 4, 13)),
}


def _cache_path(cache_dir: Path, season: str) -> Path:
    # v2 includes totals (over/under) fields
    return cache_dir / f"nba_{season.replace('/', '-')}_odds_v2.json"


def yahoo_name_to_abbr(name: str) -> str:
    name = name.strip()
    if name in YAHOO_TEAM_TO_ABBR:
        return YAHOO_TEAM_TO_ABBR[name]
    # try normalize
    return normalize_abbr(name[:3].upper())


def _parse_game_date(value: str) -> date:
    # formats like 2024-10-22-10:00
    text = str(value).strip()
    return datetime.strptime(text[:10], "%Y-%m-%d").date()


def _to_float(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def fetch_yahoo_odds_folder(folder_year: str, timeout: float = 60.0) -> pd.DataFrame:
    url = f"{YAHOO_ODDS_BASE}/{folder_year}/odds.csv"
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "beatvegas/0.1"})
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))


def load_odds_for_season(
    season: str,
    *,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
    regular_season_only: bool = True,
) -> list[OddsLine]:
    cache_dir = cache_dir or nba_cache.DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, season)
    if path.exists() and not force_refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [_line_from_dict(d) for d in payload["lines"]]

    # Map season → folder(s). Prefer the tipoff-year folder.
    start_year = season.split("-")[0]
    folders = [start_year]
    # 2022-23 may also appear in 2022 folder partially; also try adjacent
    if season == "2022-23":
        folders = ["2022"]

    frames = []
    errors = []
    for folder in folders:
        try:
            frames.append(fetch_yahoo_odds_folder(folder))
        except Exception as exc:
            errors.append(f"{folder}: {exc}")
    if not frames:
        raise RuntimeError(f"Could not load odds for {season}: {errors}")

    df = pd.concat(frames, ignore_index=True)
    start, end = SEASON_DATE_RANGE.get(season, (None, None))
    lines: list[OddsLine] = []
    for _, row in df.iterrows():
        try:
            gdate = _parse_game_date(row["game_date"])
        except Exception:
            continue
        if regular_season_only and start and end and not (start <= gdate <= end):
            continue

        home = yahoo_name_to_abbr(str(row["home_team"]))
        away = yahoo_name_to_abbr(str(row["away_team"]))
        home_ml = _to_float(row.get("money_home_odds"))
        away_ml = _to_float(row.get("money_away_odds"))
        home_spread = _to_float(row.get("spread_home_points"))
        away_spread = _to_float(row.get("spread_away_points"))
        home_sp_odds = _to_float(row.get("spread_home_odds"))
        away_sp_odds = _to_float(row.get("spread_away_odds"))
        total_line = _to_float(row.get("total_over_points"))
        over_odds = _to_float(row.get("total_over_odds"))
        under_odds = _to_float(row.get("total_under_odds"))
        if None in (home_ml, away_ml, home_spread, away_spread, home_sp_odds, away_sp_odds):
            continue

        game_id = str(row.get("game_id", f"{gdate}-{away}@{home}"))
        lines.append(
            OddsLine(
                game_key=game_id,
                date=gdate,
                home_team_id=home,
                away_team_id=away,
                home_ml=float(home_ml),
                away_ml=float(away_ml),
                home_spread=float(home_spread),
                away_spread=float(away_spread),
                home_spread_odds=float(home_sp_odds),
                away_spread_odds=float(away_sp_odds),
                total_line=total_line,
                over_odds=over_odds,
                under_odds=under_odds,
            )
        )

    # Deduplicate by date/home/away
    uniq: dict[tuple, OddsLine] = {}
    for line in lines:
        uniq[(line.date, line.home_team_id, line.away_team_id)] = line
    lines = sorted(uniq.values(), key=lambda x: (x.date, x.game_key))

    path.write_text(
        json.dumps(
            {
                "season": season,
                "source": "csdurfee/scrape_yahoo_odds",
                "n": len(lines),
                "lines": [_line_to_dict(x) for x in lines],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return lines


def _line_to_dict(line: OddsLine) -> dict:
    return {
        "game_key": line.game_key,
        "date": line.date.isoformat(),
        "home_team_id": line.home_team_id,
        "away_team_id": line.away_team_id,
        "home_ml": line.home_ml,
        "away_ml": line.away_ml,
        "home_spread": line.home_spread,
        "away_spread": line.away_spread,
        "home_spread_odds": line.home_spread_odds,
        "away_spread_odds": line.away_spread_odds,
        "total_line": line.total_line,
        "over_odds": line.over_odds,
        "under_odds": line.under_odds,
        "home_score": line.home_score,
        "away_score": line.away_score,
    }


def _line_from_dict(d: dict) -> OddsLine:
    return OddsLine(
        game_key=d["game_key"],
        date=date.fromisoformat(d["date"]),
        home_team_id=d["home_team_id"],
        away_team_id=d["away_team_id"],
        home_ml=float(d["home_ml"]),
        away_ml=float(d["away_ml"]),
        home_spread=float(d["home_spread"]),
        away_spread=float(d["away_spread"]),
        home_spread_odds=float(d["home_spread_odds"]),
        away_spread_odds=float(d["away_spread_odds"]),
        total_line=d.get("total_line"),
        over_odds=d.get("over_odds"),
        under_odds=d.get("under_odds"),
        home_score=d.get("home_score"),
        away_score=d.get("away_score"),
    )


def match_rate(odds_lines: list[OddsLine], season_games_dates_teams: set[tuple]) -> float:
    if not season_games_dates_teams:
        return 0.0
    matched = sum(
        1
        for o in odds_lines
        if (o.date, o.home_team_id, o.away_team_id) in season_games_dates_teams
    )
    return matched / len(season_games_dates_teams)
