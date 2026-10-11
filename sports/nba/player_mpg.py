"""Season per-game minutes from Basketball-Reference (prior-season warm start)."""

from __future__ import annotations

import json
import time
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from sports.nba import cache as nba_cache
from sports.nba.bref import season_end_year
from sports.nba.teams import from_bref_abbr

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; beatvegas/0.1; +https://github.com/local/beatvegas)",
    "Accept-Language": "en-US,en;q=0.9",
}


def _cache_path(cache_dir: Path, season: str) -> Path:
    return cache_dir / f"nba_{season.replace('/', '-')}_player_mpg.json"


def fetch_season_player_mpg(season: str, *, delay_s: float = 0.5) -> dict[str, dict]:
    """
    Return mapping normalized later by caller; keys are raw player names.
    Values: {player, team_id, mpg, games}.
    """
    end_year = season_end_year(season)
    url = f"https://www.basketball-reference.com/leagues/NBA_{end_year}_per_game.html"
    resp = requests.get(url, headers=HEADERS, timeout=45)
    resp.raise_for_status()
    html = resp.text.replace("<!--", "").replace("-->", "")
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise RuntimeError(f"No per-game table for {season}")
    df = tables[0]
    # Filter repeated header rows
    if "Player" not in df.columns or "MP" not in df.columns:
        raise RuntimeError(f"Unexpected per-game columns: {list(df.columns)}")
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        player = str(row.get("Player", "")).strip()
        if not player or player == "Player":
            continue
        team_raw = str(row.get("Team", row.get("Tm", ""))).strip().upper()
        if not team_raw or team_raw in {"TM", "TEAM", "2TM", "3TM"}:
            # Multi-team: keep last row seen; skip aggregate rows without team
            if team_raw in {"2TM", "3TM"}:
                continue
        try:
            mpg = float(row["MP"])
        except (TypeError, ValueError):
            continue
        try:
            games = int(float(row.get("G", 0)))
        except (TypeError, ValueError):
            games = 0
        team_id = from_bref_abbr(team_raw) if team_raw else ""
        # Prefer the row with more games if duplicates
        prev = out.get(player)
        if prev is not None and prev.get("games", 0) >= games:
            continue
        out[player] = {
            "player": player,
            "team_id": team_id,
            "mpg": mpg,
            "games": games,
        }
    if delay_s > 0:
        time.sleep(delay_s)
    if len(out) < 100:
        raise RuntimeError(f"Too few player mpg rows for {season}: {len(out)}")
    return out


def load_season_player_mpg(
    season: str,
    *,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> dict[str, dict]:
    cache_dir = cache_dir or nba_cache.DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, season)
    if path.exists() and not force_refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["players"]

    players = fetch_season_player_mpg(season)
    path.write_text(
        json.dumps(
            {
                "season": season,
                "source": "basketball-reference per-game",
                "fetched_at": datetime.utcnow().isoformat() + "Z",
                "n": len(players),
                "players": players,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return players
