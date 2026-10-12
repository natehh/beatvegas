"""Basketball-Reference box scores → player minutes (and optional game pace)."""

from __future__ import annotations

import json
import re
import time
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

from core.injuries import PlayerMinuteLine
from core.models import Game
from sports.nba import cache as nba_cache
from sports.nba.teams import to_bref_abbr

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; beatvegas/0.1; +https://github.com/local/beatvegas)",
    "Accept-Language": "en-US,en;q=0.9",
}


def _cache_path(cache_dir: Path, season: str) -> Path:
    return cache_dir / f"nba_{season.replace('/', '-')}_minutes.json"


def _parse_minutes(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    text = str(val).strip()
    if not text or text.lower() in {"nan", "did not play", "did not dress", "player suspended", "not with team"}:
        return 0.0
    if ":" in text:
        try:
            mm, ss = text.split(":", 1)
            return int(mm) + int(ss) / 60.0
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def boxscore_url(game: Game) -> str:
    d = game.date.strftime("%Y%m%d")
    home = to_bref_abbr(game.home_team_id)
    return f"https://www.basketball-reference.com/boxscores/{d}0{home}.html"


def fetch_boxscore_minutes(game: Game, *, timeout: float = 45.0) -> list[PlayerMinuteLine]:
    url = boxscore_url(game)
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    if resp.status_code == 404:
        # Neutral-site / rare slug miss
        return []
    resp.raise_for_status()
    html = resp.text.replace("<!--", "").replace("-->", "")
    tables = pd.read_html(StringIO(html))
    lines: list[PlayerMinuteLine] = []

    # Basic box score tables have MultiIndex cols with MP and a player name column.
    # There are usually two basic tables (away, home) before advanced duplicates.
    basic = []
    for t in tables:
        cols = t.columns
        flat = []
        if isinstance(cols, pd.MultiIndex):
            flat = [str(c[-1]) for c in cols]
            top = [str(c[0]) for c in cols]
            if "MP" in flat and any("Basic" in x for x in top):
                basic.append(t)
        else:
            flat = [str(c) for c in cols]
            if "MP" in flat and "FG" in flat:
                basic.append(t)
    # First two basic tables are away then home on BRef.
    team_order = [game.away_team_id, game.home_team_id]
    for idx, table in enumerate(basic[:2]):
        team_id = team_order[idx] if idx < 2 else team_order[-1]
        t = table.copy()
        if isinstance(t.columns, pd.MultiIndex):
            name_col = t.columns[0]
            mp_col = next(c for c in t.columns if str(c[-1]) == "MP")
        else:
            name_col = t.columns[0]
            mp_col = "MP"
        for _, row in t.iterrows():
            player = str(row[name_col]).strip()
            if not player or player.lower() in {"team totals", "reserves", "starters", "nan"}:
                continue
            if re.match(r"^\d+$", player):
                continue
            mins = _parse_minutes(row[mp_col])
            if mins is None:
                continue
            lines.append(
                PlayerMinuteLine(
                    date=game.date,
                    team_id=team_id,
                    player=player,
                    minutes=float(mins),
                )
            )
    return lines


def load_player_minutes(
    season: str,
    games: Iterable[Game],
    *,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
    delay_s: float = 0.55,
    max_games: Optional[int] = None,
) -> list[PlayerMinuteLine]:
    cache_dir = cache_dir or nba_cache.DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, season)
    if path.exists() and not force_refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [
            PlayerMinuteLine(
                date=date.fromisoformat(d["date"]),
                team_id=d["team_id"],
                player=d["player"],
                minutes=float(d["minutes"]),
            )
            for d in payload["lines"]
        ]

    games_list = sorted((g for g in games if g.is_final), key=lambda g: (g.date, g.id))
    if max_games is not None:
        games_list = games_list[:max_games]

    lines: list[PlayerMinuteLine] = []
    errors: list[str] = []
    for i, g in enumerate(games_list, 1):
        try:
            got = fetch_boxscore_minutes(g)
            if not got:
                errors.append(f"{g.date} {g.away_team_id}@{g.home_team_id}: empty")
            lines.extend(got)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{g.date} {g.away_team_id}@{g.home_team_id}: {exc}")
        if delay_s > 0:
            time.sleep(delay_s)
        if i % 50 == 0:
            # Incremental cache so long scrapes are resumable-ish
            _write_minutes_cache(path, season, lines, errors)

    _write_minutes_cache(path, season, lines, errors)
    if len(lines) < 1000 and max_games is None:
        raise RuntimeError(f"Minutes fetch too thin for {season}: {len(lines)} lines; sample errors={errors[:5]}")
    return lines


def _write_minutes_cache(path: Path, season: str, lines: list[PlayerMinuteLine], errors: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "season": season,
                "source": "basketball-reference box scores",
                "fetched_at": datetime.utcnow().isoformat() + "Z",
                "n": len(lines),
                "errors": errors[-50:],
                "lines": [
                    {
                        "date": ln.date.isoformat(),
                        "team_id": ln.team_id,
                        "player": ln.player,
                        "minutes": ln.minutes,
                    }
                    for ln in lines
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
