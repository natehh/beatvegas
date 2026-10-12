"""Settle collected prop quotes against free ESPN box-score actuals."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from core.injuries import normalize_player_name
from sports.nba.prop_odds import DEFAULT_PROPS_DIR, load_quotes, props_dir
from sports.nba.teams import TEAM_NAME_TO_ABBR, normalize_abbr

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

MARKET_TO_STAT = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_threes": "threes",
    "player_blocks": "blocks",
    "player_steals": "steals",
    "player_points_rebounds_assists": "pra",
    "player_points_rebounds": "pr",
    "player_points_assists": "pa",
}


@dataclass
class SettledProp:
    event_id: str
    commence_time: str
    home_abbr: str
    away_abbr: str
    bookmaker: str
    market: str
    player: str
    line: float
    over_odds: Optional[float]
    under_odds: Optional[float]
    snapshot_at: str
    actual: float
    over_won: Optional[bool]
    under_won: Optional[bool]
    settled_at: str
    source_outcome: str = "espn"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def settled_path(root: Optional[Path] = None) -> Path:
    return props_dir(root) / "prop_settled.jsonl"


def _team_abbr_from_espn(team_obj: dict) -> str:
    name = str(team_obj.get("displayName") or team_obj.get("name") or "").strip().lower()
    if name in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[name]
    abbr = str(team_obj.get("abbreviation") or "").strip().upper()
    return normalize_abbr(abbr) if abbr else ""


def _parse_stat_num(val: Any) -> float:
    if val is None:
        return 0.0
    text = str(val).strip()
    if text in {"", "--", "N/A"}:
        return 0.0
    if "-" in text and text[0].isdigit():
        # made-attempted like "3-7"
        try:
            return float(text.split("-", 1)[0])
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def fetch_espn_game_ids_for_date(game_date: date, *, timeout: float = 30.0) -> list[str]:
    stamp = game_date.strftime("%Y%m%d")
    resp = requests.get(ESPN_SCOREBOARD, params={"dates": stamp}, timeout=timeout)
    resp.raise_for_status()
    events = resp.json().get("events") or []
    return [str(e["id"]) for e in events if e.get("id")]


def load_espn_box(
    event_id: str, *, timeout: float = 45.0
) -> tuple[str, str, dict[str, dict[str, float]]]:
    """Return (home_abbr, away_abbr, player_stats)."""
    resp = requests.get(ESPN_SUMMARY, params={"event": event_id}, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    home_abbr = away_abbr = ""
    comps = (payload.get("header") or {}).get("competitions") or []
    if comps:
        for c in comps[0].get("competitors") or []:
            abbr = _team_abbr_from_espn(c.get("team") or {})
            if c.get("homeAway") == "home":
                home_abbr = abbr
            elif c.get("homeAway") == "away":
                away_abbr = abbr

    out: dict[str, dict[str, float]] = {}
    for team_block in (payload.get("boxscore") or {}).get("players") or []:
        stats_groups = team_block.get("statistics") or []
        if not stats_groups:
            continue
        group = stats_groups[0]
        labels = [str(x).lower() for x in (group.get("labels") or [])]
        idx = {lab: i for i, lab in enumerate(labels)}
        for athlete in group.get("athletes") or []:
            if athlete.get("didNotPlay"):
                continue
            a = athlete.get("athlete") or {}
            name = str(a.get("displayName") or a.get("fullName") or "").strip()
            if not name:
                continue
            stats = athlete.get("stats") or []

            def _get(*keys: str) -> float:
                for k in keys:
                    if k in idx and idx[k] < len(stats):
                        return _parse_stat_num(stats[idx[k]])
                return 0.0

            pts = _get("pts", "points")
            reb = _get("reb", "rebounds")
            ast = _get("ast", "assists")
            threes = _get("3pt")
            blk = _get("blk", "blocks")
            stl = _get("stl", "steals")
            out[normalize_player_name(name)] = {
                "points": pts,
                "rebounds": reb,
                "assists": ast,
                "threes": threes,
                "blocks": blk,
                "steals": stl,
                "pra": pts + reb + ast,
                "pr": pts + reb,
                "pa": pts + ast,
            }
    return home_abbr, away_abbr, out


def _game_date_from_commence(commence_time: str) -> date:
    tip = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    # Approximate US/Eastern calendar date for NBA tipoffs.
    return (tip - timedelta(hours=5)).date()


def _grade(actual: float, line: float) -> tuple[Optional[bool], Optional[bool]]:
    diff = actual - line
    if abs(diff) < 1e-9:
        return None, None
    over_won = diff > 0
    return over_won, (not over_won)


def latest_quotes_before_tip(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last snapshot per (event, book, market, player, line) with snapshot <= commence."""
    best: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        commence = row.get("commence_time") or ""
        snap = row.get("snapshot_at") or ""
        if commence and snap > commence:
            continue
        key = (
            row.get("event_id"),
            row.get("bookmaker"),
            row.get("market"),
            normalize_player_name(str(row.get("player", ""))),
            float(row.get("line")),
        )
        prev = best.get(key)
        if prev is None or snap >= (prev.get("snapshot_at") or ""):
            best[key] = row
    return list(best.values())


def settle_prop_quotes(
    *,
    root: Optional[Path] = None,
    only_finished_hours: float = 3.0,
    force: bool = False,
) -> dict[str, Any]:
    """
    Settle ledger quotes for games that tipped off at least `only_finished_hours` ago.
    Appends to prop_settled.jsonl; skips already-settled keys unless force=True.
    """
    root = root or DEFAULT_PROPS_DIR
    quotes = load_quotes(props_dir(root) / "prop_quotes.jsonl")
    if not quotes:
        return {"n_settled": 0, "n_skipped": 0, "message": "no quotes in ledger"}

    settled_file = settled_path(root)
    already: set[tuple] = set()
    if settled_file.exists() and not force:
        with settled_file.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                already.add(
                    (
                        d.get("event_id"),
                        d.get("bookmaker"),
                        d.get("market"),
                        normalize_player_name(str(d.get("player", ""))),
                        float(d.get("line")),
                        d.get("snapshot_at"),
                    )
                )

    now = datetime.now(timezone.utc)
    candidates = latest_quotes_before_tip(quotes)
    by_date: dict[date, list[dict[str, Any]]] = {}
    for row in candidates:
        commence = row.get("commence_time") or ""
        if not commence:
            continue
        tip = datetime.fromisoformat(commence.replace("Z", "+00:00"))
        if (now - tip).total_seconds() < only_finished_hours * 3600:
            continue
        gdate = _game_date_from_commence(commence)
        by_date.setdefault(gdate, []).append(row)

    settled_at = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    n_settled = 0
    n_skipped = 0
    n_missing_player = 0

    with settled_file.open("a", encoding="utf-8") as out_fh:
        for gdate, rows in sorted(by_date.items()):
            date_stats: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
            try:
                event_ids = fetch_espn_game_ids_for_date(gdate)
            except Exception:
                event_ids = []
            for eid in event_ids:
                try:
                    home_abbr, away_abbr, player_stats = load_espn_box(eid)
                except Exception:
                    continue
                if home_abbr and away_abbr and player_stats:
                    date_stats[(home_abbr, away_abbr)] = player_stats

            for row in rows:
                key = (
                    row.get("event_id"),
                    row.get("bookmaker"),
                    row.get("market"),
                    normalize_player_name(str(row.get("player", ""))),
                    float(row.get("line")),
                    row.get("snapshot_at"),
                )
                if key in already and not force:
                    n_skipped += 1
                    continue
                home = str(row.get("home_abbr") or "")
                away = str(row.get("away_abbr") or "")
                player_stats = date_stats.get((home, away))
                if not player_stats:
                    n_skipped += 1
                    continue
                pkey = normalize_player_name(str(row.get("player", "")))
                stats = player_stats.get(pkey)
                if not stats:
                    n_missing_player += 1
                    continue
                market = str(row.get("market") or "")
                stat_key = MARKET_TO_STAT.get(market)
                if not stat_key:
                    n_skipped += 1
                    continue
                actual = float(stats.get(stat_key, 0.0))
                over_won, under_won = _grade(actual, float(row["line"]))
                settled = SettledProp(
                    event_id=str(row.get("event_id")),
                    commence_time=str(row.get("commence_time")),
                    home_abbr=home,
                    away_abbr=away,
                    bookmaker=str(row.get("bookmaker")),
                    market=market,
                    player=str(row.get("player")),
                    line=float(row["line"]),
                    over_odds=row.get("over_odds"),
                    under_odds=row.get("under_odds"),
                    snapshot_at=str(row.get("snapshot_at")),
                    actual=actual,
                    over_won=over_won,
                    under_won=under_won,
                    settled_at=settled_at,
                )
                out_fh.write(json.dumps(settled.to_dict(), ensure_ascii=False) + "\n")
                already.add(key)
                n_settled += 1

    return {
        "n_settled": n_settled,
        "n_skipped": n_skipped,
        "n_missing_player": n_missing_player,
        "settled_path": str(settled_file),
        "dates": [d.isoformat() for d in sorted(by_date)],
    }
