"""Forward collector for NBA player-prop odds via The Odds API.

Requires env THE_ODDS_API_KEY (free tier works for *upcoming* event props;
historical props are paid-only). Snapshots append to a JSONL ledger under
data/props/ for later settlement and backtests.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import requests

from sports.nba.teams import TEAM_NAME_TO_ABBR, normalize_abbr

DEFAULT_PROPS_DIR = Path(__file__).resolve().parents[2] / "data" / "props"
SPORT_KEY = "basketball_nba"
API_BASE = "https://api.the-odds-api.com/v4"

# Core markets: 3 markets × 1 region ≈ 3 credits per event on free tier.
DEFAULT_MARKETS = (
    "player_points",
    "player_rebounds",
    "player_assists",
)

# Optional extras (cost more credits).
EXTENDED_MARKETS = DEFAULT_MARKETS + (
    "player_threes",
    "player_points_rebounds_assists",
    "player_blocks",
    "player_steals",
)


@dataclass
class PropQuote:
    snapshot_at: str
    event_id: str
    commence_time: str
    home_team: str
    away_team: str
    home_abbr: str
    away_abbr: str
    bookmaker: str
    market: str
    player: str
    line: float
    over_odds: Optional[float]
    under_odds: Optional[float]
    season: str = ""
    source: str = "the-odds-api"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CollectResult:
    snapshot_at: str
    n_events: int
    n_quotes: int
    credits_remaining: Optional[int]
    credits_used: Optional[int]
    credits_last: Optional[int]
    ledger_path: Path
    event_ids: list[str] = field(default_factory=list)


def props_dir(root: Optional[Path] = None) -> Path:
    d = root or DEFAULT_PROPS_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / "snapshots").mkdir(parents=True, exist_ok=True)
    return d


def ledger_path(root: Optional[Path] = None) -> Path:
    return props_dir(root) / "prop_quotes.jsonl"


def get_api_key(explicit: Optional[str] = None) -> str:
    key = (explicit or os.environ.get("THE_ODDS_API_KEY") or os.environ.get("ODDS_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "Missing THE_ODDS_API_KEY. Get a free key at https://the-odds-api.com/ and export it."
        )
    return key


def season_from_commence(commence_time: str) -> str:
    """Map tipoff UTC ISO to NBA season label (Oct+ → start year)."""
    dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    year = dt.year
    if dt.month >= 8:
        return f"{year}-{str(year + 1)[-2:]}"
    return f"{year - 1}-{str(year)[-2:]}"


def team_to_abbr(name: str) -> str:
    key = name.strip().lower()
    if key in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[key]
    # Odds API sometimes uses short city names; best-effort
    for full, abbr in TEAM_NAME_TO_ABBR.items():
        if key in full or full.startswith(key):
            return abbr
    return normalize_abbr("".join(ch for ch in name if ch.isalpha())[:3].upper())


def _headers_usage(resp: requests.Response) -> dict[str, Optional[int]]:
    def _i(name: str) -> Optional[int]:
        v = resp.headers.get(name)
        try:
            return int(v) if v is not None else None
        except ValueError:
            return None

    return {
        "remaining": _i("x-requests-remaining"),
        "used": _i("x-requests-used"),
        "last": _i("x-requests-last"),
    }


def list_nba_events(
    api_key: str,
    *,
    timeout: float = 30.0,
) -> tuple[list[dict[str, Any]], dict[str, Optional[int]]]:
    url = f"{API_BASE}/sports/{SPORT_KEY}/events"
    resp = requests.get(url, params={"apiKey": api_key}, timeout=timeout)
    resp.raise_for_status()
    return resp.json(), _headers_usage(resp)


def fetch_event_props(
    api_key: str,
    event_id: str,
    *,
    markets: Sequence[str] = DEFAULT_MARKETS,
    regions: str = "us",
    timeout: float = 45.0,
) -> tuple[dict[str, Any], dict[str, Optional[int]]]:
    url = f"{API_BASE}/sports/{SPORT_KEY}/events/{event_id}/odds"
    resp = requests.get(
        url,
        params={
            "apiKey": api_key,
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": "american",
            "dateFormat": "iso",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json(), _headers_usage(resp)


def flatten_event_quotes(
    event_payload: dict[str, Any],
    *,
    snapshot_at: str,
) -> list[PropQuote]:
    event_id = str(event_payload.get("id", ""))
    commence = str(event_payload.get("commence_time", ""))
    home = str(event_payload.get("home_team", ""))
    away = str(event_payload.get("away_team", ""))
    home_abbr = team_to_abbr(home)
    away_abbr = team_to_abbr(away)
    season = season_from_commence(commence) if commence else ""
    quotes: list[PropQuote] = []

    for book in event_payload.get("bookmakers") or []:
        book_key = str(book.get("key", ""))
        for market in book.get("markets") or []:
            market_key = str(market.get("key", ""))
            # Pair Over/Under by (player, line)
            paired: dict[tuple[str, float], dict[str, Optional[float]]] = {}
            for outcome in market.get("outcomes") or []:
                name = str(outcome.get("name", "")).strip().lower()
                player = str(outcome.get("description", "")).strip()
                if not player:
                    continue
                try:
                    line = float(outcome["point"])
                except (KeyError, TypeError, ValueError):
                    continue
                try:
                    price = float(outcome["price"])
                except (KeyError, TypeError, ValueError):
                    continue
                slot = paired.setdefault((player, line), {"over": None, "under": None})
                if name == "over":
                    slot["over"] = price
                elif name == "under":
                    slot["under"] = price
            for (player, line), prices in paired.items():
                if prices["over"] is None and prices["under"] is None:
                    continue
                quotes.append(
                    PropQuote(
                        snapshot_at=snapshot_at,
                        event_id=event_id,
                        commence_time=commence,
                        home_team=home,
                        away_team=away,
                        home_abbr=home_abbr,
                        away_abbr=away_abbr,
                        bookmaker=book_key,
                        market=market_key,
                        player=player,
                        line=line,
                        over_odds=prices["over"],
                        under_odds=prices["under"],
                        season=season,
                    )
                )
    return quotes


def append_quotes(quotes: Iterable[PropQuote], path: Optional[Path] = None) -> Path:
    path = path or ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for q in quotes:
            fh.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_quotes(path: Optional[Path] = None) -> list[dict[str, Any]]:
    path = path or ledger_path()
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def collect_prop_odds(
    *,
    api_key: Optional[str] = None,
    markets: Sequence[str] = DEFAULT_MARKETS,
    regions: str = "us",
    max_events: Optional[int] = None,
    hours_ahead: float = 36.0,
    include_live: bool = True,
    root: Optional[Path] = None,
) -> CollectResult:
    """
    Snapshot upcoming (and optionally live) NBA player props into the JSONL ledger.

    Credit tip (free tier ≈ 500/month): events list is cheap; each event costs
    roughly (#markets returned) × (#regions). Default 3 markets × us ≈ 3 credits/event.
    """
    key = get_api_key(api_key)
    now = datetime.now(timezone.utc)
    snapshot_at = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    events, usage_events = list_nba_events(key)
    selected: list[dict[str, Any]] = []
    for ev in events:
        commence = ev.get("commence_time")
        if not commence:
            continue
        tip = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
        hours = (tip - now).total_seconds() / 3600.0
        if hours > hours_ahead:
            continue
        if hours < 0 and not include_live:
            continue
        # Skip games that tipped off more than 4h ago (likely finished / props stale)
        if hours < -4:
            continue
        selected.append(ev)
    selected.sort(key=lambda e: e.get("commence_time") or "")
    if max_events is not None:
        selected = selected[: max(0, int(max_events))]

    all_quotes: list[PropQuote] = []
    raw_dir = props_dir(root) / "snapshots" / snapshot_at.replace(":", "-")
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "events.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    remaining = usage_events.get("remaining")
    used = usage_events.get("used")
    last = usage_events.get("last")

    for ev in selected:
        eid = str(ev["id"])
        payload, usage = fetch_event_props(key, eid, markets=markets, regions=regions)
        remaining = usage.get("remaining", remaining)
        used = usage.get("used", used)
        last = usage.get("last", last)
        (raw_dir / f"{eid}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        all_quotes.extend(flatten_event_quotes(payload, snapshot_at=snapshot_at))

    path = append_quotes(all_quotes, ledger_path(root))
    meta = {
        "snapshot_at": snapshot_at,
        "n_events": len(selected),
        "n_quotes": len(all_quotes),
        "markets": list(markets),
        "regions": regions,
        "credits_remaining": remaining,
        "credits_used": used,
        "credits_last": last,
    }
    (raw_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return CollectResult(
        snapshot_at=snapshot_at,
        n_events=len(selected),
        n_quotes=len(all_quotes),
        credits_remaining=remaining,
        credits_used=used,
        credits_last=last,
        ledger_path=path,
        event_ids=[str(e["id"]) for e in selected],
    )
