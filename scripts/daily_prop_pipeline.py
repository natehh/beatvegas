#!/usr/bin/env python3
"""Daily NBA player-props pipeline: settle finished props, then snapshot tonight's.

Run order matters for a clean ledger:

1. **Settle** — grade every quote in ``data/props/prop_quotes.jsonl`` whose game
   tipped off at least ``--finished-hours`` ago against ESPN box scores
   (free, no key). Idempotent: already-settled quotes are skipped, so this
   safely covers "everything that settled since the last run", i.e. last
   night's games on a daily cadence.
2. **Collect** — snapshot player props (PropLine by default; The Odds API
   with ``--source oddsapi``) for games tipping off in
   the next ``--hours-ahead`` hours (not yet live). With the default 24h
   window and a 15:00 UTC (9am MT) run, that covers today's afternoon and
   evening slate, including weekend matinees.

Used by ``.github/workflows/daily-props.yml``; also runnable by hand:

    export THE_ODDS_API_KEY=...        # or ODDS_API_KEY / oddsapi
    python scripts/daily_prop_pipeline.py                # settle + collect
    python scripts/daily_prop_pipeline.py --skip-collect # settle only (free)

Prints a JSON summary. Exit code is 1 if either stage fails, but both stages
are always attempted and any ledger writes from a successful stage are kept.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.nba.prop_odds import (  # noqa: E402
    DEFAULT_MARKETS,
    EXTENDED_MARKETS,
    collect_prop_odds,
)
from sports.nba.prop_settle import settle_prop_quotes  # noqa: E402


def run_pipeline(
    *,
    hours_ahead: float = 24.0,
    finished_hours: float = 3.0,
    markets: Optional[Sequence[str]] = None,
    regions: str = "us",
    source: str = "propline",
    max_events: Optional[int] = None,
    skip_settle: bool = False,
    skip_collect: bool = False,
) -> tuple[dict[str, Any], int]:
    """Run settle-then-collect; return (summary, exit_code).

    Default markets depend on the source: PropLine charges per request, so
    it gets the extended 7-market set; The Odds API charges per market, so
    it stays on the 3 core markets.
    """
    if markets is None:
        markets = EXTENDED_MARKETS if source == "propline" else DEFAULT_MARKETS
    summary: dict[str, Any] = {"settle": None, "collect": None}
    exit_code = 0

    if skip_settle:
        summary["settle"] = {"skipped": True}
    else:
        try:
            summary["settle"] = settle_prop_quotes(only_finished_hours=finished_hours)
        except Exception as exc:  # noqa: BLE001 - report, still try to collect
            summary["settle"] = {"error": f"{type(exc).__name__}: {exc}"}
            exit_code = 1

    if skip_collect:
        summary["collect"] = {"skipped": True}
    else:
        try:
            result = collect_prop_odds(
                markets=tuple(markets),
                regions=regions,
                max_events=max_events,
                hours_ahead=hours_ahead,
                include_live=False,
                source=source,
            )
            collect_summary = asdict(result)
            collect_summary["ledger_path"] = str(result.ledger_path)
            remaining = result.credits_remaining
            if source == "oddsapi" and remaining is not None and remaining < 50:
                collect_summary["warning"] = (
                    f"Only {remaining} Odds API credits remain; the free tier "
                    "(500/month) will not cover a full daily slate for long."
                )
            summary["collect"] = collect_summary
        except Exception as exc:  # noqa: BLE001
            summary["collect"] = {"error": f"{type(exc).__name__}: {exc}"}
            exit_code = 1

    return summary, exit_code


def main() -> int:
    p = argparse.ArgumentParser(description="Daily NBA props: settle previous, collect today")
    p.add_argument("--hours-ahead", type=float, default=24.0,
                   help="Collect games tipping within this window (default 24 = today + early tomorrow)")
    p.add_argument("--finished-hours", type=float, default=3.0,
                   help="Settle games that tipped at least this many hours ago")
    p.add_argument("--source", choices=("propline", "oddsapi"), default="propline",
                   help="Odds source (default propline; oddsapi = The Odds API)")
    p.add_argument("--markets", default=None,
                   help="Comma-separated market keys (default: extended for propline, core for oddsapi)")
    p.add_argument("--regions", default="us")
    p.add_argument("--max-events", type=int, default=None, help="Cap events (credit control)")
    p.add_argument("--skip-settle", action="store_true")
    p.add_argument("--skip-collect", action="store_true")
    args = p.parse_args()

    markets = None
    if args.markets:
        markets = tuple(m.strip() for m in args.markets.split(",") if m.strip())
    summary, exit_code = run_pipeline(
        hours_ahead=args.hours_ahead,
        finished_hours=args.finished_hours,
        markets=markets,
        regions=args.regions,
        source=args.source,
        max_events=args.max_events,
        skip_settle=args.skip_settle,
        skip_collect=args.skip_collect,
    )
    print(json.dumps(summary, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
