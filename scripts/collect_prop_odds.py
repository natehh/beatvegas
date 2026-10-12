#!/usr/bin/env python3
"""Snapshot upcoming NBA player-prop odds into data/props/prop_quotes.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.nba.prop_odds import (  # noqa: E402
    DEFAULT_MARKETS,
    EXTENDED_MARKETS,
    collect_prop_odds,
    get_api_key,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Collect NBA player prop odds (The Odds API)")
    p.add_argument("--markets", default=",".join(DEFAULT_MARKETS), help="Comma-separated market keys")
    p.add_argument("--extended", action="store_true", help="Use extended market set (more credits)")
    p.add_argument("--regions", default="us")
    p.add_argument("--hours-ahead", type=float, default=36.0)
    p.add_argument("--max-events", type=int, default=None, help="Cap events (credit control)")
    p.add_argument("--no-live", action="store_true", help="Skip games that already tipped off")
    p.add_argument("--dry-run-key-check", action="store_true", help="Only verify API key is set")
    args = p.parse_args()

    if args.dry_run_key_check:
        try:
            get_api_key()
            print("THE_ODDS_API_KEY is set.")
            return 0
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    markets = EXTENDED_MARKETS if args.extended else tuple(
        m.strip() for m in args.markets.split(",") if m.strip()
    )
    try:
        result = collect_prop_odds(
            markets=markets,
            regions=args.regions,
            max_events=args.max_events,
            hours_ahead=args.hours_ahead,
            include_live=not args.no_live,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "\nSetup: create a free key at https://the-odds-api.com/ then:\n"
            "  export THE_ODDS_API_KEY=your_key\n"
            "Free tier ≈ 500 credits/month. Default run ≈ 3 credits × N upcoming games.",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"Collect failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "snapshot_at": result.snapshot_at,
                "n_events": result.n_events,
                "n_quotes": result.n_quotes,
                "credits_remaining": result.credits_remaining,
                "credits_used": result.credits_used,
                "credits_last": result.credits_last,
                "ledger": str(result.ledger_path),
                "event_ids": result.event_ids,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
