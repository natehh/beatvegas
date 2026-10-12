#!/usr/bin/env python3
"""Settle prop_quotes.jsonl against ESPN box scores into prop_settled.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.nba.prop_settle import settle_prop_quotes  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Settle collected NBA prop odds")
    p.add_argument(
        "--finished-hours",
        type=float,
        default=3.0,
        help="Only settle games that tipped off at least this many hours ago",
    )
    p.add_argument("--force", action="store_true", help="Re-settle even if already recorded")
    args = p.parse_args()

    result = settle_prop_quotes(only_finished_hours=args.finished_hours, force=args.force)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
