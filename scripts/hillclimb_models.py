#!/usr/bin/env python3
"""
Iterative model hillclimb:
1) Evaluate a focused config list on tune→holdout
2) Print ranked results
3) Optionally expand around the best (second pass)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.betting import BettingConfig, BettingEngine, tune_min_edge
from sports.nba.adapter import NbaAdapter, prior_season
from sports.nba.odds import load_odds_for_season


def eval_config(engine, prior_tune, tune_sched, tune_odds, prior_hold, hold_sched, hold_odds, base):
    best_edge, grid, fitted = tune_min_edge(
        engine,
        prior_tune,
        tune_sched,
        tune_odds,
        base=base,
        edges=(0.01, 0.02, 0.03, 0.05, 0.07),
    )
    hold = engine.run(prior_hold, hold_sched, hold_odds, config=fitted, calibrate_sigma=False)
    tune_row = next((r for r in grid if r["min_edge"] == best_edge), grid[0] if grid else {})
    return {
        "markets": "+".join(base.markets),
        "shrink": base.market_shrink,
        "elo_blend": base.elo_blend,
        "rest": base.use_rest,
        "fast_k": base.fast_k,
        "b2b": base.b2b_penalty,
        "min_edge": best_edge,
        "tune_roi": tune_row.get("roi"),
        "tune_n": tune_row.get("n_decided"),
        "hold_roi": hold.metrics["roi"],
        "hold_n": hold.metrics["n_decided"],
        "hold_hit": hold.metrics["hit_rate"],
        "hold_profit": hold.metrics["profit"],
        "hold_brier": hold.metrics.get("brier"),
        "hold_ll": hold.metrics.get("log_loss"),
        "calib_a": fitted.calib_a,
        "calib_b": fitted.calib_b,
        "sigma": fitted.margin_sigma,
    }, fitted, hold


def main() -> int:
    adapter = NbaAdapter()
    engine = BettingEngine()
    print("Loading data…", file=sys.stderr)
    tune_sched = adapter.fetch_results("2023-24")
    hold_sched = adapter.fetch_results("2024-25")
    prior_tune = adapter.fetch_results(prior_season("2023-24"))
    prior_hold = adapter.fetch_results(prior_season("2024-25"))
    tune_odds = load_odds_for_season("2023-24")
    hold_odds = load_odds_for_season("2024-25")

    # Pass 1: focused hypotheses (keep small; expand in pass 2)
    candidates = []
    for markets in [("moneyline",), ("spread",), ("moneyline", "spread")]:
        for shrink in (0.0, 0.4, 0.7):
            for elo_b in (0.4, 0.7):
                for rest in (False, True):
                    candidates.append(
                        BettingConfig(
                            markets=markets,
                            market_shrink=shrink,
                            elo_blend=elo_b,
                            srs_blend=1.0 - elo_b,
                            use_rest=rest,
                            fast_k=36.0,
                            flat_unit_frac=0.01,
                        )
                    )

    # Deduplicate identical dicts
    print(f"Pass 1: {len(candidates)} configs", file=sys.stderr)
    rows = []
    for i, base in enumerate(candidates, 1):
        row, fitted, hold = eval_config(
            engine, prior_tune, tune_sched, tune_odds, prior_hold, hold_sched, hold_odds, base
        )
        rows.append((row, fitted))
        print(
            f"[p1 {i:03d}/{len(candidates)}] {row['markets']:16} sh={row['shrink']:.2f} "
            f"elo={row['elo_blend']:.2f} rest={str(row['rest'])[0]} "
            f"edge={row['min_edge']:.2f} hold_roi={row['hold_roi']:+.1%} "
            f"n={int(row['hold_n']):3d} brier={row['hold_brier']}",
            file=sys.stderr,
        )

    def rank_key(item):
        r = item[0]
        # Prefer positive/less-negative ROI with enough bets; break ties with brier
        n = r["hold_n"] or 0
        roi = r["hold_roi"] or -1
        brier = r["hold_brier"] or 1
        penalty = 0 if n >= 50 else 0.05
        return (-(roi - penalty), brier)

    rows.sort(key=rank_key)
    best_row, best_fitted = rows[0]

    # Pass 2: refine around best (fast_k, b2b, shrink neighbors)
    print("\nPass 2: refine around best…", file=sys.stderr)
    refine = []
    for fast_k in (20.0, 36.0, 48.0):
        for b2b in (1.0, 2.2, 3.5):
            for shrink in sorted({best_row["shrink"], max(0.0, best_row["shrink"] - 0.1), min(0.9, best_row["shrink"] + 0.1)}):
                refine.append(
                    BettingConfig(
                        markets=tuple(best_row["markets"].split("+")),
                        market_shrink=shrink,
                        elo_blend=best_row["elo_blend"],
                        srs_blend=1.0 - best_row["elo_blend"],
                        use_rest=best_row["rest"],
                        fast_k=fast_k,
                        b2b_penalty=b2b,
                        flat_unit_frac=0.01,
                    )
                )

    for i, base in enumerate(refine, 1):
        row, fitted, hold = eval_config(
            engine, prior_tune, tune_sched, tune_odds, prior_hold, hold_sched, hold_odds, base
        )
        rows.append((row, fitted))
        print(
            f"[p2 {i:02d}/{len(refine)}] sh={row['shrink']:.2f} k={row['fast_k']:.0f} "
            f"b2b={row['b2b']:.1f} hold_roi={row['hold_roi']:+.1%} n={int(row['hold_n']):3d}",
            file=sys.stderr,
        )

    rows.sort(key=rank_key)
    print("\n=== TOP 8 ===", file=sys.stderr)
    top = [r for r, _ in rows[:8]]
    for r in top:
        print(json.dumps(r), file=sys.stderr)

    best_row, best_fitted = rows[0]
    payload = {
        "best": best_row,
        "best_config": {
            "markets": list(best_fitted.markets),
            "min_edge": best_fitted.min_edge,
            "market_shrink": best_fitted.market_shrink,
            "elo_blend": best_fitted.elo_blend,
            "srs_blend": best_fitted.srs_blend,
            "use_rest": best_fitted.use_rest,
            "fast_k": best_fitted.fast_k,
            "b2b_penalty": best_fitted.b2b_penalty,
            "calib_a": best_fitted.calib_a,
            "calib_b": best_fitted.calib_b,
            "margin_sigma": best_fitted.margin_sigma,
        },
        "top": top,
    }
    out = ROOT / "data" / "cache" / "hillclimb_results.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
