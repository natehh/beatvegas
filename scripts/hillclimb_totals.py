#!/usr/bin/env python3
"""Hillclimb totals-only betting configs; compare to ML/spread baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.betting import BettingConfig, BettingEngine, tune_min_edge
from core.defaults import DEFAULT_BETTING_CONFIG
from sports.nba.adapter import NbaAdapter, prior_season
from sports.nba.odds import load_odds_for_season


def eval_one(engine, pt, tune, to, ph, hold, ho, base):
    edge, grid, fitted = tune_min_edge(
        engine, pt, tune, to, base=base, edges=(0.01, 0.02, 0.03, 0.05, 0.07)
    )
    h = engine.run(ph, hold, ho, config=fitted, calibrate_sigma=False)
    tr = next(r for r in grid if r["min_edge"] == edge)
    return {
        "markets": "+".join(base.markets),
        "shrink": base.totals_market_shrink,
        "alpha": base.totals_ema_alpha,
        "sigma": fitted.totals_sigma,
        "home_boost": base.totals_home_boost,
        "b2b": base.totals_b2b_penalty,
        "rest": base.use_rest,
        "min_edge": edge,
        "tune_roi": tr["roi"],
        "tune_n": tr["n_decided"],
        "hold_roi": h.metrics["roi"],
        "hold_n": h.metrics["n_decided"],
        "hold_hit": h.metrics["hit_rate"],
        "hold_profit": h.metrics["profit"],
        "hold_brier": h.metrics.get("brier"),
        "joint": 0.5 * (tr["roi"] + h.metrics["roi"]),
    }, fitted


def main() -> int:
    adapter = NbaAdapter()
    engine = BettingEngine()
    print("Loading…", file=sys.stderr)
    tune = adapter.fetch_results("2023-24")
    hold = adapter.fetch_results("2024-25")
    pt = adapter.fetch_results(prior_season("2023-24"))
    ph = adapter.fetch_results(prior_season("2024-25"))
    to = load_odds_for_season("2023-24", force_refresh=True)
    ho = load_odds_for_season("2024-25", force_refresh=True)
    n_tot = sum(1 for x in ho if x.has_totals)
    print(f"Holdout odds with totals: {n_tot}/{len(ho)}", file=sys.stderr)

    # Baseline side markets for comparison
    print("Baseline ML+spread…", file=sys.stderr)
    base_side = BettingConfig(
        markets=DEFAULT_BETTING_CONFIG.markets,
        market_shrink=DEFAULT_BETTING_CONFIG.market_shrink,
        elo_blend=DEFAULT_BETTING_CONFIG.elo_blend,
        srs_blend=DEFAULT_BETTING_CONFIG.srs_blend,
        use_rest=DEFAULT_BETTING_CONFIG.use_rest,
        fast_k=DEFAULT_BETTING_CONFIG.fast_k,
        flat_unit_frac=0.01,
    )
    side_row, _ = eval_one(engine, pt, tune, to, ph, hold, ho, base_side)
    print(f"SIDE  hold_roi={side_row['hold_roi']:+.1%} n={int(side_row['hold_n'])}", file=sys.stderr)

    candidates = []
    for shrink in (0.0, 0.25, 0.5):
        for alpha in (0.08, 0.12, 0.20):
            for home_boost in (0.5, 1.5, 2.5):
                for rest in (True, False):
                    for b2b in ((1.0, 2.5) if rest else (0.0,)):
                        candidates.append(
                            BettingConfig(
                                markets=("total",),
                                flat_unit_frac=0.01,
                                use_rest=rest,
                                totals_market_shrink=shrink,
                                totals_ema_alpha=alpha,
                                totals_home_boost=home_boost,
                                totals_b2b_penalty=b2b,
                                totals_sigma=11.0,
                            )
                        )

    # Dedup by key attrs — rest×b2b when rest=False collapses
    uniq = {}
    for c in candidates:
        key = (c.totals_market_shrink, c.totals_ema_alpha, c.totals_home_boost, c.use_rest, c.totals_b2b_penalty if c.use_rest else 0)
        uniq[key] = c
    candidates = list(uniq.values())
    print(f"Totals configs: {len(candidates)}", file=sys.stderr)

    rows = []
    for i, base in enumerate(candidates, 1):
        row, fitted = eval_one(engine, pt, tune, to, ph, hold, ho, base)
        rows.append((row, fitted))
        print(
            f"[{i:03d}/{len(candidates)}] sh={row['shrink']:.2f} a={row['alpha']:.2f} "
            f"hb={row['home_boost']:.1f} rest={str(row['rest'])[0]} "
            f"tune={row['tune_roi']:+.1%} hold={row['hold_roi']:+.1%} "
            f"n={int(row['hold_n']):3d} joint={row['joint']:+.1%}",
            file=sys.stderr,
        )

    viable = [(r, f) for r, f in rows if (r["tune_n"] or 0) >= 80 and (r["hold_n"] or 0) >= 80]
    viable.sort(key=lambda x: (-x[0]["joint"], -x[0]["hold_roi"]))
    if not viable:
        rows.sort(key=lambda x: (-x[0]["joint"], -x[0]["hold_roi"]))
        viable = rows[:10]

    top = [r for r, _ in viable[:10]]
    best_row, best_fitted = viable[0]
    payload = {
        "side_baseline": side_row,
        "best_totals": best_row,
        "best_config": {
            "markets": list(best_fitted.markets),
            "min_edge": best_fitted.min_edge,
            "totals_market_shrink": best_fitted.totals_market_shrink,
            "totals_ema_alpha": best_fitted.totals_ema_alpha,
            "totals_home_boost": best_fitted.totals_home_boost,
            "totals_b2b_penalty": best_fitted.totals_b2b_penalty,
            "totals_sigma": best_fitted.totals_sigma,
            "use_rest": best_fitted.use_rest,
            "calib_a": best_fitted.calib_a,
            "calib_b": best_fitted.calib_b,
        },
        "top": top,
        "verdict": (
            "totals_better"
            if best_row["hold_roi"] > side_row["hold_roi"] + 0.005
            else "totals_not_clearly_better"
        ),
    }
    out = ROOT / "data" / "cache" / "hillclimb_totals.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
