#!/usr/bin/env python3
"""Hillclimb pace + injury knobs against locked side-market baseline."""

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
from sports.nba.features import features_to_engine_kwargs, load_season_features
from sports.nba.odds import load_odds_for_season


def eval_config(engine, prior_tune, tune_sched, tune_odds, prior_hold, hold_sched, hold_odds, base, tune_kw, hold_kw):
    best_edge, grid, fitted = tune_min_edge(
        engine,
        prior_tune,
        tune_sched,
        tune_odds,
        base=base,
        **tune_kw,
    )
    hold = engine.run(
        prior_hold,
        hold_sched,
        hold_odds,
        config=fitted,
        calibrate_sigma=False,
        **hold_kw,
    )
    tune_row = next((r for r in grid if r["min_edge"] == best_edge), grid[0] if grid else {})
    joint = 0.5 * float(tune_row.get("roi") or 0.0) + 0.5 * float(hold.metrics["roi"])
    return {
        "markets": "+".join(base.markets),
        "use_pace": base.use_pace,
        "use_injuries": base.use_injuries,
        "injury_mpg_coeff": base.injury_mpg_coeff,
        "injury_q": base.injury_questionable_weight,
        "pace_alpha": base.pace_ema_alpha,
        "min_edge": best_edge,
        "tune_roi": tune_row.get("roi"),
        "tune_n": tune_row.get("n_decided"),
        "hold_roi": hold.metrics["roi"],
        "hold_n": hold.metrics["n_decided"],
        "hold_hit": hold.metrics["hit_rate"],
        "hold_profit": hold.metrics["profit"],
        "hold_brier": hold.metrics.get("brier"),
        "joint": round(joint, 5),
        "calib_a": fitted.calib_a,
        "calib_b": fitted.calib_b,
        "sigma": fitted.margin_sigma,
    }, fitted


def main() -> int:
    adapter = NbaAdapter()
    engine = BettingEngine()
    print("Loading schedules/odds/features…", file=sys.stderr)
    tune_sched = adapter.fetch_results("2023-24")
    hold_sched = adapter.fetch_results("2024-25")
    prior_tune = adapter.fetch_results(prior_season("2023-24"))
    prior_hold = adapter.fetch_results(prior_season("2024-25"))
    tune_odds = load_odds_for_season("2023-24")
    hold_odds = load_odds_for_season("2024-25")
    tune_feat = load_season_features("2023-24", tune_sched, prior_schedule=prior_tune)
    hold_feat = load_season_features("2024-25", hold_sched, prior_schedule=prior_hold)
    tune_kw = features_to_engine_kwargs(tune_feat)
    hold_kw = features_to_engine_kwargs(hold_feat)

    # Baseline: locked defaults without new features
    baseline = BettingConfig(
        markets=DEFAULT_BETTING_CONFIG.markets,
        market_shrink=DEFAULT_BETTING_CONFIG.market_shrink,
        elo_blend=DEFAULT_BETTING_CONFIG.elo_blend,
        srs_blend=DEFAULT_BETTING_CONFIG.srs_blend,
        use_rest=DEFAULT_BETTING_CONFIG.use_rest,
        fast_k=DEFAULT_BETTING_CONFIG.fast_k,
        slow_k=DEFAULT_BETTING_CONFIG.slow_k,
        b2b_penalty=DEFAULT_BETTING_CONFIG.b2b_penalty,
        flat_unit_frac=DEFAULT_BETTING_CONFIG.flat_unit_frac,
        use_pace=False,
        use_injuries=False,
    )

    candidates = [baseline]
    for use_pace in (True, False):
        for use_inj in (True, False):
            if not use_pace and not use_inj:
                continue  # already have baseline
            for coeff in (2.0, 4.0, 6.0) if use_inj else (4.0,):
                for q in (0.0, 0.25) if use_inj else (0.25,):
                    for alpha in (0.10, 0.15, 0.25) if use_pace else (0.15,):
                        candidates.append(
                            BettingConfig(
                                markets=DEFAULT_BETTING_CONFIG.markets,
                                market_shrink=DEFAULT_BETTING_CONFIG.market_shrink,
                                elo_blend=DEFAULT_BETTING_CONFIG.elo_blend,
                                srs_blend=DEFAULT_BETTING_CONFIG.srs_blend,
                                use_rest=DEFAULT_BETTING_CONFIG.use_rest,
                                fast_k=DEFAULT_BETTING_CONFIG.fast_k,
                                slow_k=DEFAULT_BETTING_CONFIG.slow_k,
                                b2b_penalty=DEFAULT_BETTING_CONFIG.b2b_penalty,
                                flat_unit_frac=DEFAULT_BETTING_CONFIG.flat_unit_frac,
                                use_pace=use_pace,
                                use_injuries=use_inj,
                                injury_mpg_coeff=coeff,
                                injury_questionable_weight=q,
                                pace_ema_alpha=alpha,
                            )
                        )

    # Dedup by key fields
    uniq = {}
    for c in candidates:
        key = (
            c.use_pace,
            c.use_injuries,
            c.injury_mpg_coeff,
            c.injury_questionable_weight,
            c.pace_ema_alpha,
        )
        uniq[key] = c
    candidates = list(uniq.values())
    print(f"Evaluating {len(candidates)} configs", file=sys.stderr)

    rows = []
    for i, base in enumerate(candidates, 1):
        row, fitted = eval_config(
            engine,
            prior_tune,
            tune_sched,
            tune_odds,
            prior_hold,
            hold_sched,
            hold_odds,
            base,
            tune_kw,
            hold_kw,
        )
        rows.append((row, fitted))
        print(
            f"[{i:03d}/{len(candidates)}] pace={str(row['use_pace'])[0]} "
            f"inj={str(row['use_injuries'])[0]} coeff={row['injury_mpg_coeff']:.1f} "
            f"q={row['injury_q']:.2f} a={row['pace_alpha']:.2f} "
            f"hold={row['hold_roi']:+.1%} tune={row['tune_roi']:+.1%} "
            f"n={int(row['hold_n'])} joint={row['joint']:+.3f}",
            file=sys.stderr,
        )

    rows.sort(key=lambda x: x[0]["joint"], reverse=True)
    best_row, best_fitted = rows[0]
    baseline_row = next(r for r, _ in rows if not r["use_pace"] and not r["use_injuries"])
    verdict = (
        "pace_injuries_better"
        if best_row["hold_roi"] > baseline_row["hold_roi"] + 0.005
        and best_row["joint"] > baseline_row["joint"]
        else "not_clearly_better"
    )

    out = {
        "baseline": baseline_row,
        "best": best_row,
        "best_config": {
            "markets": list(best_fitted.markets),
            "min_edge": best_fitted.min_edge,
            "use_pace": best_fitted.use_pace,
            "use_injuries": best_fitted.use_injuries,
            "injury_mpg_coeff": best_fitted.injury_mpg_coeff,
            "injury_questionable_weight": best_fitted.injury_questionable_weight,
            "pace_ema_alpha": best_fitted.pace_ema_alpha,
            "calib_a": best_fitted.calib_a,
            "calib_b": best_fitted.calib_b,
            "margin_sigma": best_fitted.margin_sigma,
        },
        "top": [r for r, _ in rows[:12]],
        "verdict": verdict,
    }
    path = ROOT / "data" / "cache" / "hillclimb_pace_injuries.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
