#!/usr/bin/env python3
"""Grade the v2 prop model against forward-collected Odds API lines.

Two modes:

* ``--picks`` (pregame): project every player-market in the forward
  ledger (default: the latest game date in ``data/props/prop_quotes.jsonl``)
  for games that have not tipped yet, and write a picks CSV. Committing
  that CSV before tip-off is a genuine pre-registration: git timestamps
  the model's side, line and EV before any result exists.
* default (postgame): join the model projections to *settled* forward
  props (``data/props/prop_settled.jsonl``, produced free by
  ``scripts/settle_prop_odds.py`` / the daily pipeline) and report
  model-vs-line MAE and ROI exactly like the DunkAPI backtest, so the
  two line sources grade identically.

Projections for a future game need a synthetic next-game row per
player: strictly-pregame features are shifted, so appending a
zero-stat row dated on game night and running the normal feature
pipeline yields that player's pregame state. Rotation-absence and
own-missed-games for the synthetic row are overridden with the
player's last actual game's values (the synthetic row's one-player
"lineup" would otherwise look like the whole rotation is out).

Pre-registered hypothesis for the forward sample (locked 2026-10-24,
before the first forward slate settles): assists graded with the
EV>=0 rule is the only market allowed to be called promising, based
on it being the sole positive market across both DunkAPI windows
combined (n=48, +13.0%, MAE tied with the close). Everything else is
exploratory until the forward sample says otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import unicodedata
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUOTES = ROOT / "data" / "props" / "prop_quotes.jsonl"
DEFAULT_SETTLED = ROOT / "data" / "props" / "prop_settled.jsonl"
DEFAULT_GAMELOG_CACHE = ROOT / "data" / "cache" / "props_logs"
MARKET_TO_STAT = {
    "player_points": "pts",
    "player_rebounds": "reb",
    "player_assists": "ast",
}


def _norm(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    return " ".join(name.lower().replace(".", " ").split())


def consensus_forward(quotes: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """One row per event-player-market at the median quoted line."""
    df = pd.DataFrame(quotes)
    df = df[df["market"].isin(MARKET_TO_STAT)].copy()
    df["player_norm"] = df["player"].map(_norm)
    out = []
    for (event_id, pnorm, market), grp in df.groupby(["event_id", "player_norm", "market"]):
        lines = [float(x) for x in grp["line"]]
        median_line = float(np.median(lines))
        rep = grp.assign(_d=(grp["line"].astype(float) - median_line).abs())
        rep = rep.sort_values(["_d", "bookmaker"]).iloc[0]
        out.append({
            "event_id": str(event_id),
            "commence_time": rep["commence_time"],
            "home_abbr": rep["home_abbr"],
            "away_abbr": rep["away_abbr"],
            "player": rep["player"],
            "player_norm": pnorm,
            "market": market,
            "stat": MARKET_TO_STAT[market],
            "line": median_line,
            "over_odds": float(rep["over_odds"]) if rep["over_odds"] is not None else -110.0,
            "under_odds": float(rep["under_odds"]) if rep["under_odds"] is not None else -110.0,
            "bookmaker": rep["bookmaker"],
            "n_books": int(grp["bookmaker"].nunique()),
        })
    return pd.DataFrame(out)


def _load_research():
    spec = importlib.util.spec_from_file_location(
        "player_props_research", ROOT / "scripts" / "player_props_research.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def build_forward_projections(
    consensus: pd.DataFrame, gamelog_cache: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Project consensus props with the v2 model via synthetic rows."""
    mod = _load_research()
    mod._cache_dir = gamelog_cache
    gamelog_cache.mkdir(parents=True, exist_ok=True)
    raw = mod.build_logs()
    raw = raw.copy()
    raw["player_norm"] = raw["player"].map(_norm)
    latest = raw.sort_values("date").groupby("player_norm").tail(1)
    latest_by_norm = {r.player_norm: r for r in latest.itertuples()}

    # Team games (dates) per team, for own-missed-games override.
    team_game_dates: dict[str, list[str]] = {}
    for team, grp in raw.groupby("team"):
        team_game_dates[team] = sorted(grp["date"].unique())

    synth_rows = []
    skipped: dict[str, int] = {"player_not_in_logs": 0, "team_mismatch": 0}
    seen: set[tuple[str, str]] = set()
    for r in consensus.itertuples():
        if (r.player_norm, r.event_id) in seen:
            continue
        seen.add((r.player_norm, r.event_id))
        last = latest_by_norm.get(r.player_norm)
        if last is None:
            skipped["player_not_in_logs"] += 1
            continue
        if last.team == r.home_abbr:
            opp = r.away_abbr
        elif last.team == r.away_abbr:
            opp = r.home_abbr
        else:
            skipped["team_mismatch"] += 1
            continue
        game_date = str(r.commence_time)[:10]
        missed = sum(1 for d in team_game_dates.get(last.team, []) if last.date < d < game_date)
        synth_rows.append({
            "player_id": last.player_id, "player": last.player,
            "season": "2026-27", "event_id": f"fwd-{r.event_id}-{last.player_id}",
            "date": game_date, "team": last.team, "opp": opp,
            "game_total": 230.0, "min": 0.0, "pts": 0.0, "reb": 0.0,
            "ast": 0.0, "fg3m": 0.0, "stl": 0.0, "blk": 0.0,
            "fga": 0.0, "fg3a": 0.0, "fta": 0.0, "tov": 0.0,
            "_synthetic": True, "_fwd_event": r.event_id,
            "_own_missed_override": missed,
        })
    if not synth_rows:
        return pd.DataFrame(), {"skipped": skipped, "n_synthetic": 0}
    raw["_synthetic"] = False
    combined = pd.concat([raw.drop(columns=["player_norm"]), pd.DataFrame(synth_rows)],
                         ignore_index=True)
    featured = mod.add_pregame_features(combined)
    # Fix synthetic context fields before projecting (see module docstring).
    last_feat = featured[~featured["_synthetic"].fillna(False)].sort_values("date")
    last_missing = last_feat.groupby("player_id")["missing_share5"].last()
    synth_mask = featured["_synthetic"].fillna(False)
    featured.loc[synth_mask, "missing_share5"] = featured.loc[synth_mask, "player_id"].map(last_missing).fillna(0.0)
    featured.loc[synth_mask, "own_missed_games"] = featured.loc[synth_mask, "_own_missed_override"].fillna(0)
    projected = mod.add_ml_projections(mod.add_projections(featured))
    synth = projected[projected["_synthetic"].fillna(False)].copy()

    tune = projected[(projected["season"] == "2024-25") & (projected["games_prior"] >= 10)]
    sd = {c: float((tune[c] - tune[f"{c}_proj"]).std()) for c in ["pts", "reb", "ast", "pra", "fg3m"]}
    synth["player_norm"] = synth["player"].map(_norm)
    merged = consensus.merge(
        synth[["player_norm", "_fwd_event", "games_prior", "min_proj"]
              + [f"{c}_proj" for c in ["pts", "reb", "ast", "pra"]]],
        left_on=["player_norm", "event_id"],
        right_on=["player_norm", "_fwd_event"], how="inner",
    )
    merged["proj"] = merged.apply(lambda r: r[f"{r['stat']}_proj"], axis=1)
    merged["edge"] = merged["proj"] - merged["line"]
    merged["resid_sd"] = merged["stat"].map(sd)
    info = {"skipped": skipped, "n_synthetic": int(len(synth)),
            "n_matched": int(len(merged)), "resid_sd": sd}
    return merged, info


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def add_ev(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def row_ev(r) -> tuple[str, float]:
        if not r.resid_sd or pd.isna(r.resid_sd):
            return "", float("nan")
        p = _norm_cdf((r.proj - r.line) / r.resid_sd)
        def profit(odds):
            return 100.0 * (100.0 / abs(odds) if odds < 0 else odds / 100.0)
        ev_o = p * profit(r.over_odds) - (1 - p) * 100.0
        ev_u = (1 - p) * profit(r.under_odds) - p * 100.0
        return ("over", ev_o) if ev_o >= ev_u else ("under", ev_u)
    evs = [row_ev(r) for r in df.itertuples()]
    df["side"] = [e[0] for e in evs]
    df["ev_per_100"] = [e[1] for e in evs]
    return df


def grade_settled(picks: pd.DataFrame, settled_path: Path) -> dict[str, Any]:
    """Join pregame projections to settled forward actuals and score."""
    if not settled_path.exists():
        return {"n_settled": 0, "message": f"no settled ledger at {settled_path}"}
    settled = [json.loads(x) for x in settled_path.read_text().splitlines() if x.strip()]
    sdf = pd.DataFrame(settled)
    sdf = sdf[sdf["market"].isin(MARKET_TO_STAT)].copy()
    sdf["player_norm"] = sdf["player"].map(_norm)
    # Settled rows are per book; take each book's actual (they agree) once.
    cons = sdf.groupby(["event_id", "player_norm", "market"], as_index=False).agg(
        actual=("actual", "first"))
    merged = picks.drop(columns=["actual"], errors="ignore").merge(cons, left_on=["event_id", "player_norm", "market"],
                         right_on=["event_id", "player_norm", "market"], how="inner")
    out: dict[str, Any] = {"n_settled_matched": int(len(merged)), "by_stat": {}}
    for stat, grp in merged.groupby("stat"):
        bets = []
        for r in grp.itertuples():
            diff = r.actual - r.line
            if diff == 0:
                bets.append(0.0)
                continue
            side = "over" if r.edge > 0 else "under"
            odds = r.over_odds if side == "over" else r.under_odds
            win = diff > 0 if side == "over" else diff < 0
            bets.append(100.0 * (100.0 / abs(odds) if odds < 0 else odds / 100.0) if win else -100.0)
        out["by_stat"][stat] = {
            "n": int(len(grp)),
            "model_mae": float((grp["actual"] - grp["proj"]).abs().mean()),
            "line_mae": float((grp["actual"] - grp["line"]).abs().mean()),
            "roi_point_edge": float(sum(bets) / (100 * len(bets))) if bets else None,
        }
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quotes", type=Path, default=DEFAULT_QUOTES)
    ap.add_argument("--settled", type=Path, default=DEFAULT_SETTLED)
    ap.add_argument("--gamelog-cache", type=Path, default=DEFAULT_GAMELOG_CACHE)
    ap.add_argument("--picks", type=Path, default=None,
                    help="Write pregame picks CSV here (pre-registration)")
    ap.add_argument("--grade", action="store_true",
                    help="Also grade against the settled ledger")
    args = ap.parse_args(argv)

    quotes = [json.loads(x) for x in args.quotes.read_text().splitlines() if x.strip()]
    consensus = consensus_forward(quotes)
    print(f"consensus forward props: {len(consensus)} "
          f"({consensus['event_id'].nunique()} events)")
    merged, info = build_forward_projections(consensus, args.gamelog_cache)
    print(f"synthetic players: {info['n_synthetic']}, matched props: {info['n_matched']}, "
          f"skipped: {info['skipped']}")
    graded_pool = merged[(merged["games_prior"] >= 10) & (merged["min_proj"] >= 20)]
    print(f"prop-relevant (games_prior>=10, min_proj>=20): {len(graded_pool)}")
    merged = add_ev(merged)
    if args.picks:
        cols = ["commence_time", "event_id", "home_abbr", "away_abbr", "player",
                "market", "stat", "line", "bookmaker", "n_books", "proj", "edge",
                "side", "ev_per_100", "games_prior", "min_proj"]
        merged.sort_values(["commence_time", "player", "market"])[cols].to_csv(args.picks, index=False)
        print(f"wrote pregame picks: {args.picks} ({len(merged)} rows)")
        ev_pos = merged[merged["ev_per_100"] >= 0]
        print(f"EV>=0 picks: {len(ev_pos)}; by market: "
              f"{ev_pos.groupby('market').size().to_dict()}")
    if args.grade:
        result = grade_settled(merged, args.settled)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
