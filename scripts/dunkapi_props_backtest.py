#!/usr/bin/env python3
"""Historical player-props backtest: projection model vs DunkAPI closes.

Closes the loop left open by ``scripts/player_props_research.py``: that
study validated the ESPN-gamelog projection model against *simulated*
lines and a small forward-collected sample, because no free historical
prop archive was available. DunkAPI archives closing prop lines back to
May 2023, so the same strictly-pregame projections can now be graded
against real closing prices.

Credit discipline (free tier: 500 credits/month)
-------------------------------------------------
Docs headline ``get_player_props`` as 10 credits/call, but
``estimate_only`` (free) shows the player+date-range form is metered at
10 credits *per game returned* -- a full season for one star estimates
at 640-730 credits, more than a month of free tier. So this script:

* always runs ``estimate_only`` first and prints every per-player cost,
* refuses to pull if the estimates sum past ``--budget-credits``
  (default 400) unless ``--force`` is given,
* pulls by player + date range (one request per player, all markets and
  books) rather than game-by-game, and
* caches every raw response under ``data/cache/dunkapi/`` (gitignored),
  so re-analysis costs 0 credits. Delete a cache file to re-pull.

Default sample: 8 high-minute starters over the first 11 days of
2025-26 (2025-10-21..2025-10-31). Holdout-season lines only on purpose:
the edge thresholds (k * residual SD per stat) are sized on 2024-25
gamelog residuals, which are free, so no credits are spent on a tune
season. Projections still use the full 2024-25 history as prior, so
``games_prior >= 10`` holds from opening night. Player ids are ESPN
athlete ids (DunkAPI ``search_players``, 1 credit/name, already spent
for the defaults below and hard-coded to avoid re-spending).

Betting convention: one bet per player-game-market at the *consensus*
(median across books) closing line, priced at the book whose line is
closest to the median. Stake 100 units, settled at that book's actual
American odds for the chosen side. Betting every book separately would
multiply-count the same opinion.

Results (2025-10-21..2025-10-31, pulled 2026-10-24; 360 credits of
pulls + 9 of search_players; 1,345 raw rows -> 180 consensus bets over
34 games / 8 players, median 6-7 books per market)
------------------------------------------------------------------------------
The closing line beat the model on MAE for every stat (pts 7.75 vs
8.99, reb 2.47 vs 2.60, ast 1.78 vs 1.89, pra 8.76 vs 11.02, fg3m 1.11
vs 1.32), and betting the model lost at every edge threshold:

    k=0.0  n=180 hit=39.4% ROI=-28.0%
    k=0.3  n= 92 hit=39.1% ROI=-28.6%
    k=0.5  n= 52 hit=30.8% ROI=-45.1%

The only positive slice (ast, k=0.5: +9.4% on n=11) is noise-scale.
Caveats: this is the hardest window of the season -- projections lean
on prior-season rates while books already price role changes (Maxey
without Embiid/George, Wembanyama's leap, Doncic in LA all ran far
above both model and line; Jokic opened cold against his). 34 games
cannot settle the question for a full season, but it is the first
grade against real closes, and it gives no evidence of an edge:
consistent with the forward-collected diagnostic, books are not the
naive baseline the simulated-line backtest beat. Re-running from cache
(--analyze-only) costs 0 credits; a larger verdict needs either a paid
tier or a longer forward ledger.

Usage:
    export DUNKAPI_API_KEY=...
    python scripts/dunkapi_props_backtest.py --estimate-only
    python scripts/dunkapi_props_backtest.py            # pull (cached) + grade
    python scripts/dunkapi_props_backtest.py --analyze-only  # cache only, no API
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://mcp.dunkapi.com/v1"
DEFAULT_CACHE = ROOT / "data" / "cache" / "dunkapi"
DEFAULT_GAMELOG_CACHE = ROOT / "data" / "cache" / "props_logs"

# ESPN athlete ids, confirmed via DunkAPI search_players (1 credit each).
DEFAULT_PLAYERS: dict[str, str] = {
    "Nikola Jokic": "3112335",
    "Shai Gilgeous-Alexander": "4278073",
    "Giannis Antetokounmpo": "3032977",
    "Luka Doncic": "3945274",
    "Anthony Edwards": "4594268",
    "Tyrese Maxey": "4431678",
    "Jalen Brunson": "3934672",
    "Victor Wembanyama": "5104157",
}

MARKET_TO_STAT = {
    "player_points": "pts",
    "player_rebounds": "reb",
    "player_assists": "ast",
    "player_threes": "fg3m",
    "player_points_rebounds_assists": "pra",
}
STATS = ["pts", "reb", "ast", "pra", "fg3m"]


class BudgetExceeded(RuntimeError):
    pass


class DunkClient:
    def __init__(self, api_key: str, base_url: str = BASE_URL) -> None:
        if not api_key:
            raise ValueError("DUNKAPI_API_KEY is not set")
        self._key = api_key
        self.base_url = base_url

    def _post(self, tool: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}/{tool}",
            headers={"x-api-key": self._key, "Content-Type": "application/json"},
            json=body,
            timeout=120,
        )
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", payload)
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"DunkAPI error: {data['error']}")
        return data

    def estimate_player_props(self, player_id: str, from_date: str, to_date: str) -> int:
        data = self._post(
            "get_player_props",
            {
                "player_id": player_id,
                "from_date": from_date,
                "to_date": to_date,
                "snapshot_type": "closing",
                "estimate_only": True,
            },
        )
        return int(data.get("credits", 0))

    def get_player_props(self, player_id: str, from_date: str, to_date: str) -> dict[str, Any]:
        return self._post(
            "get_player_props",
            {
                "player_id": player_id,
                "from_date": from_date,
                "to_date": to_date,
                "snapshot_type": "closing",
            },
        )


def cache_path(cache_dir: Path, player_id: str, from_date: str, to_date: str) -> Path:
    return cache_dir / f"props_{player_id}_{from_date}_{to_date}_closing.json"


def fetch_props(
    players: dict[str, str],
    from_date: str,
    to_date: str,
    *,
    client: Optional[DunkClient] = None,
    cache_dir: Path = DEFAULT_CACHE,
    budget_credits: int = 400,
    estimate_only: bool = False,
    analyze_only: bool = False,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Return raw prop rows, spending credits only on uncached players.

    Estimates are always obtained (free) for uncached players before any
    paid pull, and the paid pull is aborted if the total estimate would
    exceed ``budget_credits``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    pending: list[tuple[str, str]] = []
    for name, pid in players.items():
        path = cache_path(cache_dir, pid, from_date, to_date)
        if path.exists():
            rows.extend(json.loads(path.read_text()).get("props", []))
            log(f"cache hit  {name} ({pid}): 0 credits")
        else:
            pending.append((name, pid))

    if not pending:
        return rows
    if analyze_only:
        raise FileNotFoundError(
            f"no cached DunkAPI props for {len(pending)} player(s); run without --analyze-only"
        )
    if client is None:
        raise ValueError("DUNKAPI_API_KEY is not set (needed for uncached players)")

    estimates: dict[str, int] = {}
    for name, pid in pending:
        estimates[pid] = client.estimate_player_props(pid, from_date, to_date)
        log(f"estimate   {name} ({pid}): {estimates[pid]} credits")
    total = sum(estimates.values())
    log(f"estimate total for {from_date}..{to_date}: {total} credits "
        f"(budget {budget_credits})")
    if estimate_only:
        return rows
    if total > budget_credits and not force:
        raise BudgetExceeded(
            f"estimated {total} credits exceeds budget {budget_credits}; "
            "narrow the date range / player list or pass --force"
        )

    for name, pid in pending:
        data = client.get_player_props(pid, from_date, to_date)
        cache_path(cache_dir, pid, from_date, to_date).write_text(json.dumps(data))
        props = data.get("props", [])
        log(f"pulled     {name} ({pid}): {len(props)} prop rows "
            f"({estimates[pid]} credits)")
        rows.extend(props)
    return rows


def consensus_props(props: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """One row per player-game-market at the median closing line.

    The settled price is taken from the book whose line is closest to
    the median (ties: bookmaker name), so the line and odds always come
    from a real, bettable quote.
    """
    if not props:
        return pd.DataFrame(
            columns=["player_id", "game_id", "game_date", "player", "market",
                     "line", "over_odds", "under_odds", "bookmaker", "n_books",
                     "line_min", "line_max"]
        )
    df = pd.DataFrame(props)
    df = df[df["market"].isin(MARKET_TO_STAT)].copy()
    out = []
    for (pid, gid, market), grp in df.groupby(["player_id", "game_id", "market"]):
        lines = [float(x) for x in grp["line"]]
        median_line = statistics.median(lines)
        rep = grp.assign(_dist=(grp["line"].astype(float) - median_line).abs())
        rep = rep.sort_values(["_dist", "bookmaker"]).iloc[0]
        out.append({
            "player_id": str(pid),
            "game_id": str(gid),
            "game_date": rep["game_date"],
            "player": rep["player_name"],
            "market": market,
            "stat": MARKET_TO_STAT[market],
            "line": median_line,
            "over_odds": float(rep["over_odds"]),
            "under_odds": float(rep["under_odds"]),
            "bookmaker": rep["bookmaker"],
            "n_books": int(grp["bookmaker"].nunique()),
            "line_min": min(lines),
            "line_max": max(lines),
        })
    return pd.DataFrame(out)


def settle_with_odds(actual: float, line: float, side: str, odds: float) -> float:
    """Profit on a 100-unit stake at American ``odds`` for ``side``."""
    diff = actual - line
    if diff == 0:
        return 0.0
    win = diff > 0 if side == "over" else diff < 0
    if not win:
        return -100.0
    return 100.0 * (100.0 / abs(odds) if odds < 0 else odds / 100.0)


def _norm_cdf(x: float) -> float:
    import math

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _win_profit(odds: float) -> float:
    return 100.0 * (100.0 / abs(odds) if odds < 0 else odds / 100.0)


def ev_bets(graded: pd.DataFrame, min_ev: float) -> list[float]:
    """Bets chosen by model-implied win probability vs price.

    Actual ~ Normal(proj, resid_sd) (resid_sd fit on the tune season),
    so P(over) = Phi((proj - line) / sd). A side is bet only when its
    expected profit per 100 staked is at least ``min_ev``. This replaces
    the raw point-edge rule, under which larger model-vs-line gaps
    turned out to be mostly model error.
    """
    profits: list[float] = []
    for r in graded.itertuples():
        if not r.resid_sd or r.resid_sd <= 0:
            continue
        p_over = _norm_cdf((r.proj - r.line) / r.resid_sd)
        ev_over = p_over * _win_profit(r.over_odds) - (1 - p_over) * 100.0
        ev_under = (1 - p_over) * _win_profit(r.under_odds) - p_over * 100.0
        side, ev = ("over", ev_over) if ev_over >= ev_under else ("under", ev_under)
        if ev < min_ev:
            continue
        odds = r.over_odds if side == "over" else r.under_odds
        profits.append(settle_with_odds(r.actual, r.line, side, odds))
    return profits


def _bet_summary(profits: list[float]) -> dict[str, Any]:
    return {
        "n": int(len(profits)),
        "hit_rate": (float(sum(1 for p in profits if p > 0) / len(profits))
                     if profits else None),
        "roi": (float(sum(profits) / (100 * len(profits))) if profits else None),
        "profit_units_100": float(sum(profits)),
    }


def load_projections(gamelog_cache: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build strictly-pregame projections with the research model."""
    spec = importlib.util.spec_from_file_location(
        "player_props_research", ROOT / "scripts" / "player_props_research.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod._cache_dir = gamelog_cache
    gamelog_cache.mkdir(parents=True, exist_ok=True)
    df = mod.add_ml_projections(mod.add_projections(mod.add_pregame_features(mod.build_logs())))
    tune = df[(df["season"] == "2024-25") & (df["games_prior"] >= 10)]
    sd = {c: float((tune[c] - tune[f"{c}_proj"]).std()) for c in STATS}
    return df, sd


def grade(consensus: pd.DataFrame, logs: pd.DataFrame, sd: dict[str, float]) -> pd.DataFrame:
    """Join consensus closes to actuals + pregame projections."""
    logs = logs.copy()
    logs["player_id"] = logs["player_id"].astype(str)
    logs["event_id"] = logs["event_id"].astype(str)
    keep = ["player_id", "event_id", "date", "season", "games_prior", "min_proj"] + STATS + [
        f"{c}_proj" for c in STATS
    ]
    merged = consensus.merge(
        logs[keep], left_on=["player_id", "game_id"],
        right_on=["player_id", "event_id"], how="inner",
    )
    merged["actual"] = merged.apply(lambda r: r[r["stat"]], axis=1)
    merged["proj"] = merged.apply(lambda r: r[f"{r['stat']}_proj"], axis=1)
    merged["edge"] = merged["proj"] - merged["line"]
    merged["resid_sd"] = merged["stat"].map(sd)
    return merged


def summarize(graded: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_consensus_bets_pool": int(len(graded)), "by_stat": {}}
    for stat, grp in graded.groupby("stat"):
        entry: dict[str, Any] = {
            "n": int(len(grp)),
            "model_mae": float((grp["actual"] - grp["proj"]).abs().mean()),
            "line_mae": float((grp["actual"] - grp["line"]).abs().mean()),
            "model_bias": float((grp["proj"] - grp["actual"]).mean()),
            "line_bias": float((grp["line"] - grp["actual"]).mean()),
            "bets": {},
        }
        for k in (0.0, 0.3, 0.5):
            bets = grp[grp["edge"].abs() >= k * grp["resid_sd"]]
            profits = [
                settle_with_odds(r.actual, r.line,
                                 "over" if r.edge > 0 else "under",
                                 r.over_odds if r.edge > 0 else r.under_odds)
                for r in bets.itertuples()
            ]
            entry["bets"][str(k)] = _bet_summary(profits)
        entry["ev_bets"] = {
            str(t): _bet_summary(ev_bets(grp, float(t))) for t in (0, 5, 10)
        }
        summary["by_stat"][stat] = entry
    pooled: dict[str, Any] = {}
    for k in (0.0, 0.3, 0.5):
        bets = graded[graded["edge"].abs() >= k * graded["resid_sd"]]
        profits = [
            settle_with_odds(r.actual, r.line,
                             "over" if r.edge > 0 else "under",
                             r.over_odds if r.edge > 0 else r.under_odds)
            for r in bets.itertuples()
        ]
        pooled[str(k)] = _bet_summary(profits)
    summary["pooled_bets"] = pooled
    summary["pooled_ev_bets"] = {
        str(t): _bet_summary(ev_bets(graded, float(t))) for t in (0, 5, 10)
    }
    return summary


def print_report(summary: dict[str, Any], graded: pd.DataFrame, sd: dict[str, float]) -> None:
    print("\n===== DunkAPI closing-line backtest (consensus line, real odds) =====")
    print(f"matched player-game-markets: {len(graded)} "
          f"({graded['game_id'].nunique()} games, {graded['player_id'].nunique()} players, "
          f"{graded['game_date'].min()}..{graded['game_date'].max()})")
    print(f"tune (2024-25 gamelog) residual SD: "
          + ", ".join(f"{c}={sd[c]:.2f}" for c in STATS))
    print("\naccuracy: model vs closing line (lower MAE wins)")
    for stat, e in sorted(summary["by_stat"].items()):
        print(f"  {stat:4s} n={e['n']:4d} model MAE={e['model_mae']:.2f} "
              f"line MAE={e['line_mae']:.2f} model bias={e['model_bias']:+.2f} "
              f"line bias={e['line_bias']:+.2f}")
    print("\nbetting the model vs the close (100u stakes, actual odds):")
    for stat, e in sorted(summary["by_stat"].items()):
        for k, b in e["bets"].items():
            if b["n"]:
                print(f"  {stat:4s} k={k}: n={b['n']:4d} hit={b['hit_rate'] * 100:5.1f}% "
                      f"ROI={b['roi'] * 100:+6.2f}% profit={b['profit_units_100']:+9.0f}")
    print("  pooled:")
    for k, b in summary["pooled_bets"].items():
        if b["n"]:
            print(f"  all  k={k}: n={b['n']:4d} hit={b['hit_rate'] * 100:5.1f}% "
                  f"ROI={b['roi'] * 100:+6.2f}% profit={b['profit_units_100']:+9.0f}")
    print("\nprobability/EV rule (Normal CDF on tune-season residual SD):")
    for stat, e in sorted(summary["by_stat"].items()):
        for t, b in e["ev_bets"].items():
            if b["n"]:
                print(f"  {stat:4s} EV>={t}: n={b['n']:4d} hit={b['hit_rate'] * 100:5.1f}% "
                      f"ROI={b['roi'] * 100:+6.2f}% profit={b['profit_units_100']:+9.0f}")
    for t, b in summary["pooled_ev_bets"].items():
        if b["n"]:
            print(f"  all  EV>={t}: n={b['n']:4d} hit={b['hit_rate'] * 100:5.1f}% "
                  f"ROI={b['roi'] * 100:+6.2f}% profit={b['profit_units_100']:+9.0f}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-date", default="2025-10-21")
    ap.add_argument("--to-date", default="2025-10-31")
    ap.add_argument("--budget-credits", type=int, default=400)
    ap.add_argument("--estimate-only", action="store_true",
                    help="Print credit estimates (free) and stop")
    ap.add_argument("--analyze-only", action="store_true",
                    help="Use cached DunkAPI responses only (0 credits)")
    ap.add_argument("--force", action="store_true", help="Pull even if over budget")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--gamelog-cache", type=Path, default=DEFAULT_GAMELOG_CACHE)
    ap.add_argument("--players", default="",
                    help="Comma-separated player names from the default set (default: all)")
    ap.add_argument("--out", type=Path, default=None, help="Write JSON summary here")
    args = ap.parse_args(argv)

    players = DEFAULT_PLAYERS
    if args.players:
        wanted = {p.strip() for p in args.players.split(",") if p.strip()}
        players = {n: p for n, p in DEFAULT_PLAYERS.items() if n in wanted or p in wanted}
        if not players:
            ap.error(f"no default players match {sorted(wanted)}")

    client = None
    if not args.analyze_only:
        key = os.environ.get("DUNKAPI_API_KEY", "")
        print(f"DUNKAPI_API_KEY present: {bool(key)} (len={len(key)})")
        if key:
            client = DunkClient(key)
    try:
        props = fetch_props(
            players, args.from_date, args.to_date, client=client,
            cache_dir=args.cache_dir, budget_credits=args.budget_credits,
            estimate_only=args.estimate_only, analyze_only=args.analyze_only,
            force=args.force,
        )
    except (BudgetExceeded, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    if args.estimate_only:
        return 0
    print(f"raw prop rows: {len(props)}")

    consensus = consensus_props(props)
    print(f"consensus player-game-markets (all minutes): {len(consensus)}")
    logs, sd = load_projections(args.gamelog_cache)
    graded_all = grade(consensus, logs, sd)
    graded = graded_all[
        (graded_all["games_prior"] >= 10) & (graded_all["min_proj"] >= 20)
    ].reset_index(drop=True)
    print(f"matched before min/games filter: {len(graded_all)}; "
          f"after (games_prior>=10, min_proj>=20): {len(graded)}")
    summary = summarize(graded)
    summary["meta"] = {
        "from_date": args.from_date, "to_date": args.to_date,
        "players": players, "raw_props": int(len(props)),
        "consensus_all": int(len(consensus)), "matched_all": int(len(graded_all)),
        "tune_resid_sd": sd,
    }
    print_report(summary, graded, sd)
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
