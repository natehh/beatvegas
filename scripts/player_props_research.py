#!/usr/bin/env python3
"""NBA player props: can a statistical model find an edge?

The hard constraint is data. There is NO free historical archive of book
prop lines: The Odds API's historical props are paid-tier (verified —
the free key returns HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN), Kaggle
sets are game-lines only, and the public dumps people ask for in forums
don't seem to exist. So this study has three legs:

  1. A projection model (projected minutes x shrunk per-minute rates,
     pace/opponent adjustments) built from ESPN athlete gamelogs,
     evaluated walk-forward on 2025-26 against naive baselines. It is
     genuinely better than season-average / last-10 baselines
     (points MAE 4.83 vs 5.06 / 4.85; PRA 6.32 vs 6.77 / 6.39) — real
     but modest signal, mostly from minutes projection.
  2. A SIMULATED-LINE backtest: pseudo-lines set at a player's recent-
     form baseline, betting the model at -110. This produces gaudy
     ROIs (+16% to +40%) and is NOT evidence of a book edge — real
     books price minutes, injuries, and usage shifts that a naive
     baseline ignores (see leg 3). It only sizes the model's signal
     against a weak line-setter.
  3. A live diagnostic on real lines from the repo's forward collector
     (data/props/prop_quotes.jsonl, The Odds API): opening-night lines
     already embed Tatum returning from injury (Brown's points line
     ~20.5 vs a 27.7 naive average), Embiid minutes restrictions, etc.
     Books are not naive. Overs are mildly shaded (median -112 over vs
     -116/-121 under), and the median cross-book line gap is 1.0 pt —
     line shopping is worth real points once you can bet.

The honest edge test is the forward one: collect lines daily
(`python scripts/collect_prop_odds.py --hours-ahead 30`), settle
(`python scripts/settle_prop_odds.py`), then grade the model against
actual closing prices. A DunkAPI free key (500 credits/month) would
also unlock archived closing lines back to May 2023 for a true
historical backtest.

Usage:
    python scripts/player_props_research.py            # fetch + evaluate
    python scripts/player_props_research.py --analyze-only
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "cache" / "props_logs"
UA = {"User-Agent": "Mozilla/5.0 beatvegas-research"}

LABEL_IDX = {"min": 0, "fg3m": 3, "reb": 7, "ast": 8, "blk": 9, "stl": 10, "pts": 13}
SEASONS = [(2025, "2024-25"), (2026, "2025-26")]

_cache_dir = DEFAULT_CACHE


def fetch_json(url: str, name: str) -> dict | None:
    path = _cache_dir / name
    if path.exists():
        return json.loads(path.read_text())
    try:
        req = urllib.request.Request(url, headers=UA)
        d = json.load(urllib.request.urlopen(req, timeout=40))
        path.write_text(json.dumps(d))
        return d
    except Exception as exc:  # noqa: BLE001
        print(f"fetch fail {name}: {exc}")
        return None


def get_athletes() -> list[dict]:
    out: dict[str, dict] = {}
    for tid in range(1, 31):
        d = fetch_json(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{tid}/roster",
            f"roster_{tid}.json",
        )
        if not d:
            continue
        for a in d.get("athletes", []):
            items = a.get("items") if isinstance(a, dict) else None
            for ath in (items if items else [a]):
                if ath.get("id"):
                    out[str(ath["id"])] = {
                        "id": str(ath["id"]),
                        "name": ath.get("displayName") or ath.get("fullName") or "",
                    }
    return list(out.values())


def parse_gamelog(d: dict, athlete: dict, season_label: str) -> list[dict]:
    """Extract regular-season player-games from an ESPN gamelog payload."""
    rows = []
    events = d.get("events") or {}
    reg: dict[str, list] = {}
    for st in d.get("seasonTypes") or []:
        if "Regular Season" not in (st.get("displayName") or ""):
            continue
        for cat in st.get("categories") or []:
            for cev in cat.get("events") or []:
                reg[str(cev.get("eventId"))] = cev.get("stats")
    for eid, stats in reg.items():
        ev = events.get(eid)
        if not ev or not stats:
            continue
        try:
            mins = float(stats[LABEL_IDX["min"]])
        except Exception:
            continue
        if mins <= 0:
            continue

        def made(s) -> float:
            try:
                return float(str(s).split("-")[0])
            except Exception:
                return 0.0

        rows.append(
            {
                "player_id": athlete["id"],
                "player": athlete["name"],
                "season": season_label,
                "event_id": eid,
                "date": (ev.get("gameDate") or "")[:10],
                "team": (ev.get("team") or {}).get("abbreviation") or "",
                "opp": (ev.get("opponent") or {}).get("abbreviation") or "",
                "game_total": float(ev.get("homeTeamScore") or 0)
                + float(ev.get("awayTeamScore") or 0),
                "min": mins,
                "pts": float(stats[LABEL_IDX["pts"]]),
                "reb": float(stats[LABEL_IDX["reb"]]),
                "ast": float(stats[LABEL_IDX["ast"]]),
                "fg3m": made(stats[LABEL_IDX["fg3m"]]),
                "stl": float(stats[LABEL_IDX["stl"]]),
                "blk": float(stats[LABEL_IDX["blk"]]),
            }
        )
    return rows


def build_logs() -> pd.DataFrame:
    cache_file = _cache_dir / "gamelogs.pkl"
    if cache_file.exists():
        return pd.read_pickle(cache_file)
    athletes = get_athletes()
    print(f"athletes: {len(athletes)}")

    def one(args):
        ath, year, label = args
        d = fetch_json(
            "https://site.api.espn.com/apis/common/v3/sports/basketball/nba/"
            f"athletes/{ath['id']}/gamelog?season={year}",
            f"gamelog_{ath['id']}_{year}.json",
        )
        return parse_gamelog(d, ath, label) if d else []

    jobs = [(a, y, lab) for a in athletes for y, lab in SEASONS]
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for got in ex.map(one, jobs):
            rows.extend(got)
    df = pd.DataFrame(rows).sort_values(["player_id", "date"]).reset_index(drop=True)
    df.to_pickle(cache_file)
    return df


def add_pregame_features(df: pd.DataFrame) -> pd.DataFrame:
    """Strictly pregame rolling features (expanding/rolling on prior games)."""
    df = df.copy()
    df["pra"] = df["pts"] + df["reb"] + df["ast"]

    ev = df.groupby(["season", "event_id"]).agg(
        date=("date", "first"), total=("game_total", "first")
    ).reset_index()
    team_ev = df.groupby(["season", "event_id", "team"], as_index=False).agg(
        team_score=("pts", "sum")
    )
    team_ev = team_ev.merge(ev, on=["season", "event_id"])
    team_ev["opp_score"] = team_ev["total"] - team_ev["team_score"]
    team_ev = team_ev.sort_values(["team", "date"])
    team_ev["allowed_prior"] = team_ev.groupby("team")["opp_score"].transform(
        lambda s: s.expanding().mean().shift(1)
    )
    team_ev["pace_prior"] = team_ev.groupby("team")["total"].transform(
        lambda s: s.expanding().mean().shift(1)
    )
    ctx = team_ev[["season", "event_id", "team", "pace_prior"]]
    opp_ctx = team_ev[["season", "event_id", "team", "allowed_prior", "pace_prior"]].rename(
        columns={"team": "opp", "allowed_prior": "opp_allowed_prior",
                 "pace_prior": "opp_pace_prior"}
    )
    df = df.merge(ctx, on=["season", "event_id", "team"], how="left")
    df = df.merge(opp_ctx, on=["season", "event_id", "opp"], how="left")

    df = df.sort_values(["player_id", "date", "event_id"]).reset_index(drop=True)
    g = df.groupby("player_id")
    df["games_prior"] = g.cumcount()
    last_date = g["date"].shift(1)
    df["rest_days"] = (
        pd.to_datetime(df["date"]) - pd.to_datetime(last_date)
    ).dt.days.fillna(7).clip(0, 14)
    for c in ["min", "pts", "reb", "ast", "pra", "fg3m"]:
        prior = g[c].shift(1)
        df[f"{c}_season_prior"] = prior.groupby(df["player_id"]).transform(
            lambda s: s.expanding().mean()
        )
        df[f"{c}_l10_prior"] = prior.groupby(df["player_id"]).transform(
            lambda s: s.rolling(10, min_periods=1).mean()
        )
        df[f"{c}_l5_prior"] = prior.groupby(df["player_id"]).transform(
            lambda s: s.rolling(5, min_periods=1).mean()
        )
    for c in ["pts", "reb", "ast", "pra", "fg3m"]:
        df[f"{c}_permin_prior"] = (
            g[c].shift(1).groupby(df["player_id"]).transform(lambda s: s.expanding().sum())
            / g["min"].shift(1).groupby(df["player_id"]).transform(lambda s: s.expanding().sum())
        )
    df["min_proj"] = 0.65 * df["min_l10_prior"] + 0.35 * df["min_season_prior"]
    league_total = df.groupby("season")["game_total"].transform("mean")
    game_pace = (
        df["pace_prior"].fillna(league_total) + df["opp_pace_prior"].fillna(league_total)
    ) / 2.0
    df["pace_factor"] = (game_pace / league_total).fillna(1.0).clip(0.9, 1.1)
    df["opp_def_factor"] = (df["opp_allowed_prior"] / (league_total / 2.0)).fillna(1.0).clip(0.9, 1.1)
    return df


def add_projections(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["pts", "reb", "ast", "pra", "fg3m"]:
        season_rate = df[f"{c}_permin_prior"]
        l10_rate = df[f"{c}_l10_prior"] / df["min_l10_prior"].replace(0, np.nan)
        rate = 0.6 * season_rate + 0.4 * l10_rate.fillna(season_rate)
        df[f"{c}_proj"] = df["min_proj"] * rate * df["pace_factor"]
    df["pts_proj"] = df["pts_proj"] * df["opp_def_factor"]
    return df


def _roi_str(bets: list[float]) -> str:
    n = len(bets)
    if n == 0:
        return "n=0"
    w = sum(1 for b in bets if b > 0)
    p = sum(bets)
    return f"n={n:5d} hit={w / n * 100:5.1f}% ROI={p / (100 * n) * 100:+6.2f}% profit={p:+9.0f}"


def _settle(actual: float, line: float, side: str) -> float:
    d = actual - line
    if d == 0:
        return 0.0
    win = (d > 0) if side == "over" else (d < 0)
    return 100 * 100 / 110 if win else -100.0


def evaluate(df: pd.DataFrame) -> None:
    test = df[(df["season"] == "2025-26") & (df["games_prior"] >= 10)]
    tune = df[(df["season"] == "2024-25") & (df["games_prior"] >= 10)]
    print("\n===== Projection accuracy, 2025-26 (games_prior >= 10) =====")
    for c in ["pts", "reb", "ast", "pra"]:
        cells = []
        for name, col in [("model", f"{c}_proj"), ("season_avg", f"{c}_season_prior"),
                          ("l10_avg", f"{c}_l10_prior"), ("l5_avg", f"{c}_l5_prior")]:
            mae = (test[c] - test[col]).abs().mean()
            cells.append(f"{name} MAE={mae:.2f}")
        print(f"  {c:4s} " + " | ".join(cells))
    sub = test[test["min_proj"] >= 20]
    print(f"-- prop-relevant subset (min_proj>=20, n={len(sub)}) --")
    for c in ["pts", "reb", "ast", "pra"]:
        print(f"  {c:4s} model MAE={(sub[c] - sub[c + '_proj']).abs().mean():.2f} | "
              f"season {(sub[c] - sub[c + '_season_prior']).abs().mean():.2f} | "
              f"L10 {(sub[c] - sub[c + '_l10_prior']).abs().mean():.2f}")

    print("\n===== SIMULATED-LINE backtest (pseudo-lines, NOT real book lines) =====")
    print("pseudo-line = player baseline average rounded to a .5 line; bet model at -110")
    for c in ["pts", "reb", "ast", "pra"]:
        sd = (tune[c] - tune[f"{c}_proj"]).std()
        for pname, pcol in [("L10", f"{c}_l10_prior"), ("season", f"{c}_season_prior")]:
            for k in (0.3, 0.5):
                bets = []
                for _, r in test.iterrows():
                    line = np.floor(r[pcol] * 2) / 2 + 0.5
                    e = r[f"{c}_proj"] - line
                    if abs(e) < k * sd:
                        continue
                    bets.append(_settle(r[c], line, "over" if e > 0 else "under"))
                print(f"  {c:4s} vs {pname:6s} k={k}: {_roi_str(bets)}")


def live_diagnostic(df: pd.DataFrame) -> None:
    ledger = ROOT / "data" / "props" / "prop_quotes.jsonl"
    if not ledger.exists():
        print("\n(no forward-collected quotes yet; run scripts/collect_prop_odds.py)")
        return
    print("\n===== Live-line diagnostic (forward-collected real book lines) =====")
    quotes = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    last = df[df["season"] == "2025-26"]
    agg = last.groupby("player").agg(
        mpg=("min", "mean"), pts=("pts", "mean"), reb=("reb", "mean"), ast=("ast", "mean")
    )
    mkt = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast"}
    rows = []
    for q in quotes:
        stat = mkt.get(q["market"])
        if not stat or q["player"] not in agg.index:
            continue
        rows.append({
            "player": q["player"], "book": q["bookmaker"], "market": q["market"],
            "line": q["line"], "over": q["over_odds"], "under": q["under_odds"],
            "naive_avg": float(agg.loc[q["player"], stat]),
        })
    if not rows:
        print("no matched quotes")
        return
    r = pd.DataFrame(rows)
    r["gap"] = (r["line"] - r["naive_avg"]).abs()
    print(f"matched quotes: {len(r)}")
    print("mean |line - last-season average| by market:")
    print(r.groupby("market")["gap"].mean().round(2).to_string())
    print("median over/under odds by market (over shading check):")
    print(r.groupby("market")[["over", "under"]].median().to_string())
    g = r.groupby(["player", "market"])["line"].agg(["min", "max"])
    g["disp"] = g["max"] - g["min"]
    print(f"cross-book line dispersion: median {g['disp'].median():.1f} pts; "
          f"{(g['disp'] >= 1).sum()}/{len(g)} props differ by >= 1 pt across books")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    global _cache_dir
    _cache_dir = args.cache_dir
    _cache_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only and (_cache_dir / "gamelogs.pkl").exists():
        df = pd.read_pickle(_cache_dir / "gamelogs.pkl")
    else:
        df = build_logs()
    print(f"gamelog rows: {len(df)}")
    print(df.groupby("season").size().to_string())
    df = add_projections(add_pregame_features(df))
    evaluate(df)
    live_diagnostic(df)


if __name__ == "__main__":
    main()
