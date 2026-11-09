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

     The v2 revision (after the DunkAPI closing-line grade below)
     keeps that skeleton and adds: a 2023-24 warm-up season, EWMA +
     current-season features with season-discounted expanding priors,
     a usage-based points rate (usage/min x points per possession),
     per-stat opponent factors, rotation-absence and return-from-
     injury minutes adjustments, and a walk-forward gradient-
     boosting layer trained only on prior seasons (add_ml_projections,
     50/50 blend, FG3M excluded). On the 2025-26 rotation subset it
     improves minutes MAE 4.78 -> 4.47 and points 5.36 -> 5.16
     versus v1; calibration scaling fit on 2024-25 was tested and
     rejected (<=0.01 MAE either way).
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
actual closing prices. That grade has now been run historically too:
`scripts/dunkapi_props_backtest.py` uses DunkAPI archived closes
(back to May 2023) -- v1 lost badly on October 2025 lines, and the
v2 model above still loses that regime-shift window but beat the
close on MAE in 4 of 5 stats (and pooled +30% ROI on the EV rule,
n=42) in a small fresh January 2026 holdout. See that script's
docstring; neither window is large enough to claim an edge.

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
# 2023-24 is a warm-up season only: it gives 2024-25 a prior-season history
# (so season-boundary effects can be learned) and a training season for the
# walk-forward ML layer. Headline evaluation remains 2025-26.
SEASONS = [(2024, "2023-24"), (2025, "2024-25"), (2026, "2025-26"), (2027, "2026-27")]

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

        def attempted(s) -> float:
            try:
                return float(str(s).split("-")[1])
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
                # Usage components (FG/FT attempts, turnovers) let the
                # model separate volume from efficiency instead of
                # projecting raw per-minute scoring.
                "fga": attempted(stats[1]) if len(stats) > 1 else 0.0,
                "fg3a": attempted(stats[3]) if len(stats) > 3 else 0.0,
                "fta": attempted(stats[5]) if len(stats) > 5 else 0.0,
                "tov": float(stats[12]) if len(stats) > 12 else 0.0,
            }
        )
    return rows


def build_logs() -> pd.DataFrame:
    cache_file = _cache_dir / "gamelogs_v4.pkl"
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
    df = _add_v2_features(df)
    return df


def _add_discounted_priors(df: pd.DataFrame, discount: float = 0.25) -> pd.DataFrame:
    """Expanding priors whose weight is multiplied by ``discount`` at
    each season boundary (``*_dprior`` means, ``*_dpermin`` rates).

    With three seasons of history a plain expanding mean lets 100+
    old games swamp the current season -- exactly the opening-weeks
    staleness the closing-line test exposed. Discounting keeps the
    prior-season level as a starting point while letting a handful of
    current-season games move it.
    """
    stats = ["pts", "reb", "ast", "pra", "fg3m"]
    cols = ["min"] + stats
    n = len(df)
    out_mean = {c: np.full(n, np.nan) for c in cols}
    out_rate = {c: np.full(n, np.nan) for c in stats}
    out_ppp = np.full(n, np.nan)
    seasons = df["season"].to_numpy()
    mins = df["min"].to_numpy(dtype=float)
    ppp_vals = (df["pts"] / df["usage_poss"].replace(0, np.nan)).to_numpy(dtype=float)
    arrays = {c: df[c].to_numpy(dtype=float) for c in cols}
    for idxs in df.groupby("player_id", sort=False).indices.values():
        idxs = np.sort(np.asarray(idxs))
        w = 0.0
        wppp = 0.0
        smin = 0.0
        sums = {c: 0.0 for c in cols}
        sppp = 0.0
        prev = None
        for i in idxs:
            if prev is not None and seasons[i] != prev:
                w *= discount
                wppp *= discount
                smin *= discount
                sppp *= discount
                for c in cols:
                    sums[c] *= discount
            prev = seasons[i]
            if w > 0:
                for c in cols:
                    out_mean[c][i] = sums[c] / w
                if smin > 0:
                    for c in stats:
                        out_rate[c][i] = sums[c] / smin
                out_ppp[i] = sppp / wppp if wppp > 0 else np.nan
            w += 1.0
            smin += mins[i]
            for c in cols:
                sums[c] += arrays[c][i]
            if not np.isnan(ppp_vals[i]):
                sppp += ppp_vals[i]
                wppp += 1.0
    for c in cols:
        df[f"{c}_dprior"] = out_mean[c]
    for c in stats:
        df[f"{c}_dpermin"] = out_rate[c]
    df["ppp_dprior"] = out_ppp
    return df


def _add_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recency, opponent-per-stat, rotation-absence and usage features.

    Added after the DunkAPI closing-line test (Oct 2025) showed the v1
    model's errors were concentrated in (a) stale prior-season rates in
    the opening weeks, (b) minutes -- projected minutes were nearly
    uncorrelated with actual minutes for starters, and (c) role changes
    (teammate absences, returns from injury) that books price and v1
    ignored. All features remain strictly pregame: every rolling /
    expanding / EWMA statistic is shifted by one game, opponent factors
    use only prior games, and rotation absence is measured against the
    team's previous five games (a depth-chart proxy known pregame).
    """
    from collections import deque

    for col in ["fga", "fg3a", "fta", "tov"]:
        if col not in df:
            df[col] = 0.0
    df["usage_poss"] = df["fga"] + 0.44 * df["fta"] + df["tov"]
    df["cs_games_prior"] = df.groupby(["player_id", "season"]).cumcount()

    stats = ["pts", "reb", "ast", "pra", "fg3m"]
    for h in (3, 5, 8):
        for c in ["min", "usage_poss"] + stats:
            df[f"{c}_ewm{h}"] = df.groupby("player_id")[c].transform(
                lambda s, h=h: s.shift(1).ewm(halflife=h, adjust=False, min_periods=1).mean()
            )
    for c in ["min"] + stats:
        gcs = df.groupby(["player_id", "season"])[c]
        df[f"{c}_cs_prior"] = gcs.transform(lambda s: s.shift(1).expanding().mean())
        df[f"{c}_cs_l5"] = gcs.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    ppp = df["pts"] / df["usage_poss"].replace(0, np.nan)
    df["_ppp"] = ppp
    gp = df.groupby("player_id")["_ppp"]
    df["ppp_ewm5"] = gp.transform(
        lambda s: s.shift(1).ewm(halflife=5, adjust=False, min_periods=1).mean()
    )
    df["ppp_season_prior"] = gp.transform(lambda s: s.shift(1).expanding().mean())
    df = df.drop(columns=["_ppp"])
    df["usage_permin_ewm5"] = df["usage_poss_ewm5"] / df["min_ewm5"].replace(0, np.nan)
    df = _add_discounted_priors(df, discount=0.25)
    df["ppp_season_prior"] = df.get("ppp_dprior", df["ppp_season_prior"])

    # Opponent factors per stat (what the opposing team has allowed).
    team_tot = df.groupby(["season", "event_id", "team"], as_index=False).agg(
        date=("date", "first"), opp=("opp", "first"),
        pts=("pts", "sum"), reb=("reb", "sum"), ast=("ast", "sum"), fg3m=("fg3m", "sum"),
    )
    allowed = team_tot.rename(
        columns={"team": "opp", "opp": "team", "pts": "a_pts", "reb": "a_reb",
                 "ast": "a_ast", "fg3m": "a_fg3m"}
    )[["season", "event_id", "team", "a_pts", "a_reb", "a_ast", "a_fg3m"]]
    team_tot = team_tot.merge(allowed, on=["season", "event_id", "team"], how="left")
    team_tot = team_tot.sort_values(["team", "date"])
    for c in ["pts", "reb", "ast", "fg3m"]:
        prior = team_tot.groupby("team")[f"a_{c}"].transform(
            lambda s: s.expanding().mean().shift(1)
        )
        league = team_tot.groupby("season")[f"a_{c}"].transform("mean")
        team_tot[f"opp_{c}_factor"] = (prior / league).clip(0.90, 1.10)
    # team_tot's factor for a team is what that team has *allowed*, i.e.
    # its defensive tendency -- so a player uses his opponent's row.
    df = df.merge(
        team_tot[["season", "event_id", "team"]
                 + [f"opp_{c}_factor" for c in ["pts", "reb", "ast", "fg3m"]]]
        .rename(columns={"team": "opp"}),
        on=["season", "event_id", "opp"], how="left",
    )
    for c in ["pts", "reb", "ast", "fg3m"]:
        df[f"opp_{c}_factor"] = df[f"opp_{c}_factor"].fillna(1.0)

    # Rotation absence (share of expected team minutes missing, from the
    # last-5-games depth chart) and the player's own missed team games.
    df = df.sort_values(["team", "date", "event_id", "player_id"]).reset_index(drop=True)
    events = (
        df[["season", "team", "event_id", "date"]]
        .drop_duplicates()
        .sort_values(["season", "team", "date", "event_id"])
    )
    seq: dict[tuple, int] = {}
    for (season, team), grp_ev in events.groupby(["season", "team"]):
        for i, eid in enumerate(grp_ev["event_id"]):
            seq[(season, team, eid)] = i
    hist: dict[tuple, deque] = {}
    player_last: dict[tuple, int] = {}
    recs: list[tuple] = []
    for (season, team, eid), grp in df.groupby(["season", "team", "event_id"], sort=False):
        key = (season, team)
        idx = seq[(season, team, eid)]
        dq = hist.get(key, deque(maxlen=5))
        expected: dict[str, float] = {}
        if dq:
            for pid in set().union(*[set(d.keys()) for d in dq]):
                expected[pid] = sum(d.get(pid, 0.0) for d in dq) / len(dq)
        present = set(grp["player_id"])
        missing_min = sum(
            m for p, m in expected.items() if p not in present and m >= 8
        )
        for row in grp.itertuples():
            last = player_last.get((season, row.player_id))
            missed = 0 if last is None else max(0, idx - last - 1)
            recs.append((row.player_id, eid, missing_min / 240.0, missed))
            player_last[(season, row.player_id)] = idx
        dq.append(dict(zip(grp["player_id"], grp["min"])))
        hist[key] = dq
    feat = pd.DataFrame(
        recs, columns=["player_id", "event_id", "missing_share5", "own_missed_games"]
    )
    df = df.merge(feat, on=["player_id", "event_id"], how="left")
    df["missing_share5"] = df["missing_share5"].fillna(0.0)
    df["own_missed_games"] = df["own_missed_games"].fillna(0)
    return df.sort_values(["player_id", "date", "event_id"]).reset_index(drop=True)


def add_projections(df: pd.DataFrame) -> pd.DataFrame:
    """v2 heuristic projections (see `_add_v2_features` for the why).

    Minutes blend the 3-game EWMA, current-season L5 and the all-history
    mean, boosted slightly when rotation teammates are missing and cut
    10% for players returning after 2+ missed team games (returners were
    over-projected by ~3.6 minutes in the closing-line test). Rates
    blend the original expanding/L10 per-minute rate with a 5-game EWMA
    rate; points instead blends the original rate with a usage-based
    rate (usage possessions per minute x points per possession), and
    each stat gets its own opponent factor. PRA is projected directly,
    not as the sum of components (it graded better both seasons).
    """
    df = df.copy()
    cs_min = df["min_cs_l5"].fillna(df["min_ewm5"]) if "min_cs_l5" in df else df["min_l10_prior"]
    long_min = df["min_dprior"].fillna(df["min_season_prior"]) if "min_dprior" in df else df["min_season_prior"]
    min_base = (
        0.45 * df["min_ewm3"] + 0.30 * cs_min + 0.25 * long_min
        if "min_ewm3" in df
        else df["min_proj"]
    )
    min_adj = min_base * (1 + 0.20 * df.get("missing_share5", 0.0))
    # First three games of a season: coaches run wider rotations and
    # ramp starters up -- projected minutes ran a consistent
    # actual/projected ratio of 0.92-0.95 in cs<=2 across 2024-25,
    # 2025-26 and 2026-27 (bias +1.4 to +2.0 min), ~1.0 thereafter.
    cs_games = df.get("cs_games_prior", pd.Series(99, index=df.index))
    min_adj = min_adj * np.where(cs_games <= 2, 0.95, 1.0)
    missed = df.get("own_missed_games", pd.Series(0, index=df.index))
    df["min_proj"] = np.where(missed >= 2, min_adj * 0.90, min_adj).clip(0, 48)

    def blended_rate(c: str) -> pd.Series:
        season_rate = (
            df[f"{c}_dpermin"].fillna(df[f"{c}_permin_prior"])
            if f"{c}_dpermin" in df
            else df[f"{c}_permin_prior"]
        )
        l10_rate = df[f"{c}_l10_prior"] / df["min_l10_prior"].replace(0, np.nan)
        orig = 0.6 * season_rate + 0.4 * l10_rate.fillna(season_rate)
        if f"{c}_ewm5" in df:
            ew = df[f"{c}_ewm5"] / df["min_ewm5"].replace(0, np.nan)
            return 0.5 * orig + 0.5 * ew.fillna(orig), orig
        return orig, orig

    for c in ["reb", "ast", "fg3m"]:
        rate, _ = blended_rate(c)
        factor = df.get(f"opp_{c}_factor", df["opp_def_factor"])
        df[f"{c}_proj"] = (df["min_proj"] * rate * df["pace_factor"] * factor).clip(lower=0)
    rate_pts, _ = blended_rate("pts")
    if "usage_permin_ewm5" in df:
        usage_rate = df["usage_permin_ewm5"] * (
            0.5 * df["ppp_ewm5"] + 0.5 * df["ppp_season_prior"]
        )
        rate_pts = 0.5 * rate_pts + 0.5 * usage_rate.fillna(rate_pts)
    df["pts_proj"] = (
        df["min_proj"] * rate_pts * df["pace_factor"]
        * df.get("opp_pts_factor", df["opp_def_factor"])
    ).clip(lower=0)
    rate_pra, _ = blended_rate("pra")
    df["pra_proj"] = (df["min_proj"] * rate_pra * df["pace_factor"]).clip(lower=0)
    return df


_ML_TARGETS = ["min", "pts", "reb", "ast", "pra"]  # fg3m: ML graded worse; kept heuristic


def _ml_feature_columns(target: str) -> list[str]:
    feats = [
        "games_prior", "rest_days", "pace_factor", "opp_def_factor",
        "min_season_prior", "min_l10_prior", "min_l5_prior",
        "min_ewm3", "min_ewm5", "min_ewm8", "min_cs_l5",
        "missing_share5", "own_missed_games", "cs_games_prior",
        "opp_pts_factor", "opp_reb_factor", "opp_ast_factor", "opp_fg3m_factor",
        "usage_permin_ewm5", "ppp_ewm5", "min_proj",
    ]
    feats.append("min_dprior")
    for c in ["pts", "reb", "ast", "pra", "fg3m"]:
        feats += [
            f"{c}_season_prior", f"{c}_l10_prior", f"{c}_l5_prior",
            f"{c}_permin_prior", f"{c}_ewm5", f"{c}_ewm8", f"{c}_cs_l5",
            f"{c}_dprior", f"{c}_dpermin", f"{c}_proj",
        ]
    return [f for f in dict.fromkeys(feats)]


def add_ml_projections(df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward gradient-boosting blend, trained only on prior seasons.

    For each season with at least one earlier season of history, a
    HistGradientBoostingRegressor per target is fit on all prior-season
    rows (games_prior >= 10) and blended 50/50 with the v2 heuristic.
    Seasons without prior history (and any environment without
    scikit-learn) keep the heuristic projections unchanged, so this is
    always safe to call. Heuristic values are kept in ``*_proj_heur``.
    """
    df = df.copy()
    for c in _ML_TARGETS:
        col = "min_proj" if c == "min" else f"{c}_proj"
        df[f"{col}_heur"] = df[col]
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except Exception:  # noqa: BLE001 - ML is optional
        return df
    seasons = sorted(df["season"].unique())
    for i, season in enumerate(seasons):
        if i == 0:
            continue
        # Train on ALL rows (games_prior >= 10). An earlier version
        # trained only on rotation players (min_proj >= 20) and then
        # applied the model to everyone: out of distribution, it
        # over-projected bench minutes by +4 to +7 in every season
        # (found Nov 2026 on 2026-27 data). Training on all rows
        # removes that bias and leaves rotation accuracy unchanged.
        train = df[
            df["season"].isin(seasons[:i])
            & (df["games_prior"] >= 10)
        ]
        test_mask = (df["season"] == season) & (df["games_prior"] >= 10)
        if len(train) < 500 or not test_mask.any():
            continue
        for target in _ML_TARGETS:
            col = "min_proj" if target == "min" else f"{target}_proj"
            feats = [f for f in _ml_feature_columns(target) if f in df.columns]
            med = train[feats].median()
            try:
                model = HistGradientBoostingRegressor(
                    learning_rate=0.03, max_depth=2, max_leaf_nodes=15,
                    max_iter=300, l2_regularization=1.0, random_state=0,
                )
                model.fit(train[feats].fillna(med), train[target])
                pred = model.predict(df.loc[test_mask, feats].fillna(med))
            except Exception:  # noqa: BLE001 - never break the heuristic path
                continue
            blended = 0.5 * pred + 0.5 * df.loc[test_mask, f"{col}_heur"].to_numpy()
            if target == "min":
                blended = np.clip(blended, 0, 48)
            df.loc[test_mask, col] = np.clip(blended, 0, None)
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

    if args.analyze_only and (_cache_dir / "gamelogs_v4.pkl").exists():
        df = pd.read_pickle(_cache_dir / "gamelogs_v4.pkl")
    else:
        df = build_logs()
    print(f"gamelog rows: {len(df)}")
    print(df.groupby("season").size().to_string())
    df = add_ml_projections(add_projections(add_pregame_features(df)))
    evaluate(df)
    live_diagnostic(df)


if __name__ == "__main__":
    main()
