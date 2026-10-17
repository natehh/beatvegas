#!/usr/bin/env python3
"""Non-Elo modelling families for college basketball vs the consensus line.

Companion to scripts/college_public_fade_research.py (which builds the
dataset and tests the Elo family). This script asks whether *other*
modelling families do any better against the VegasInsider consensus line:

  1. Massey least-squares power ratings (weekly ridge refits on margins)
  2. Adjusted efficiency: per-team offence/defence least squares on scores
     (KenPom-style adjusted ratings, weekly refits)
  3. Ridge + gradient boosting on pregame features (Elo/Massey/efficiency
     predictions, recent form, rest, scoring), both outright-margin and
     market-residual targets

Protocol is identical for every family: calibrate/select thresholds on
2024-25, evaluate untouched on 2025-26, settle flat $100 bets at the VI
consensus line/price, and also test fading each model.

Results (holdout 2025-26, edge thresholds 0-6 pts): every family loses
in both directions. Massey -5.3% to -6.4%, efficiency -5.6% to -6.1%,
ridge -3.9% to -4.7%, gradient boosting -3.0% to -4.1% (-2.4% to -5.1%
for the market-residual variant). GBM's spectacular tuning-season
numbers (+16% to +90% depending on target) are pure in-sample overfit.
Conclusion: model family is not the bottleneck — a public-data model's
disagreement with the college consensus line is model error, not edge.

Requires the cache built by college_public_fade_research.py
(dataset.json + espn_ncaab_*.json in --cache-dir). The ML section needs
scikit-learn (research-only dependency, not in requirements.txt).

Usage:
    python scripts/college_public_fade_research.py          # build cache first
    python scripts/college_model_families_research.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "college_public_fade_research", ROOT / "scripts" / "college_public_fade_research.py"
)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

Elo, season_of, norm = base.Elo, base.season_of, base.norm
parse_price, settle_spread, summarize = base.parse_price, base.settle_spread, base.summarize


def load_espn_games(cache_dir: Path) -> list[dict]:
    games: dict[str, dict] = {}
    for f in sorted(cache_dir.glob("espn_ncaab_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for ev in d.get("events", []):
            try:
                comp = ev["competitions"][0]
                t = {}
                for c in comp["competitors"]:
                    team = c["team"]
                    t[c["homeAway"]] = {
                        "id": str(team.get("id")),
                        "location": (team.get("location") or "").strip(),
                        "score": int(c["score"]),
                    }
                games[str(ev["id"])] = {
                    "espn_id": str(ev["id"]),
                    "date": ev.get("date") or comp.get("date"),
                    "neutral": bool(comp.get("neutralSite")),
                    "home": t["home"],
                    "away": t["away"],
                }
            except Exception:
                continue
    return sorted(games.values(), key=lambda g: (g["date"], g["espn_id"]))


def weekly_refit_predictions(espn, vi_ids, mode: str, lam: float = 10.0):
    """Pregame predictions from least-squares ratings refit every 7 days.

    mode 'massey': margin = r_home - r_away + hca
    mode 'eff':    margin = (off_h - def_a) - (off_a - def_h) + hca
    """
    preds: dict[str, dict] = {}
    seasons: dict[str, list[dict]] = defaultdict(list)
    for g in espn:
        seasons[season_of(g["date"])].append(g)

    for _season, games in seasons.items():
        games = sorted(games, key=lambda g: (g["date"], g["espn_id"]))
        teams = sorted({g["home"]["id"] for g in games} | {g["away"]["id"] for g in games})
        tidx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        played: list[dict] = []
        games_played: dict[str, int] = defaultdict(int)
        ratings = None
        last_refit = None

        def refit():
            if mode == "massey":
                X = np.zeros((len(played), n + 1))
                y = np.zeros(len(played))
                for i, g in enumerate(played):
                    X[i, tidx[g["home"]["id"]]] = 1.0
                    X[i, tidx[g["away"]["id"]]] = -1.0
                    X[i, n] = 0.0 if g["neutral"] else 1.0
                    y[i] = g["home"]["score"] - g["away"]["score"]
                XtX = X.T @ X + lam * np.eye(n + 1)
                XtX[n, n] -= lam  # don't shrink home advantage
                beta = np.linalg.solve(XtX, X.T @ y)
                return beta, beta[n]
            X = np.zeros((2 * len(played), 2 * n + 2))
            y = np.zeros(2 * len(played))
            for i, g in enumerate(played):
                hi, ai = tidx[g["home"]["id"]], tidx[g["away"]["id"]]
                X[2 * i, hi] = 1.0
                X[2 * i, n + ai] = -1.0
                X[2 * i, 2 * n] = 1.0
                X[2 * i, 2 * n + 1] = 0.0 if g["neutral"] else 1.0
                y[2 * i] = g["home"]["score"]
                X[2 * i + 1, ai] = 1.0
                X[2 * i + 1, n + hi] = -1.0
                X[2 * i + 1, 2 * n] = 1.0
                y[2 * i + 1] = g["away"]["score"]
            XtX = X.T @ X + lam * np.eye(2 * n + 2)
            XtX[2 * n, 2 * n] -= lam
            XtX[2 * n + 1, 2 * n + 1] -= lam
            beta = np.linalg.solve(XtX, X.T @ y)
            return beta, beta[2 * n + 1]

        for g in games:
            gd = date.fromisoformat(g["date"][:10])
            if (ratings is None or (gd - last_refit).days >= 7) and len(played) >= 100:
                ratings = refit()
                last_refit = gd
            if g["espn_id"] in vi_ids and ratings is not None:
                beta, hca = ratings
                hi, ai = tidx[g["home"]["id"]], tidx[g["away"]["id"]]
                if mode == "massey":
                    mp = beta[hi] - beta[ai] + (0.0 if g["neutral"] else hca)
                else:
                    mp = (beta[hi] - beta[n + ai]) - (beta[ai] - beta[n + hi]) + \
                        (0.0 if g["neutral"] else hca)
                preds[g["espn_id"]] = {
                    "model_margin": float(mp),
                    "min_games": min(games_played[g["home"]["id"]],
                                     games_played[g["away"]["id"]]),
                }
            played.append(g)
            games_played[g["home"]["id"]] += 1
            games_played[g["away"]["id"]] += 1
    return preds


def elo_predictions(espn, vi_ids) -> dict[str, dict]:
    elo = Elo()
    preds: dict[str, dict] = {}
    cur = None
    for g in espn:
        s = season_of(g["date"])
        if cur is not None and s != cur:
            elo.regress_season()
        cur = s
        h, a = g["home"]["id"], g["away"]["id"]
        if g["espn_id"] in vi_ids:
            preds[g["espn_id"]] = {
                "elo_diff": elo.diff(h, a, g["neutral"]),
                "min_games": min(elo.games_played.get(h, 0), elo.games_played.get(a, 0)),
            }
        elo.update(h, a, g["home"]["score"] - g["away"]["score"], g["neutral"])
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path,
                    default=ROOT / "data" / "cache" / "college_vi")
    args = ap.parse_args()
    cache = args.cache_dir
    dataset = cache / "dataset.json"
    if not dataset.exists():
        sys.exit(f"{dataset} missing — run scripts/college_public_fade_research.py first")

    vi_games = json.loads(dataset.read_text())
    espn = load_espn_games(cache)
    espn_by_id = {g["espn_id"]: g for g in espn}

    def close(a: str, b: str) -> bool:
        return a == b or a in b or b in a or \
            re.sub(r" state$", "", a) == re.sub(r" state$", "", b)

    orientation: dict[str, bool] = {}
    for g in vi_games:
        eid = g.get("espn_id")
        if eid is None:
            continue
        eg = espn_by_id.get(str(eid))
        if not eg:
            continue
        ea, eh = norm(eg["away"]["location"]), norm(eg["home"]["location"])
        va, vh = norm(g.get("away")), norm(g.get("home"))
        if close(ea, va) and close(eh, vh):
            orientation[str(eid)] = True
        elif close(ea, vh) and close(eh, va):
            orientation[str(eid)] = False
    vi_ids = set(orientation)
    print(f"vi-espn linked games: {len(vi_ids)}")

    base_rows: dict[str, dict] = {}
    for g in vi_games:
        eid = g.get("espn_id")
        if eid is None or str(eid) not in orientation:
            continue
        t = g.get("trends")
        if not t or len(t.get("bets") or []) != 2:
            continue
        if g.get("away_score") is None or g.get("orientation_unknown"):
            continue
        away_line, away_price = parse_price(g.get("away_consensus"))
        home_line, home_price = parse_price(g.get("home_consensus"))
        if away_line is None or home_line is None or abs(away_line + home_line) > 0.51:
            continue
        base_rows[str(eid)] = {
            "season": g.get("season_label"),
            "margin": g["home_score"] - g["away_score"],
            "away_line": away_line, "home_line": home_line,
            "away_price": away_price, "home_price": home_price,
            "flip": not orientation[str(eid)],
        }

    def evaluate(name: str, pred_map: dict[str, dict]) -> list[dict]:
        rows = []
        for eid, br in base_rows.items():
            p = pred_map.get(eid)
            if p is None:
                continue
            r = dict(br)
            r["edge_home"] = (-p["model_margin"] if br["flip"] else p["model_margin"]) \
                + br["home_line"]
            r["min_games"] = p.get("min_games", 99)
            rows.append(r)
        s1 = [r for r in rows if r["season"] == "2024-25"]
        s2 = [r for r in rows if r["season"] == "2025-26"]

        def bets(rs, th, direction="with", mg=5):
            out = []
            for r in rs:
                if r["min_games"] < mg or abs(r["edge_home"]) < th:
                    continue
                side = "home" if r["edge_home"] > 0 else "away"
                if direction == "fade":
                    side = "away" if side == "home" else "home"
                out.append(settle_spread(r, side))
            return out

        print(f"\n===== {name} =====")
        for th in (0, 2, 4, 6):
            print(f"  tune th>={th}: {summarize(bets(s1, th))}")
        for th in (0, 2, 4, 6):
            print(f"  HOLDOUT th>={th}: {summarize(bets(s2, th))}")
        print(f"  HOLDOUT fade th>=4: {summarize(bets(s2, 4, 'fade'))} | "
              f"th>=6: {summarize(bets(s2, 6, 'fade'))}")
        return rows

    massey = weekly_refit_predictions(espn, vi_ids, "massey")
    print("massey preds:", len(massey))
    evaluate("Massey LSQ ratings", massey)

    eff = weekly_refit_predictions(espn, vi_ids, "eff")
    print("efficiency preds:", len(eff))
    evaluate("Adjusted efficiency (off/def LSQ)", eff)

    # Elo margins, calibrated on 2024-25 (same protocol as companion script)
    elo_raw = elo_predictions(espn, vi_ids)
    s1_pairs = [
        ((-p["elo_diff"] if base_rows[eid]["flip"] else p["elo_diff"]),
         base_rows[eid]["margin"])
        for eid, p in elo_raw.items()
        if eid in base_rows and base_rows[eid]["season"] == "2024-25" and p["min_games"] >= 3
    ]
    x = np.array([d for d, _ in s1_pairs])
    y = np.array([m for _, m in s1_pairs], dtype=float)
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y, rcond=None)
    a, b = coef
    elo_map = {
        eid: {"model_margin": a + b * p["elo_diff"], "min_games": p["min_games"]}
        for eid, p in elo_raw.items()
    }
    evaluate("Elo (recap)", elo_map)

    # ------------------------------------------------------------ ML features
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
    except ImportError:
        print("\nscikit-learn not installed; skipping ML section (research-only dep)")
        return

    hist: dict[str, list[tuple[date, int, int]]] = defaultdict(list)
    last_date: dict[str, date] = {}
    feat_by_eid: dict[str, dict] = {}
    for g in espn:
        gd = date.fromisoformat(g["date"][:10])
        h, a_ = g["home"]["id"], g["away"]["id"]
        if g["espn_id"] in vi_ids:
            def form(t):
                hs = hist[t][-10:]
                if not hs:
                    return (0.0, 0.0, 71.0)
                return (
                    float(np.mean([m for _, _, m in hs[-5:]])),
                    float(np.mean([m for _, _, m in hs])),
                    float(np.mean([pf for _, pf, _ in hs])),
                )

            fh, fa_ = form(h), form(a_)
            feat_by_eid[g["espn_id"]] = {
                "form5_diff": fh[0] - fa_[0],
                "form10_diff": fh[1] - fa_[1],
                "ppg_diff": fh[2] - fa_[2],
                "rest_diff": ((gd - last_date[h]).days if h in last_date else 7)
                - ((gd - last_date[a_]).days if a_ in last_date else 7),
                "neutral": 1.0 if g["neutral"] else 0.0,
            }
        hm = g["home"]["score"] - g["away"]["score"]
        hist[h].append((gd, g["home"]["score"], hm))
        hist[a_].append((gd, g["away"]["score"], -hm))
        last_date[h] = gd
        last_date[a_] = gd

    ml_rows = []
    for eid, br in base_rows.items():
        f = feat_by_eid.get(eid)
        em, mm, fm = elo_map.get(eid), massey.get(eid), eff.get(eid)
        if f is None or em is None or mm is None or fm is None:
            continue
        r = dict(br)
        for key, pm in (("elo_m", em), ("massey_m", mm), ("eff_m", fm)):
            r[key] = -pm["model_margin"] if br["flip"] else pm["model_margin"]
        r["min_games"] = em["min_games"]
        r.update(f)
        if br["flip"]:
            for k in ("form5_diff", "form10_diff", "ppg_diff", "rest_diff"):
                r[k] = -r[k]
        r["market_margin"] = -br["home_line"]
        ml_rows.append(r)
    print("\nML rows:", len(ml_rows))

    s1 = [r for r in ml_rows if r["season"] == "2024-25" and r["min_games"] >= 5]
    s2 = [r for r in ml_rows if r["season"] == "2025-26" and r["min_games"] >= 5]
    feats = ["elo_m", "massey_m", "eff_m", "form5_diff", "form10_diff",
             "ppg_diff", "rest_diff", "neutral"]
    X1 = np.array([[r[k] for k in feats] for r in s1])
    y1 = np.array([r["margin"] for r in s1], dtype=float)
    X2 = np.array([[r[k] for k in feats] for r in s2])

    def report(name, p1, p2, resid=False):
        for label, rs, pr in (("tune", s1, p1), ("HOLDOUT", s2, p2)):
            rows = []
            for r, p in zip(rs, pr):
                edge = float(p) if resid else float(p) + r["home_line"]
                rows.append((r, edge))

            def bets(th):
                return [settle_spread(r, "home" if e > 0 else "away")
                        for r, e in rows if abs(e) >= th]

            print(f"  {name} {label}: " +
                  " | ".join(f"th>={th} {summarize(bets(th))}" for th in (2, 4, 6)))

    print("\n===== ML: predict margin (no market feature) =====")
    for name, model in (("ridge", Ridge(alpha=50.0)),
                        ("gbm", HistGradientBoostingRegressor(
                            max_iter=300, learning_rate=0.05, max_depth=3))):
        model.fit(X1, y1)
        report(name, model.predict(X1), model.predict(X2))

    print("\n===== ML: predict residual vs market (market feature included) =====")
    X1r = np.column_stack([X1, [r["market_margin"] for r in s1]])
    X2r = np.column_stack([X2, [r["market_margin"] for r in s2]])
    y1r = y1 - np.array([r["market_margin"] for r in s1])
    for name, model in (("ridge-resid", Ridge(alpha=50.0)),
                        ("gbm-resid", HistGradientBoostingRegressor(
                            max_iter=300, learning_rate=0.05, max_depth=3))):
        model.fit(X1r, y1r)
        report(name, model.predict(X1r), model.predict(X2r), resid=True)


if __name__ == "__main__":
    main()
