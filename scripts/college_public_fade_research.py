#!/usr/bin/env python3
"""Does the NBA public-fade edge transfer to college basketball / football?

Data (free, no API key):
  * VegasInsider historical odds pages enumerate games (dated pages for
    NCAAB, weekly pages for NCAAF) with per-book + consensus lines.
  * Each game's VegasInsider "betting trends" fragment reports the share of
    bets (tickets) and money on each side of the spread/total/moneyline,
    plus opening and consensus (closing) lines. VI's splits were verified
    against Yahoo's NBA splits on identical games (66% vs 66.2%, 71% vs
    70.9%, ...) — same feed family as the NBA strategy in
    scripts/walkforward_research.py.
  * Final scores come from ESPN's public scoreboard API, joined on
    date + team names with home/away orientation verification.

Coverage is limited by VegasInsider's archive: NCAAB from ~2024-11-18
(2024-25 partial + 2025-26), NCAAF 2025 + partial 2026. Games against
non-Division-I opponents (absent from ESPN's D1 feed) are excluded.

Headline results (flat $100 bets at VI consensus line/price):
  * NCAAB: fading the public LOSES in every band tested, both seasons
    (NBA's 65-72% band: -4.2% / -4.5% ROI, n~1.6k). Riding the public is
    also negative. No college-basketball edge here.
  * NCAAF: fading sides with >=80% of bets was +10.5% (n=196, positive in
    both seasons), but the sample is small (bootstrap 95% CI includes 0)
    and the effect is mostly "lay big points with huge favourites": in
    mismatch games the public takes the underdog +30, so the fade is
    usually the -30 favourite. Unconditional big-favourite betting was
    also positive. Treat as a lead for forward tracking, NOT a locked
    strategy — hence no preset in core/defaults.py.

Usage:
    python scripts/college_public_fade_research.py            # scrape + analyze
    python scripts/college_public_fade_research.py --analyze-only
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "cache" / "college_vi"
BASE = "https://www.vegasinsider.com"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) beatvegas-research"}

NCAAB_RANGES = [
    ("2024-25", date(2024, 11, 18), date(2025, 4, 8)),
    ("2025-26", date(2025, 11, 3), date(2026, 4, 6)),
]
NCAAF_SEASONS = [(2025, range(1, 17)), (2026, range(1, 10))]
CUTOFF_2026 = "2026-10-16"  # last fully-played date when the study was run

_cache_dir = DEFAULT_CACHE


def fetch(url: str, cache_name: str, retries: int = 3) -> str | None:
    path = _cache_dir / cache_name
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as resp:
                text = resp.read().decode("utf-8", errors="ignore")
            path.write_text(text, encoding="utf-8")
            time.sleep(0.05)
            return text
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    print(f"FETCH FAIL {url}: {last}")
    return None


# ------------------------------------------------------------- VI page parsing
def parse_ld_events(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    events = []
    for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            d = json.loads(s.string or "")
        except Exception:
            continue
        if isinstance(d, dict) and d.get("@type") == "SportsEvent":
            events.append(d)
    return events


def parse_odds_table(html: str) -> list[dict]:
    """Spread section of a VI odds page: away/home rows in event order."""
    soup = BeautifulSoup(html, "html.parser")
    tbody = soup.find("tbody", id=re.compile(r"odds-table-spread"))
    if tbody is None:
        return []
    rows: list[dict] = []
    for tr in tbody.find_all("tr"):
        team_cell = tr.find("td", class_="game-team")
        if team_cell is None:
            continue
        name_el = team_cell.find(class_="team-name")
        team = name_el.get_text(" ", strip=True) if name_el else ""
        abbr = name_el.get("data-abbr") if name_el is not None else None
        vals = []
        for c in tr.find_all("td", class_="game-odds"):
            dv = c.find(class_="data-value")
            do = c.find(class_="data-odds")
            vals.append(
                (
                    dv.get_text(" ", strip=True) if dv else c.get_text(" ", strip=True),
                    do.get_text(" ", strip=True) if do else None,
                )
            )
        consensus = next((v for v in reversed(vals) if v[0]), None)
        rows.append({"team": team, "abbr": abbr, "consensus": consensus})
    return [{"away": rows[i], "home": rows[i + 1]} for i in range(0, len(rows) - 1, 2)]


_TEAM_NAME = re.compile(r'class="team-name[^"]*"[^>]*>\s*<span>\s*([^<]+?)\s*</span>')
_PILL = re.compile(r'class="pill[^"]*">\s*(\d{1,3})%')


def _tbody_chunk(html: str, kind: str) -> str:
    m = re.search(rf'id="trends-table-{kind}--\d+"', html)
    if not m:
        return ""
    end = html.find("</tbody>", m.start())
    return html[m.start() : end if end > 0 else len(html)]


def _parse_trends_table(chunk: str) -> list[dict]:
    rows = []
    for part in chunk.split('<td class="game-team">')[1:]:
        part = part.split('<tr class="game-odds')[0]
        name_m = _TEAM_NAME.search(part)
        pcts = [int(x) for x in _PILL.findall(part)]
        while len(pcts) < 3:
            pcts.append(None)
        rows.append(
            {
                "team": name_m.group(1).strip() if name_m else "",
                "spread_pct": pcts[0],
                "total_pct": pcts[1],
                "ml_pct": pcts[2],
            }
        )
        if len(rows) == 2:
            break
    return rows


def _parse_line_row(chunk: str, css: str) -> dict:
    m = re.search(rf'<tr class="game-odds {css}">(.*?)</tr>', chunk, re.S)
    if not m:
        return {"spread": None, "total": None, "ml": None}
    cells = re.findall(r"<td[^>]*>(.*?)</td>", m.group(1), re.S)
    cells = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c)).strip() for c in cells]
    vals = (cells[1:4] + [None, None, None])[:3]
    return {"spread": vals[0], "total": vals[1], "ml": vals[2]}


def parse_trends(html: str) -> dict | None:
    """Parse a VI event trends fragment (Bets%/Money% + Open/Consensus lines)."""
    bets_chunk = _tbody_chunk(html, "bets")
    if not bets_chunk:
        return None
    money_chunk = _tbody_chunk(html, "money")
    header_m = re.search(r"<h2>\s*(.*?)\s*</h2>", html, re.S)
    return {
        "header": re.sub(r"\s+", " ", header_m.group(1)).strip() if header_m else None,
        "bets": _parse_trends_table(bets_chunk),
        "money": _parse_trends_table(money_chunk) if money_chunk else [],
        "open": _parse_line_row(bets_chunk, "open"),
        "consensus": _parse_line_row(bets_chunk, "current"),
    }


# ------------------------------------------------------------------ ESPN join
def norm(s) -> str:
    if not s:
        return ""
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\bst\b", "saint", s)
    return re.sub(r"\s+", " ", s).strip()


def espn_scoreboard(sport_path: str, params: str, cache_name: str) -> list[dict]:
    url = f"https://site.api.espn.com/apis/site/v2/sports/{sport_path}/scoreboard?{params}"
    text = fetch(url, cache_name)
    if text is None:
        return []
    try:
        d = json.loads(text)
    except Exception:
        return []
    games = []
    for ev in d.get("events", []):
        try:
            comp = ev["competitions"][0]
            teams = {}
            for c in comp["competitors"]:
                t = c["team"]
                teams[c["homeAway"]] = {
                    "location": (t.get("location") or "").strip(),
                    "abbr": t.get("abbreviation"),
                    "score": int(c["score"]) if c.get("score") not in (None, "") else None,
                }
            games.append(
                {"espn_id": ev.get("id"), "date": ev.get("date"),
                 "away": teams.get("away"), "home": teams.get("home")}
            )
        except Exception:
            continue
    return games


def _collect_games() -> list[dict]:
    games: dict[int, dict] = {}

    def add(ev_list, table, **extra):
        for i, ev in enumerate(ev_list):
            row = table[i] if i < len(table) else None
            g = {
                "event_id": int(ev["identifier"]),
                "start": ev.get("startDate"),
                "away": row["away"]["team"] if row else None,
                "home": row["home"]["team"] if row else None,
                "away_abbr": row["away"]["abbr"] if row else None,
                "home_abbr": row["home"]["abbr"] if row else None,
                "away_consensus": row["away"]["consensus"] if row else None,
                "home_consensus": row["home"]["consensus"] if row else None,
                **extra,
            }
            games.setdefault(g["event_id"], g)

    for label, d1, d2 in NCAAB_RANGES:
        d = d1
        while d <= d2:
            html = fetch(f"{BASE}/college-basketball/odds/las-vegas/?date={d.isoformat()}",
                         f"ncaab_odds_{d.isoformat()}.html")
            if html and "SportsEvent" in html:
                add(parse_ld_events(html), parse_odds_table(html),
                    sport="ncaab", season_label=label)
            d += timedelta(days=1)
        print(f"ncaab {label} enumerated ({len(games)} total)", flush=True)

    for season, weeks in NCAAF_SEASONS:
        for w in weeks:
            html = fetch(f"{BASE}/college-football/odds/las-vegas/?week={season}-reg-{w}",
                         f"ncaaf_odds_{season}_reg_{w}.html")
            if html and "SportsEvent" in html:
                tbl = parse_odds_table(html)
                all_evs = parse_ld_events(html)
                # keep only games already played; table order matches LD order
                idx = [i for i, e in enumerate(all_evs)
                       if (e.get("startDate") or "")[:10] <= CUTOFF_2026]
                keep = [all_evs[i] for i in idx]
                sub_table = [tbl[i] for i in idx if i < len(tbl)]
                add(keep, sub_table, sport="ncaaf", season_label=str(season))
        print(f"ncaaf {season} enumerated ({len(games)} total)", flush=True)

    return list(games.values())


def _attach_trends(games: list[dict]) -> None:
    for sport, league in [("ncaab", "college-basketball"), ("ncaaf", "college-football")]:
        ids = [g["event_id"] for g in games if g["sport"] == sport]

        def one(eid: int):
            text = fetch(f"{BASE}/{league}/events/{eid}/trends/",
                         f"trends_{league}_{eid}.html")
            return eid, (parse_trends(text) if text else None)

        with ThreadPoolExecutor(max_workers=6) as ex:
            res = dict(ex.map(one, ids))
        ok = 0
        for g in games:
            if g["sport"] == sport:
                g["trends"] = res.get(g["event_id"])
                if g["trends"] and len(g["trends"].get("bets") or []) == 2:
                    ok += 1
        print(f"{sport} trends ok: {ok}/{len(ids)}", flush=True)


def _attach_scores(games: list[dict]) -> None:
    espn_games: list[dict] = []
    for _label, d1, d2 in NCAAB_RANGES:
        d = d1
        while d <= d2:
            espn_games += [
                g for g in espn_scoreboard(
                    "basketball/mens-college-basketball",
                    f"dates={d:%Y%m%d}&groups=50&limit=400",
                    f"espn_ncaab_{d:%Y%m%d}.json",
                )
                if g["away"] and g["home"] and g["away"]["score"] is not None
            ]
            d += timedelta(days=1)
    for season, weeks in NCAAF_SEASONS:
        for w in weeks:
            espn_games += [
                g for g in espn_scoreboard(
                    "football/college-football",
                    f"dates={season}&seasontype=2&week={w}&groups=80&limit=400",
                    f"espn_ncaaf_{season}_w{w}.json",
                )
                if g["away"] and g["home"] and g["away"]["score"] is not None
            ]

    by_name: dict[str, list[dict]] = {}
    by_abbr: dict[str, list[dict]] = {}
    by_id: dict[str, dict] = {}
    for g in espn_games:
        by_name.setdefault(
            "|".join(sorted((norm(g["away"]["location"]), norm(g["home"]["location"])))),
            [],
        ).append(g)
        by_abbr.setdefault(
            "|".join(sorted(((g["away"]["abbr"] or "").lower(),
                             (g["home"]["abbr"] or "").lower()))),
            [],
        ).append(g)
        by_id[str(g["espn_id"])] = g

    def near(cands, gd):
        for c in cands:
            try:
                if abs((date.fromisoformat((c["date"] or "")[:10]) - gd).days) <= 2:
                    return c
            except Exception:
                continue
        return None

    def variants(n):
        n = norm(n)
        return {n, re.sub(r" state$", "", n), n + " state"}

    matched = 0
    for g in games:
        gd = date.fromisoformat((g.get("start") or "")[:10])
        hit = near(by_name.get(
            "|".join(sorted((norm(g.get("away")), norm(g.get("home"))))), []), gd)
        if hit is None:
            hit = near(by_abbr.get(
                "|".join(sorted(((g.get("away_abbr") or "").lower(),
                                 (g.get("home_abbr") or "").lower()))), []), gd)
        if hit is None:
            for a in variants(g.get("away")):
                for h in variants(g.get("home")):
                    hit = near(by_name.get("|".join(sorted((a, h))), []), gd)
                    if hit:
                        break
                if hit:
                    break
        if hit:
            g["away_score"] = hit["away"]["score"]
            g["home_score"] = hit["home"]["score"]
            g["espn_id"] = hit["espn_id"]
            matched += 1
    print(f"espn matched: {matched}/{len(games)}")

    # orientation check: ESPN home/away designation can flip at neutral sites
    def close(a, b):
        return a == b or a in b or b in a or \
            re.sub(r" state$", "", a) == re.sub(r" state$", "", b)

    flips = unknown = 0
    for g in games:
        if g.get("espn_id") is None or g.get("away_score") is None:
            continue
        eg = by_id.get(str(g["espn_id"]))
        if not eg:
            continue
        ea, eh = norm(eg["away"]["location"]), norm(eg["home"]["location"])
        va, vh = norm(g.get("away")), norm(g.get("home"))
        if close(ea, va) and close(eh, vh):
            pass
        elif close(ea, vh) and close(eh, va):
            g["away_score"], g["home_score"] = g["home_score"], g["away_score"]
            flips += 1
        else:
            g["orientation_unknown"] = True
            unknown += 1
    print(f"orientation flips fixed: {flips}, unknown (excluded): {unknown}")


# ------------------------------------------------------------------ settlement
def parse_price(cell) -> tuple[float | None, int | None]:
    if not cell:
        return None, None
    val, odds = cell
    try:
        line = float(str(val).replace("½", ".5"))
    except Exception:
        line = None
    try:
        price = int(str(odds).replace("+", "")) if odds else None
    except Exception:
        price = None
    return line, price


def payout(price: int | None, stake: float = 100.0) -> float:
    if price is None:
        price = -110
    return stake * (100.0 / abs(price)) if price < 0 else stake * price / 100.0


def build_rows(games: list[dict]) -> list[dict]:
    rows = []
    for g in games:
        t = g.get("trends")
        if not t or len(t.get("bets") or []) != 2:
            continue
        if g.get("away_score") is None or g.get("home_score") is None:
            continue
        if g.get("orientation_unknown"):
            continue
        bets, money = t["bets"], t.get("money") or []
        if norm(bets[0]["team"]) != norm(g.get("away")) or \
                norm(bets[1]["team"]) != norm(g.get("home")):
            continue
        away_line, away_price = parse_price(g.get("away_consensus"))
        home_line, home_price = parse_price(g.get("home_consensus"))
        if away_line is None or home_line is None or abs(away_line + home_line) > 0.51:
            continue
        cons = (t.get("consensus") or {}).get("total") or ""
        m = re.search(r"([ou])\s*(\d+(?:\.\d+)?)", cons)
        rows.append(
            {
                "sport": g["sport"],
                "season": g.get("season_label"),
                "date": (g.get("start") or "")[:10],
                "away": g.get("away"),
                "home": g.get("home"),
                "away_score": g["away_score"],
                "home_score": g["home_score"],
                "away_line": away_line,
                "home_line": home_line,
                "away_price": away_price,
                "home_price": home_price,
                "total_line": float(m.group(2)) if m else None,
                "away_bets": bets[0]["spread_pct"],
                "home_bets": bets[1]["spread_pct"],
                "away_bets_total": bets[0]["total_pct"],
                "away_money": money[0]["spread_pct"] if len(money) == 2 else None,
                "home_money": money[1]["spread_pct"] if len(money) == 2 else None,
                "margin": g["home_score"] - g["away_score"],
                "total_pts": g["home_score"] + g["away_score"],
            }
        )
    return rows


def settle_spread(r: dict, side: str) -> float:
    if side == "home":
        diff, price = r["margin"] + r["home_line"], r["home_price"]
    else:
        diff, price = -r["margin"] + r["away_line"], r["away_price"]
    return payout(price) if diff > 0 else (-100.0 if diff < 0 else 0.0)


def settle_total(r: dict, side: str) -> float:
    if r["total_line"] is None:
        return 0.0
    diff = r["total_pts"] - r["total_line"]
    if side == "under":
        diff = -diff
    return payout(-110) if diff > 0 else (-100.0 if diff < 0 else 0.0)


def summarize(bets: list[float]) -> str:
    n = len(bets)
    if n == 0:
        return "n=0"
    wins = sum(1 for b in bets if b > 0)
    pushes = sum(1 for b in bets if b == 0)
    profit = sum(bets)
    roi = profit / (100.0 * n) * 100
    mean = profit / n
    var = sum((b - mean) ** 2 for b in bets) / max(n - 1, 1)
    se = math.sqrt(var / n)
    hit = wins / (n - pushes) * 100 if n > pushes else 0.0
    return f"n={n:5d} hit={hit:5.1f}% ROI={roi:+6.2f}% (±{se:.1f}) profit={profit:+9.0f}"


BANDS = {
    "65-72 (NBA band)": lambda p: 65 <= p < 72,
    ">=65 all": lambda p: p >= 65,
    "60-70": lambda p: 60 <= p < 70,
    "65-75": lambda p: 65 <= p < 75,
    ">=70": lambda p: p >= 70,
    "70-80": lambda p: 70 <= p < 80,
    ">=75": lambda p: p >= 75,
    ">=80": lambda p: p >= 80,
    "55-65 (mild)": lambda p: 55 <= p < 65,
}


def fade_bets(rows, key_a, key_h, in_band, market="spread"):
    out = []
    for r in rows:
        pa, ph = r.get(key_a), r.get(key_h)
        if pa is None or ph is None:
            continue
        if market == "spread":
            if in_band(pa) and not in_band(ph):
                out.append(settle_spread(r, "home"))
            elif in_band(ph) and not in_band(pa):
                out.append(settle_spread(r, "away"))
        else:
            if in_band(pa):
                out.append(settle_total(r, "under"))
            elif in_band(100 - pa):
                out.append(settle_total(r, "over"))
    return out


def analyze(rows: list[dict]) -> None:
    from collections import Counter

    print(f"\nsettled games with splits+lines+scores: {len(rows)}")
    print(Counter((r["sport"], r["season"]) for r in rows))

    for sport in ["ncaab", "ncaaf"]:
        srows = [r for r in rows if r["sport"] == sport]
        seasons = sorted({r["season"] for r in srows})
        print(f"\n===== {sport.upper()} spread fade (bets%) =====")
        for band_name, in_band in BANDS.items():
            all_b: list[float] = []
            per = []
            for s in seasons:
                b = fade_bets([r for r in srows if r["season"] == s],
                              "away_bets", "home_bets", in_band)
                all_b += b
                per.append(f"{s}: {summarize(b)}")
            print(f"-- {band_name:18s} ALL: {summarize(all_b)}")
            for p in per:
                print(f"     {p}")
        print(f"\n===== {sport.upper()} spread fade (money%) =====")
        for band_name, in_band in BANDS.items():
            b = fade_bets(srows, "away_money", "home_money", in_band)
            print(f"-- {band_name:18s} {summarize(b)}")

    print("\n===== NCAAB totals fade (over bets%) =====")
    srows = [r for r in rows if r["sport"] == "ncaab"]
    for band_name, in_band in BANDS.items():
        b = fade_bets(srows, "away_bets_total", "away_bets_total", in_band, market="total")
        print(f"-- {band_name:18s} {summarize(b)}")

    # Controls: is any splits effect just "big favourites/dogs cover anyway"?
    print("\n===== Control: unconditional dog/favourite by |spread| =====")
    for sport in ["ncaab", "ncaaf"]:
        srows = [r for r in rows if r["sport"] == sport]
        print(f"-- {sport.upper()}")
        for lab, lo, hi in [("7-13.5", 7, 14), ("14-20.5", 14, 21), (">=21", 21, 999)]:
            seg = [r for r in srows if lo <= abs(r["home_line"]) < hi]
            dog = [settle_spread(r, "home" if r["home_line"] > 0 else "away")
                   for r in seg if r["home_line"] != 0]
            fav = [settle_spread(r, "away" if r["home_line"] > 0 else "home")
                   for r in seg if r["home_line"] != 0]
            print(f"   |spread| {lab:8s} bet-dog: {summarize(dog)} | bet-fav: {summarize(fav)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--analyze-only", action="store_true",
                    help="skip scraping; analyze the cached dataset JSON")
    args = ap.parse_args()

    global _cache_dir
    _cache_dir = args.cache_dir
    _cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = _cache_dir / "dataset.json"

    if args.analyze_only and dataset_path.exists():
        games = json.loads(dataset_path.read_text())
    else:
        games = _collect_games()
        _attach_trends(games)
        _attach_scores(games)
        dataset_path.write_text(json.dumps(games), encoding="utf-8")
        print(f"wrote {dataset_path}")

    analyze(build_rows(games))


if __name__ == "__main__":
    main()
