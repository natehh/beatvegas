"""Unit tests for the college public-fade research script's parsers/settlement."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "college_public_fade_research.py"
spec = importlib.util.spec_from_file_location("college_public_fade_research", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Abbreviated copy of a real VegasInsider trends fragment, including the
# malformed unclosed <span>/<div> around the percentage pills.
TRENDS_HTML = """
<div id="trends-component" class="module">
 <header><h2> Siena @ Duke Betting Trends </h2></header>
 <table class="odds-table game-table">
  <tbody id="trends-table-bets--0" class="trends-table-bets--0 active">
   <tr class="header"><th>Final</th><th>Spread</th><th>Total</th><th>Moneyline</th></tr>
   <tr>
    <td class="game-team"><span> 717 </span>
     <a href="/college-basketball/teams/siena/" class="team-name " data-abbr="SIE" aria-label="Siena"> <span> Siena </span> </a></td>
    <td data-auth="member"><div class="mobile-wrap"><span class="small"> 17-9 ATS </span> <span class="pill bold highlight">80%</td> </div> </td>
    <td data-auth="member"><div class="mobile-wrap"><span class="small"> 9-17 O/U </span> <span class="pill bold highlight">92%</td> </div> </td>
    <td data-auth="member"><div class="mobile-wrap"><span class="small"> 23-12 SU </span> <span class="pill bold matte">6%</td> </div> </td>
   </tr>
   <tr>
    <td class="game-team"><span> 718 </span>
     <a href="/college-basketball/teams/duke/" class="team-name " data-abbr="DUKE" aria-label="Duke"> <span> Duke </span> </a></td>
    <td data-auth="member"><div class="mobile-wrap"><span class="small"> 20-10 ATS </span> <span class="pill bold matte">20%</td> </div> </td>
    <td data-auth="member"><div class="mobile-wrap"><span class="small"> 15-15 O/U </span> <span class="pill bold matte">8%</td> </div> </td>
    <td data-auth="member"><div class="mobile-wrap"><span class="small"> 33-2 SU </span> <span class="pill bold highlight">94%</td> </div> </td>
   </tr>
   <tr class="game-odds open"> <td>Open</td> <td> DUKE -27.5 </td> <td> o138.5 </td> <td> -20000 </td> </tr>
   <tr class="game-odds current"> <td>Consensus</td> <td> DUKE -28.5 </td> <td> o137.5 </td> <td> -20000 </td> </tr>
  </tbody>
  <tbody id="trends-table-money--0" class="trends-table-money--0 ">
   <tr>
    <td class="game-team"><span> 717 </span>
     <a href="/college-basketball/teams/siena/" class="team-name " data-abbr="SIE" aria-label="Siena"> <span> Siena </span> </a></td>
    <td data-auth="member"><div class="mobile-wrap"><span class="small"> 17-9 ATS </span> <span class="pill bold highlight">77%</td> </div> </td>
    <td data-auth="member"><div class="mobile-wrap"><span class="small"> 9-17 O/U </span> <span class="pill bold highlight">70%</td> </div> </td>
    <td data-auth="member"><div class="mobile-wrap"><span class="small"> 23-12 SU </span> <span class="pill bold matte">3%</td> </div> </td>
   </tr>
   <tr>
    <td class="game-team"><span> 718 </span>
     <a href="/college-basketball/teams/duke/" class="team-name " data-abbr="DUKE" aria-label="Duke"> <span> Duke </span> </a></td>
    <td data-auth="member"><div class="mobile-wrap"><span class="small"> 20-10 ATS </span> <span class="pill bold matte">23%</td> </div> </td>
    <td data-auth="member"><div class="mobile-wrap"><span class="small"> 15-15 O/U </span> <span class="pill bold matte">30%</td> </div> </td>
    <td data-auth="member"><div class="mobile-wrap"><span class="small"> 33-2 SU </span> <span class="pill bold highlight">97%</td> </div> </td>
   </tr>
  </tbody>
 </table>
</div>
"""


def test_parse_trends_splits_and_lines():
    t = mod.parse_trends(TRENDS_HTML)
    assert t is not None
    assert t["header"] == "Siena @ Duke Betting Trends"
    assert t["bets"][0] == {"team": "Siena", "spread_pct": 80, "total_pct": 92, "ml_pct": 6}
    assert t["bets"][1] == {"team": "Duke", "spread_pct": 20, "total_pct": 8, "ml_pct": 94}
    assert t["money"][0]["spread_pct"] == 77
    assert t["money"][1]["spread_pct"] == 23
    assert t["open"]["spread"] == "DUKE -27.5"
    assert t["consensus"] == {"spread": "DUKE -28.5", "total": "o137.5", "ml": "-20000"}


def test_parse_trends_missing_returns_none():
    assert mod.parse_trends("<html><body>nothing</body></html>") is None


def test_parse_price():
    assert mod.parse_price(("+3.5", "-115")) == (3.5, -115)
    assert mod.parse_price(("-110", None)) == (-110.0, None)
    assert mod.parse_price(("N/A", None)) == (None, None)
    assert mod.parse_price(None) == (None, None)


def _row(**over):
    r = {
        "sport": "ncaaf", "season": "2025", "date": "2025-09-01",
        "away": "A", "home": "B", "away_score": 10, "home_score": 40,
        "away_line": 28.5, "home_line": -28.5,
        "away_price": -110, "home_price": -110,
        "total_line": 55.5,
        "away_bets": 80, "home_bets": 20,
        "away_bets_total": 50, "away_money": None, "home_money": None,
        "margin": 30, "total_pts": 50,
    }
    r.update(over)
    return r


def test_settle_spread_favourite_covers_and_push():
    r = _row()  # home -28.5 wins by 30 -> covers
    assert mod.settle_spread(r, "home") == pytest.approx(100 * 100 / 110)
    assert mod.settle_spread(r, "away") == -100.0
    push = _row(away_line=30.0, home_line=-30.0)
    assert mod.settle_spread(push, "home") == 0.0


def test_fade_band_bets_opposite_of_popular_side():
    r = _row()  # 80% of bets on away -> fade bets home
    band = lambda p: p >= 75  # noqa: E731
    assert mod.fade_bets([r], "away_bets", "home_bets", band) == [
        pytest.approx(100 * 100 / 110)
    ]
    # balanced splits -> no bet
    quiet = _row(away_bets=55, home_bets=45)
    assert mod.fade_bets([quiet], "away_bets", "home_bets", band) == []


def test_fade_uses_each_sides_reported_pct_not_complement():
    # VI occasionally reports pairs that do not sum to 100; both sides in the
    # band means no bet (ambiguous), only one side in band means fade it.
    r = _row(away_bets=84, home_bets=20)
    band = lambda p: p >= 80  # noqa: E731
    assert len(mod.fade_bets([r], "away_bets", "home_bets", band)) == 1
    both = _row(away_bets=84, home_bets=82)
    assert mod.fade_bets([both], "away_bets", "home_bets", band) == []
