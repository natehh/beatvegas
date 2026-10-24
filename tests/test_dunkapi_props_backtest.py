"""Unit tests for the DunkAPI props backtest (no network, no credits)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dunkapi_props_backtest.py"
spec = importlib.util.spec_from_file_location("dunkapi_props_backtest", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _prop(book, line, over=-110, under=-110, market="player_points"):
    return {
        "game_id": "401", "game_date": "2025-10-25", "player_name": "P",
        "player_id": "1", "market": market, "bookmaker": book,
        "line": line, "over_odds": over, "under_odds": under,
        "captured_at": "2025-10-25T22:00:00Z", "source": "archive",
    }


def test_consensus_uses_median_line_and_its_book_odds():
    df = mod.consensus_props([
        _prop("a", 20.5, over=-120, under=100),
        _prop("b", 21.5, over=-110, under=-110),
        _prop("c", 24.5, over=100, under=-120),
        _prop("a", 99.5, market="player_fouls"),  # unmapped market dropped
    ])
    assert len(df) == 1
    r = df.iloc[0]
    assert r["line"] == 21.5 and r["bookmaker"] == "b"
    assert r["over_odds"] == -110 and r["n_books"] == 3
    assert r["line_min"] == 20.5 and r["line_max"] == 24.5
    assert r["stat"] == "pts"


def test_settle_with_odds_american_prices():
    assert mod.settle_with_odds(25, 20.5, "over", -110) == pytest.approx(100 * 100 / 110)
    assert mod.settle_with_odds(25, 20.5, "over", 150) == pytest.approx(150.0)
    assert mod.settle_with_odds(15, 20.5, "over", -110) == -100.0
    assert mod.settle_with_odds(15, 20.5, "under", -110) == pytest.approx(100 * 100 / 110)
    assert mod.settle_with_odds(20, 20.0, "over", -110) == 0.0  # push


class _FakeClient:
    def __init__(self, credits):
        self.credits = credits
        self.pulled = []

    def estimate_player_props(self, pid, frm, to):
        return self.credits

    def get_player_props(self, pid, frm, to):  # pragma: no cover - guarded below
        self.pulled.append(pid)
        return {"props": [_prop("a", 20.5)]}


def test_fetch_props_budget_guard_spends_nothing_over_budget(tmp_path):
    client = _FakeClient(credits=500)
    with pytest.raises(mod.BudgetExceeded):
        mod.fetch_props({"P": "1"}, "2025-10-21", "2025-10-31", client=client,
                        cache_dir=tmp_path, budget_credits=400, log=lambda *_: None)
    assert client.pulled == []


def test_fetch_props_estimate_only_never_pulls(tmp_path):
    client = _FakeClient(credits=50)
    rows = mod.fetch_props({"P": "1"}, "2025-10-21", "2025-10-31", client=client,
                           cache_dir=tmp_path, estimate_only=True, log=lambda *_: None)
    assert rows == [] and client.pulled == []
    assert list(tmp_path.iterdir()) == []


def test_fetch_props_cache_hit_costs_zero(tmp_path):
    path = mod.cache_path(tmp_path, "1", "2025-10-21", "2025-10-31")
    path.write_text('{"props": []}')
    # client=None would raise if any uncached pull were attempted
    rows = mod.fetch_props({"P": "1"}, "2025-10-21", "2025-10-31", client=None,
                           cache_dir=tmp_path, log=lambda *_: None)
    assert rows == []


def test_ev_bets_only_bet_when_probability_beats_price():
    graded = pd.DataFrame([
        # proj far above line at -110: model says over is +EV, and it hits
        {"actual": 30.0, "line": 20.5, "proj": 28.0, "resid_sd": 5.0,
         "over_odds": -110, "under_odds": -110},
        # proj == line at -110 both sides: no +EV side, never bet
        {"actual": 30.0, "line": 20.5, "proj": 20.5, "resid_sd": 5.0,
         "over_odds": -110, "under_odds": -110},
    ])
    profits = mod.ev_bets(graded, min_ev=0.0)
    assert len(profits) == 1 and profits[0] > 0
    assert mod.ev_bets(graded, min_ev=10_000.0) == []


def test_grade_joins_on_player_and_event_and_maps_stat():
    consensus = pd.DataFrame([{
        "player_id": "1", "game_id": "401", "game_date": "2025-10-25",
        "player": "P", "market": "player_points", "stat": "pts", "line": 20.5,
        "over_odds": -110, "under_odds": -110, "bookmaker": "b", "n_books": 1,
        "line_min": 20.5, "line_max": 20.5,
    }])
    logs = pd.DataFrame([{
        "player_id": "1", "event_id": "401", "date": "2025-10-25",
        "season": "2025-26", "games_prior": 50, "min_proj": 32.0,
        "pts": 25.0, "reb": 5.0, "ast": 5.0, "pra": 35.0, "fg3m": 2.0,
        "pts_proj": 23.0, "reb_proj": 5.0, "ast_proj": 5.0,
        "pra_proj": 33.0, "fg3m_proj": 2.0,
    }])
    graded = mod.grade(consensus, logs, {c: 5.0 for c in mod.STATS})
    assert len(graded) == 1
    assert graded.iloc[0]["actual"] == 25.0 and graded.iloc[0]["proj"] == 23.0
    assert graded.iloc[0]["edge"] == pytest.approx(2.5)
