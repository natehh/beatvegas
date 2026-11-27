"""Unit tests for forward-prop consensus + settled grading (no network)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "grade_forward_props.py"
spec = importlib.util.spec_from_file_location("grade_forward_props", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _quote(book, line, over=-110, under=-110, player="Test Player"):
    return {
        "event_id": "e1", "commence_time": "2026-10-24T23:00:00Z",
        "home_abbr": "BOS", "away_abbr": "NYK", "bookmaker": book,
        "market": "player_points", "player": player, "line": line,
        "over_odds": over, "under_odds": under,
    }


def test_consensus_forward_median_and_accents():
    df = mod.consensus_forward([
        _quote("a", 20.5), _quote("b", 21.5, over=-105), _quote("c", 24.5),
        _quote("a", 5.5, player="Luka Dončić"),
    ])
    assert len(df) == 2
    r = df[df["player_norm"] == "test player"].iloc[0]
    assert r["line"] == 21.5 and r["n_books"] == 3 and r["stat"] == "pts"
    # accent-insensitive name normalisation matches gamelog names
    assert "luka doncic" in set(df["player_norm"])


def test_consensus_excludes_dfs_books_when_traditional_present():
    df = mod.consensus_forward([
        _quote("prizepicks", 21.5, over=100, under=100),
        _quote("fanduel", 20.5, over=-115, under=-105),
    ])
    assert len(df) == 1
    r = df.iloc[0]
    assert r["bookmaker"] == "fanduel" and r["line"] == 20.5 and r["over_odds"] == -115


def test_grade_settled_scores_against_actuals(tmp_path):
    picks = pd.DataFrame([{
        "event_id": "e1", "player_norm": "test player", "market": "player_points",
        "stat": "pts", "line": 20.5, "proj": 25.0, "edge": 4.5,
        "over_odds": -110, "under_odds": -110, "actual": 0.0,
    }])
    settled = tmp_path / "prop_settled.jsonl"
    settled.write_text(json.dumps({
        "event_id": "e1", "player": "Test Player", "market": "player_points",
        "bookmaker": "a", "line": 20.5, "actual": 28.0,
    }) + "\n")
    out = mod.grade_settled(picks, settled)
    assert out["n_settled_matched"] == 1
    assert out["by_stat"]["pts"]["roi_point_edge"] > 0
    assert out["by_stat"]["pts"]["model_mae"] == 3.0


def test_grade_settled_missing_ledger_is_graceful(tmp_path):
    out = mod.grade_settled(pd.DataFrame(), tmp_path / "nope.jsonl")
    assert out["n_settled"] == 0
