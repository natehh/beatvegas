"""Unit tests for the player-props research script (parsing + no-lookahead)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "player_props_research.py"
spec = importlib.util.spec_from_file_location("player_props_research", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

GAMELOG = {
    "events": {
        "401": {
            "gameDate": "2026-01-05T00:00:00.000+00:00",
            "homeTeamScore": "110",
            "awayTeamScore": "100",
            "team": {"id": "25", "abbreviation": "OKC"},
            "opponent": {"id": "24", "abbreviation": "SA"},
            "homeTeamId": "25",
        },
        "402": {  # postseason event: must be excluded
            "gameDate": "2026-05-01T00:00:00.000+00:00",
            "homeTeamScore": "99",
            "awayTeamScore": "98",
            "team": {"id": "25", "abbreviation": "OKC"},
            "opponent": {"id": "24", "abbreviation": "SA"},
            "homeTeamId": "25",
        },
        "403": {  # DNP: zero minutes, excluded
            "gameDate": "2026-01-07T00:00:00.000+00:00",
            "homeTeamScore": "101",
            "awayTeamScore": "99",
            "team": {"id": "25", "abbreviation": "OKC"},
            "opponent": {"id": "24", "abbreviation": "SA"},
            "homeTeamId": "24",
        },
    },
    "seasonTypes": [
        {
            "displayName": "2025-26 Regular Season",
            "categories": [
                {
                    "displayName": "january",
                    "events": [
                        {"eventId": "401", "stats": ["30", "5-9", "55.0", "2-4", "50.0",
                                                      "4-4", "100.0", "6", "5", "1", "2",
                                                      "2", "1", "20"]},
                        {"eventId": "403", "stats": ["0", "0-0", "0.0", "0-0", "0.0",
                                                     "0-0", "0.0", "0", "0", "0", "0",
                                                     "0", "0", "0"]},
                    ],
                }
            ],
        },
        {
            "displayName": "2025-26 Postseason",
            "categories": [
                {"displayName": "Postseason",
                 "events": [{"eventId": "402", "stats": ["30"] * 14}]}
            ],
        },
    ],
}


def test_parse_gamelog_regular_season_only():
    rows = mod.parse_gamelog(GAMELOG, {"id": "1", "name": "Test Player"}, "2025-26")
    assert len(rows) == 1
    r = rows[0]
    assert r["pts"] == 20.0 and r["reb"] == 6.0 and r["ast"] == 5.0
    assert r["fg3m"] == 2.0 and r["min"] == 30.0
    assert r["fga"] == 9.0 and r["fg3a"] == 4.0 and r["fta"] == 4.0 and r["tov"] == 1.0
    assert r["game_total"] == 210.0
    assert r["team"] == "OKC" and r["opp"] == "SA"


def _toy_logs() -> pd.DataFrame:
    rows = []
    for i, pts in enumerate([10.0, 20.0, 30.0, 40.0]):
        rows.append({
            "player_id": "1", "player": "P", "season": "2025-26",
            "event_id": str(i), "date": f"2026-01-{10 + i:02d}",
            "team": "OKC", "opp": "SA", "game_total": 220.0,
            "min": 30.0, "pts": pts, "reb": 5.0, "ast": 5.0,
            "fg3m": 1.0, "stl": 1.0, "blk": 1.0,
        })
    return pd.DataFrame(rows)


def test_pregame_features_use_only_prior_games():
    df = mod.add_pregame_features(_toy_logs())
    # game 3 (index 2): season prior = mean(10, 20); l5 same; no leakage of 30/40
    assert df.loc[2, "pts_season_prior"] == pytest.approx(15.0)
    assert df.loc[3, "pts_season_prior"] == pytest.approx(20.0)
    assert df.loc[0, "games_prior"] == 0
    df2 = mod.add_projections(df)
    # projection for game 3 comes only from games 1-2 (rates are constant
    # here; v2's opponent factor, built from this toy's one-player team
    # totals, and the cs<=2 early-season minutes shrink pull it off 15)
    assert df2.loc[2, "pts_proj"] == pytest.approx(15.0, abs=2.6)
    assert list(df["cs_games_prior"]) == [0, 1, 2, 3]
    # v2 feature block is populated and pregame
    assert df.loc[2, "min_ewm3"] == pytest.approx(30.0)
    assert df.loc[0, "games_prior"] == 0 and df.loc[2, "own_missed_games"] == 0


def test_ml_projections_fallback_without_prior_season():
    df = mod.add_projections(mod.add_pregame_features(_toy_logs()))
    out = mod.add_ml_projections(df)
    # single-season toy: no prior-season training data, heuristic kept
    assert (out["pts_proj"].fillna(-1) == out["pts_proj_heur"].fillna(-1)).all()
    assert (out["min_proj"].fillna(-1) == out["min_proj_heur"].fillna(-1)).all()
