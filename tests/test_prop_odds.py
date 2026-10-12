"""Tests for prop odds flattening and settlement helpers."""

from sports.nba.prop_odds import flatten_event_quotes, season_from_commence, team_to_abbr
from sports.nba.prop_settle import _grade, latest_quotes_before_tip


def test_team_to_abbr_and_season():
    assert team_to_abbr("Boston Celtics") == "BOS"
    assert season_from_commence("2025-10-22T23:30:00Z") == "2025-26"
    assert season_from_commence("2026-01-15T00:00:00Z") == "2025-26"


def test_flatten_event_quotes_pairs_over_under():
    payload = {
        "id": "abc123",
        "commence_time": "2025-11-01T00:00:00Z",
        "home_team": "Boston Celtics",
        "away_team": "New York Knicks",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "player_points",
                        "outcomes": [
                            {
                                "name": "Over",
                                "description": "Jayson Tatum",
                                "price": -110,
                                "point": 27.5,
                            },
                            {
                                "name": "Under",
                                "description": "Jayson Tatum",
                                "price": -110,
                                "point": 27.5,
                            },
                        ],
                    }
                ],
            }
        ],
    }
    quotes = flatten_event_quotes(payload, snapshot_at="2025-10-31T18:00:00Z")
    assert len(quotes) == 1
    q = quotes[0]
    assert q.player == "Jayson Tatum"
    assert q.line == 27.5
    assert q.home_abbr == "BOS"
    assert q.away_abbr == "NYK"
    assert q.over_odds == -110
    assert q.under_odds == -110
    assert q.market == "player_points"


def test_latest_quotes_before_tip_and_grade():
    rows = [
        {
            "event_id": "e1",
            "bookmaker": "dk",
            "market": "player_points",
            "player": "A Player",
            "line": 20.5,
            "commence_time": "2025-11-01T00:00:00Z",
            "snapshot_at": "2025-10-31T12:00:00Z",
            "over_odds": -110,
        },
        {
            "event_id": "e1",
            "bookmaker": "dk",
            "market": "player_points",
            "player": "A Player",
            "line": 20.5,
            "commence_time": "2025-11-01T00:00:00Z",
            "snapshot_at": "2025-10-31T20:00:00Z",
            "over_odds": -105,
        },
        {
            "event_id": "e1",
            "bookmaker": "dk",
            "market": "player_points",
            "player": "A Player",
            "line": 20.5,
            "commence_time": "2025-11-01T00:00:00Z",
            "snapshot_at": "2025-11-01T01:00:00Z",  # after tip — ignore
            "over_odds": 100,
        },
    ]
    latest = latest_quotes_before_tip(rows)
    assert len(latest) == 1
    assert latest[0]["over_odds"] == -105
    assert _grade(25, 20.5) == (True, False)
    assert _grade(20.5, 20.5) == (None, None)
