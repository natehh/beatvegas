"""Unit tests for the PropLine/Odds-API source switch in prop_odds (no network)."""

from __future__ import annotations

import pytest

from sports.nba import prop_odds


class _Resp:
    headers: dict = {}

    def raise_for_status(self):
        pass

    def json(self):
        return []


def _capture_get(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return _Resp()

    monkeypatch.setattr(prop_odds.requests, "get", fake_get)
    return calls


def test_get_api_key_propline(monkeypatch):
    monkeypatch.setenv("PROPLINE_API_KEY", "pl-key")
    monkeypatch.setenv("THE_ODDS_API_KEY", "oa-key")
    assert prop_odds.get_api_key(source="propline") == "pl-key"
    assert prop_odds.get_api_key(source="oddsapi") == "oa-key"
    monkeypatch.delenv("PROPLINE_API_KEY")
    with pytest.raises(RuntimeError, match="PROPLINE_API_KEY"):
        prop_odds.get_api_key(source="propline")


def test_propline_requests_use_header_auth(monkeypatch):
    calls = _capture_get(monkeypatch)
    prop_odds.list_nba_events("pl-key", source="propline")
    prop_odds.fetch_event_props("pl-key", "evt1", markets=("player_points",), source="propline")
    (events_url, events_kwargs), (odds_url, odds_kwargs) = calls
    assert events_url == "https://api.prop-line.com/v1/sports/basketball_nba/events"
    assert events_kwargs["headers"] == {"X-API-Key": "pl-key"}
    assert odds_url.endswith("/sports/basketball_nba/events/evt1/odds")
    assert odds_kwargs["headers"] == {"X-API-Key": "pl-key"}
    assert odds_kwargs["params"] == {"markets": "player_points"}


def test_oddsapi_requests_use_query_key(monkeypatch):
    calls = _capture_get(monkeypatch)
    prop_odds.list_nba_events("oa-key")
    prop_odds.fetch_event_props("oa-key", "evt1", markets=("player_points",))
    (events_url, events_kwargs), (odds_url, odds_kwargs) = calls
    assert events_url == "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    assert events_kwargs["params"] == {"apiKey": "oa-key"}
    assert odds_kwargs["params"]["apiKey"] == "oa-key"
    assert odds_kwargs["params"]["regions"] == "us"


def test_flatten_tags_quote_source():
    payload = {
        "id": "e1",
        "commence_time": "2026-10-25T23:00:00Z",
        "home_team": "Boston Celtics",
        "away_team": "New York Knicks",
        "bookmakers": [{
            "key": "draftkings",
            "markets": [{
                "key": "player_points",
                "outcomes": [
                    {"name": "Over", "description": "Jaylen Brown", "point": 24.5, "price": -110},
                    {"name": "Under", "description": "Jaylen Brown", "point": 24.5, "price": -110},
                ],
            }],
        }],
    }
    quotes = prop_odds.flatten_event_quotes(
        payload, snapshot_at="2026-10-25T15:00:00Z", source="propline"
    )
    assert len(quotes) == 1
    assert quotes[0].source == "propline"
    assert quotes[0].player == "Jaylen Brown" and quotes[0].line == 24.5
