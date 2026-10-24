"""Unit tests for the daily props pipeline orchestration (no network)."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "daily_prop_pipeline.py"
spec = importlib.util.spec_from_file_location("daily_prop_pipeline", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@dataclass
class _FakeCollectResult:
    snapshot_at: str = "2026-10-24T21:00:00Z"
    n_events: int = 7
    n_quotes: int = 900
    credits_remaining: Optional[int] = 400
    credits_used: Optional[int] = 100
    credits_last: Optional[int] = 3
    ledger_path: Path = Path("data/props/prop_quotes.jsonl")
    event_ids: list = None


def _install_fakes(monkeypatch, calls, *, collect_credits=400, collect_raises=False):
    def fake_settle(*, only_finished_hours):
        calls.append(("settle", only_finished_hours))
        return {"n_settled": 12}

    def fake_collect(**kwargs):
        calls.append(("collect", kwargs))
        if collect_raises:
            raise RuntimeError("Missing THE_ODDS_API_KEY")
        result = _FakeCollectResult(credits_remaining=collect_credits)
        result.event_ids = ["e1"]
        return result

    monkeypatch.setattr(mod, "settle_prop_quotes", fake_settle)
    monkeypatch.setattr(mod, "collect_prop_odds", fake_collect)


def test_pipeline_settles_then_collects_tonight(monkeypatch):
    calls = []
    _install_fakes(monkeypatch, calls)
    summary, code = mod.run_pipeline()
    assert code == 0
    assert [c[0] for c in calls] == ["settle", "collect"]  # settle first
    collect_kwargs = calls[1][1]
    assert collect_kwargs["include_live"] is False
    assert collect_kwargs["hours_ahead"] == 18.0
    assert summary["settle"]["n_settled"] == 12
    assert summary["collect"]["n_quotes"] == 900
    assert "warning" not in summary["collect"]


def test_pipeline_warns_on_low_credits(monkeypatch):
    calls = []
    _install_fakes(monkeypatch, calls, collect_credits=20)
    summary, code = mod.run_pipeline()
    assert code == 0
    assert "warning" in summary["collect"]


def test_pipeline_collect_failure_keeps_settlement(monkeypatch):
    calls = []
    _install_fakes(monkeypatch, calls, collect_raises=True)
    summary, code = mod.run_pipeline()
    assert code == 1
    assert summary["settle"]["n_settled"] == 12  # settlement still ran and is kept
    assert "Missing THE_ODDS_API_KEY" in summary["collect"]["error"]


def test_pipeline_skip_flags(monkeypatch):
    calls = []
    _install_fakes(monkeypatch, calls)
    summary, code = mod.run_pipeline(skip_settle=True, skip_collect=True)
    assert code == 0
    assert calls == []
    assert summary == {"settle": {"skipped": True}, "collect": {"skipped": True}}
