"""Tests for SRS, rest, and ensemble strength model."""

from datetime import date

from core.ensemble import EnsembleModel
from core.models import Game, Team
from core.rest import RestTracker
from core.srs import SrsEngine


def _g(gid, d, home, away, hs, aws):
    return Game(gid, date(2024, 1, d), home, away, hs, aws, "2023-24")


def test_srs_ranks_better_offense():
    games = [
        _g("1", 1, "A", "B", 120, 100),
        _g("2", 2, "A", "C", 115, 100),
        _g("3", 3, "B", "C", 110, 100),
        _g("4", 4, "A", "B", 108, 100),
        _g("5", 5, "C", "B", 95, 100),
        _g("6", 6, "A", "C", 112, 105),
    ]
    eng = SrsEngine()
    eng.fit(games, ["A", "B", "C"])
    assert eng.get_rating("A") > eng.get_rating("C")


def test_rest_tracker_b2b():
    tr = RestTracker()
    g1 = _g("1", 1, "H", "A", 100, 90)
    ctx1 = tr.context(g1)
    assert ctx1.home_days_rest >= 3
    tr.observe(g1)
    g2 = _g("2", 2, "H", "B", 100, 95)  # next day = B2B
    ctx2 = tr.context(g2)
    assert ctx2.home_b2b
    assert ctx2.margin_adjustment() < 0


def test_ensemble_fit_update():
    teams = ["A", "B"]
    games = [_g("1", 1, "A", "B", 110, 100), _g("2", 3, "B", "A", 105, 100)]
    model = EnsembleModel()
    model.fit(games, teams)
    p_before = model.predict_proba("A", "B")
    model.update(_g("3", 5, "A", "B", 125, 100))
    p_after = model.predict_proba("A", "B")
    assert 0.0 < p_before < 1.0
    assert 0.0 < p_after < 1.0
