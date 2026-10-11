"""Odds conversion, vig removal, and Kelly sizing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def american_to_decimal(american: float) -> float:
    a = float(american)
    if a == 0:
        raise ValueError("American odds cannot be 0")
    if a > 0:
        return 1.0 + a / 100.0
    return 1.0 + 100.0 / abs(a)


def decimal_to_american(decimal_odds: float) -> float:
    d = float(decimal_odds)
    if d <= 1.0:
        raise ValueError("Decimal odds must be > 1")
    if d >= 2.0:
        return round((d - 1.0) * 100.0)
    return round(-100.0 / (d - 1.0))


def american_to_implied_prob(american: float) -> float:
    """Raw implied probability including vig."""
    a = float(american)
    if a > 0:
        return 100.0 / (a + 100.0)
    return abs(a) / (abs(a) + 100.0)


def remove_vig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Proportional vig removal for a two-way market."""
    total = prob_a + prob_b
    if total <= 0:
        raise ValueError("Implied probs must be positive")
    return prob_a / total, prob_b / total


def fair_probs_from_american(odds_a: float, odds_b: float) -> tuple[float, float]:
    return remove_vig_two_way(american_to_implied_prob(odds_a), american_to_implied_prob(odds_b))


def kelly_fraction(p_win: float, decimal_odds: float, *, fraction: float = 0.25) -> float:
    """
    Fractional Kelly stake as a fraction of bankroll.
    b = decimal_odds - 1 (net odds).
    f* = (b*p - q) / b
    """
    p = float(p_win)
    q = 1.0 - p
    b = float(decimal_odds) - 1.0
    if b <= 0 or p <= 0 or p >= 1:
        return 0.0
    full = (b * p - q) / b
    if full <= 0:
        return 0.0
    return max(0.0, full * fraction)


@dataclass(frozen=True)
class EdgeOpportunity:
    market: str  # moneyline | spread
    side: str  # home | away
    model_prob: float
    market_prob: float  # fair (devigged)
    edge: float  # model - market
    american_odds: float
    decimal_odds: float
    line: Optional[float] = None  # spread line for the side, if applicable

    @property
    def has_edge(self) -> bool:
        return self.edge > 0
