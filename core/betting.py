"""Edge-based betting rules, bankroll simulation, and walk-forward tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional, Sequence

import numpy as np

from core.elo import EloConfig, EloEngine
from core.markets import MarginModel, MarketPricer, calibrate_margin_sigma
from core.models import Game, SeasonSchedule
from core.odds import EdgeOpportunity, kelly_fraction


MarketName = Literal["moneyline", "spread"]


@dataclass
class OddsLine:
    game_key: str
    date: date
    home_team_id: str
    away_team_id: str
    home_ml: float
    away_ml: float
    home_spread: float
    away_spread: float
    home_spread_odds: float
    away_spread_odds: float
    home_score: Optional[int] = None
    away_score: Optional[int] = None

    @property
    def is_final(self) -> bool:
        return self.home_score is not None and self.away_score is not None


@dataclass
class Bet:
    date: date
    game_key: str
    market: MarketName
    side: str
    stake: float
    decimal_odds: float
    american_odds: float
    model_prob: float
    market_prob: float
    edge: float
    line: Optional[float]
    won: Optional[bool]
    profit: float
    bankroll_after: float


@dataclass
class BettingConfig:
    min_edge: float = 0.03
    markets: tuple[MarketName, ...] = ("moneyline", "spread")
    kelly_fraction: float = 0.25
    max_bet_frac: float = 0.05
    starting_bankroll: float = 10_000.0
    one_bet_per_game: bool = True  # take largest edge only
    elo_per_point: float = 28.0
    margin_sigma: float = 12.0
    # Flat unit staking (fraction of starting bankroll). If set, overrides Kelly.
    flat_unit_frac: Optional[float] = 0.01
    # Affine calibration: p' = clip(a * p + b). Identity if None.
    calib_a: float = 1.0
    calib_b: float = 0.0


def calibrate_probs(model_probs: np.ndarray, outcomes: np.ndarray) -> tuple[float, float]:
    """
    Fit a simple logistic (Platt-style) map from model prob -> calibrated prob.
    Returns (a, b) for p' = sigmoid(a * logit(p) + b), stored as linear approx
    on probability space via two-point fit for robustness on small samples.

    We fit: logit(p_emp) ~ a * logit(p_model) + b on binned data, then expose
    an affine map in logit space applied at bet time.
    """
    p = np.clip(model_probs.astype(float), 1e-4, 1 - 1e-4)
    y = outcomes.astype(float)
    if len(p) < 40 or y.sum() < 5 or (1 - y).sum() < 5:
        return 1.0, 0.0
    # Bin into deciles and regress logit(emp) on logit(model)
    qs = np.quantile(p, np.linspace(0, 1, 11))
    xs, ys = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        mask = (p >= lo) & (p <= hi if hi == qs[-1] else p < hi)
        if mask.sum() < 5:
            continue
        emp = float(np.clip(y[mask].mean(), 1e-3, 1 - 1e-3))
        mid = float(np.clip(p[mask].mean(), 1e-3, 1 - 1e-3))
        xs.append(np.log(mid / (1 - mid)))
        ys.append(np.log(emp / (1 - emp)))
    if len(xs) < 3:
        return 1.0, 0.0
    a, b = np.polyfit(xs, ys, 1)
    # Keep mild shrink toward identity to avoid wild fits
    a = float(np.clip(a, 0.2, 2.5))
    b = float(np.clip(b, -2.0, 2.0))
    return a, b


def apply_calibration(p: float, a: float, b: float) -> float:
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    logit = np.log(p / (1 - p))
    z = a * logit + b
    return float(1.0 / (1.0 + np.exp(-z)))


@dataclass
class BettingResult:
    config: BettingConfig
    bets: list[Bet]
    final_bankroll: float
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "config": {
                "min_edge": self.config.min_edge,
                "markets": list(self.config.markets),
                "kelly_fraction": self.config.kelly_fraction,
                "max_bet_frac": self.config.max_bet_frac,
                "starting_bankroll": self.config.starting_bankroll,
                "one_bet_per_game": self.config.one_bet_per_game,
                "elo_per_point": self.config.elo_per_point,
                "margin_sigma": self.config.margin_sigma,
                "flat_unit_frac": self.config.flat_unit_frac,
                "calib_a": self.config.calib_a,
                "calib_b": self.config.calib_b,
            },
            "metrics": self.metrics,
            "final_bankroll": self.final_bankroll,
            "n_bets": len(self.bets),
            "bets": [
                {
                    "date": b.date.isoformat(),
                    "game_key": b.game_key,
                    "market": b.market,
                    "side": b.side,
                    "stake": round(b.stake, 2),
                    "american_odds": b.american_odds,
                    "model_prob": round(b.model_prob, 4),
                    "market_prob": round(b.market_prob, 4),
                    "edge": round(b.edge, 4),
                    "line": b.line,
                    "won": b.won,
                    "profit": round(b.profit, 2),
                    "bankroll_after": round(b.bankroll_after, 2),
                }
                for b in self.bets
            ],
        }


def _settle_moneyline(side: str, home_score: int, away_score: int) -> Optional[bool]:
    if home_score == away_score:
        return None
    home_won = home_score > away_score
    return home_won if side == "home" else (not home_won)


def _settle_spread(
    side: str,
    line: float,
    home_score: int,
    away_score: int,
) -> Optional[bool]:
    margin = home_score - away_score
    if side == "home":
        adj = margin + line
    else:
        # away line is typically +X when home is -X
        adj = (-margin) + line
    if abs(adj) < 1e-9:
        return None  # push
    return adj > 0


def _metrics(bets: list[Bet], starting: float, ending: float) -> dict[str, float]:
    decided = [b for b in bets if b.won is not None]
    n = len(decided)
    if n == 0:
        return {
            "n_bets": 0,
            "n_decided": 0,
            "hit_rate": 0.0,
            "roi": 0.0,
            "profit": 0.0,
            "final_bankroll": ending,
            "max_drawdown": 0.0,
            "avg_edge": 0.0,
            "avg_stake": 0.0,
        }
    stakes = np.array([b.stake for b in decided], dtype=float)
    profits = np.array([b.profit for b in decided], dtype=float)
    wins = np.array([1.0 if b.won else 0.0 for b in decided])
    bankrolls = np.array([b.bankroll_after for b in bets], dtype=float)
    peak = np.maximum.accumulate(np.concatenate([[starting], bankrolls]))
    dd = (peak[1:] - bankrolls) / np.clip(peak[1:], 1e-9, None)
    return {
        "n_bets": float(len(bets)),
        "n_decided": float(n),
        "hit_rate": round(float(np.mean(wins)), 4),
        "roi": round(float(profits.sum() / stakes.sum()), 4) if stakes.sum() else 0.0,
        "profit": round(float(profits.sum()), 2),
        "final_bankroll": round(float(ending), 2),
        "max_drawdown": round(float(np.max(dd)) if len(dd) else 0.0, 4),
        "avg_edge": round(float(np.mean([b.edge for b in decided])), 4),
        "avg_stake": round(float(np.mean(stakes)), 2),
        "by_moneyline_n": float(sum(1 for b in decided if b.market == "moneyline")),
        "by_spread_n": float(sum(1 for b in decided if b.market == "spread")),
    }


@dataclass
class BettingEngine:
    elo_config: EloConfig = field(default_factory=EloConfig)
    config: BettingConfig = field(default_factory=BettingConfig)

    def run(
        self,
        prior_schedule: Optional[SeasonSchedule],
        season_schedule: SeasonSchedule,
        odds_lines: Sequence[OddsLine],
        *,
        config: Optional[BettingConfig] = None,
        calibrate_sigma: bool = True,
    ) -> BettingResult:
        cfg = config or self.config
        engine = EloEngine(config=self.elo_config)
        if prior_schedule is not None:
            engine.fit(prior_schedule.completed_games(), [t.id for t in prior_schedule.teams])
        for t in season_schedule.teams:
            engine.ensure_team(t.id)
        if prior_schedule is not None:
            engine.regress_toward_mean([t.id for t in season_schedule.teams])

        games_chrono = season_schedule.completed_games()

        sigma = cfg.margin_sigma
        if calibrate_sigma and prior_schedule is not None:
            diffs, margins = [], []
            tmp = EloEngine(config=self.elo_config)
            tmp.fit([], [t.id for t in prior_schedule.teams])
            # rebuild diffs chronologically on prior season
            pe = EloEngine(config=self.elo_config)
            for g in prior_schedule.completed_games():
                if not g.is_final:
                    continue
                hca = 0.0 if g.is_neutral else pe.config.home_advantage
                d = pe.get_rating(g.home_team_id) + hca - pe.get_rating(g.away_team_id)
                diffs.append(d)
                margins.append(float(g.home_score) - float(g.away_score))  # type: ignore[arg-type]
                pe.update(g)
            if len(diffs) >= 50:
                sigma = calibrate_margin_sigma(np.array(diffs), np.array(margins), elo_per_point=cfg.elo_per_point)

        pricer = MarketPricer(
            engine=engine,
            margin=MarginModel(elo_per_point=cfg.elo_per_point, margin_sigma=sigma),
        )
        cfg = BettingConfig(
            min_edge=cfg.min_edge,
            markets=cfg.markets,
            kelly_fraction=cfg.kelly_fraction,
            max_bet_frac=cfg.max_bet_frac,
            starting_bankroll=cfg.starting_bankroll,
            one_bet_per_game=cfg.one_bet_per_game,
            elo_per_point=cfg.elo_per_point,
            margin_sigma=sigma,
            flat_unit_frac=cfg.flat_unit_frac,
            calib_a=cfg.calib_a,
            calib_b=cfg.calib_b,
        )

        bankroll = cfg.starting_bankroll
        unit = (cfg.flat_unit_frac or 0.0) * cfg.starting_bankroll
        bets: list[Bet] = []
        odds_sorted = sorted(odds_lines, key=lambda o: (o.date, o.game_key))
        odds_by_teams_date = {
            (o.date, o.home_team_id, o.away_team_id): o for o in odds_sorted
        }

        for g in games_chrono:
            o = odds_by_teams_date.get((g.date, g.home_team_id, g.away_team_id))
            opps: list[EdgeOpportunity] = []
            if o is not None:
                if "moneyline" in cfg.markets:
                    opps.extend(
                        pricer.moneyline_opportunities(
                            g.home_team_id,
                            g.away_team_id,
                            o.home_ml,
                            o.away_ml,
                            is_neutral=g.is_neutral,
                        )
                    )
                if "spread" in cfg.markets:
                    opps.extend(
                        pricer.spread_opportunities(
                            g.home_team_id,
                            g.away_team_id,
                            o.home_spread,
                            o.away_spread,
                            o.home_spread_odds,
                            o.away_spread_odds,
                            is_neutral=g.is_neutral,
                        )
                    )

            # Apply probability calibration, then recompute edge vs market.
            calibrated: list[EdgeOpportunity] = []
            for opp in opps:
                p = apply_calibration(opp.model_prob, cfg.calib_a, cfg.calib_b)
                calibrated.append(
                    EdgeOpportunity(
                        market=opp.market,
                        side=opp.side,
                        model_prob=p,
                        market_prob=opp.market_prob,
                        edge=p - opp.market_prob,
                        american_odds=opp.american_odds,
                        decimal_odds=opp.decimal_odds,
                        line=opp.line,
                    )
                )

            candidates = [x for x in calibrated if x.edge >= cfg.min_edge]
            if cfg.one_bet_per_game and candidates:
                candidates = [max(candidates, key=lambda x: x.edge)]

            home_score = g.home_score if g.home_score is not None else (o.home_score if o else None)
            away_score = g.away_score if g.away_score is not None else (o.away_score if o else None)

            for opp in candidates:
                if bankroll <= 1:
                    break
                if cfg.flat_unit_frac is not None:
                    stake = round(min(unit, bankroll * cfg.max_bet_frac), 2)
                else:
                    stake_frac = kelly_fraction(
                        opp.model_prob, opp.decimal_odds, fraction=cfg.kelly_fraction
                    )
                    stake_frac = min(stake_frac, cfg.max_bet_frac)
                    stake = round(bankroll * stake_frac, 2)
                if stake < 1:
                    continue

                won: Optional[bool] = None
                if home_score is not None and away_score is not None:
                    if opp.market == "moneyline":
                        won = _settle_moneyline(opp.side, int(home_score), int(away_score))
                    else:
                        assert opp.line is not None
                        won = _settle_spread(opp.side, float(opp.line), int(home_score), int(away_score))

                if won is True:
                    profit = stake * (opp.decimal_odds - 1.0)
                elif won is False:
                    profit = -stake
                else:
                    profit = 0.0  # push / unknown

                bankroll = bankroll + profit
                bets.append(
                    Bet(
                        date=g.date,
                        game_key=o.game_key if o is not None else g.id,
                        market=opp.market,  # type: ignore[arg-type]
                        side=opp.side,
                        stake=stake,
                        decimal_odds=opp.decimal_odds,
                        american_odds=opp.american_odds,
                        model_prob=opp.model_prob,
                        market_prob=opp.market_prob,
                        edge=opp.edge,
                        line=opp.line,
                        won=won,
                        profit=profit,
                        bankroll_after=bankroll,
                    )
                )

            if g.is_final:
                engine.update(g)

        return BettingResult(
            config=cfg,
            bets=bets,
            final_bankroll=bankroll,
            metrics=_metrics(bets, cfg.starting_bankroll, bankroll),
        )

    def fit_calibration(
        self,
        prior_schedule: SeasonSchedule,
        tune_schedule: SeasonSchedule,
        tune_odds: Sequence[OddsLine],
        *,
        markets: tuple[MarketName, ...] = ("moneyline", "spread"),
    ) -> tuple[float, float, float]:
        """Return (calib_a, calib_b, margin_sigma) fit on the tune season."""
        # Run with zero edge / tiny flat stakes just to harvest probs via a shadow walk.
        shadow_cfg = BettingConfig(
            min_edge=-1.0,  # keep all sides; we'll filter below
            markets=markets,
            starting_bankroll=10_000.0,
            flat_unit_frac=0.001,
            one_bet_per_game=False,
        )
        # Use identity calibration for the harvest pass.
        result = self.run(prior_schedule, tune_schedule, tune_odds, config=shadow_cfg)
        # Identity calib harvest: use home-side settled bets only (avoid double-counting).
        sample = [
            b
            for b in result.bets
            if b.won is not None and b.side == "home" and b.market in markets
        ]
        if len(sample) < 40:
            sample = [b for b in result.bets if b.won is not None and b.side == "home"]
        probs = np.array([b.model_prob for b in sample], dtype=float)
        outs = np.array([1.0 if b.won else 0.0 for b in sample], dtype=float)
        a, b = calibrate_probs(probs, outs)
        sigma = result.config.margin_sigma
        return a, b, sigma


def tune_min_edge(
    engine: BettingEngine,
    prior_schedule: SeasonSchedule,
    tune_schedule: SeasonSchedule,
    tune_odds: Sequence[OddsLine],
    *,
    edges: Sequence[float] = (0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10),
    base: Optional[BettingConfig] = None,
) -> tuple[float, list[dict], BettingConfig]:
    """
    Fit calibration on the tune window, then pick min_edge maximizing flat-unit ROI.
    Returns (best_edge, grid_rows, fitted_base_config).
    """
    base = base or BettingConfig()
    a, b, sigma = engine.fit_calibration(
        prior_schedule, tune_schedule, tune_odds, markets=base.markets
    )
    fitted = BettingConfig(
        min_edge=base.min_edge,
        markets=base.markets,
        kelly_fraction=base.kelly_fraction,
        max_bet_frac=base.max_bet_frac,
        starting_bankroll=base.starting_bankroll,
        one_bet_per_game=base.one_bet_per_game,
        elo_per_point=base.elo_per_point,
        margin_sigma=sigma,
        flat_unit_frac=base.flat_unit_frac if base.flat_unit_frac is not None else 0.01,
        calib_a=a,
        calib_b=b,
    )

    rows = []
    best_edge = fitted.min_edge
    best_score = -1e18
    for e in edges:
        cfg = BettingConfig(
            min_edge=float(e),
            markets=fitted.markets,
            kelly_fraction=fitted.kelly_fraction,
            max_bet_frac=fitted.max_bet_frac,
            starting_bankroll=fitted.starting_bankroll,
            one_bet_per_game=fitted.one_bet_per_game,
            elo_per_point=fitted.elo_per_point,
            margin_sigma=fitted.margin_sigma,
            flat_unit_frac=fitted.flat_unit_frac,
            calib_a=fitted.calib_a,
            calib_b=fitted.calib_b,
        )
        result = engine.run(
            prior_schedule, tune_schedule, tune_odds, config=cfg, calibrate_sigma=False
        )
        m = result.metrics
        n = m.get("n_decided", 0)
        roi = m.get("roi", 0.0)
        score = roi * np.log1p(n) if n >= 30 else -1e9 + n
        rows.append(
            {
                "min_edge": e,
                **m,
                "score": round(float(score), 4),
                "calib_a": a,
                "calib_b": b,
            }
        )
        if score > best_score:
            best_score = score
            best_edge = float(e)
    fitted.min_edge = best_edge
    return best_edge, rows, fitted
