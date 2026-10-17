"""Edge-based betting rules, bankroll simulation, and walk-forward tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional, Sequence

import numpy as np

from core.ensemble import EnsembleConfig, EnsembleModel
from core.injuries import InjuryConfig, InjuryListing, InjuryModel, PlayerMinuteLine, normalize_player_name
from core.markets import MarketPricer
from core.models import SeasonSchedule
from core.odds import EdgeOpportunity, american_to_decimal, fair_probs_from_american, kelly_fraction
from core.pace import PaceConfig, PaceModel, PaceObservation
from core.totals import TotalsConfig, TotalsModel


MarketName = Literal["moneyline", "spread", "total"]


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
    total_line: Optional[float] = None
    over_odds: Optional[float] = None
    under_odds: Optional[float] = None
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    # Public betting splits (% of wagers / % of money on each side), when the
    # odds source reports them. Yahoo reports these for spread/ML/total.
    spread_home_wager_pct: Optional[float] = None
    spread_away_wager_pct: Optional[float] = None
    spread_home_stake_pct: Optional[float] = None
    spread_away_stake_pct: Optional[float] = None
    money_home_wager_pct: Optional[float] = None
    money_away_wager_pct: Optional[float] = None
    total_over_wager_pct: Optional[float] = None
    total_under_wager_pct: Optional[float] = None

    @property
    def is_final(self) -> bool:
        return self.home_score is not None and self.away_score is not None

    @property
    def has_totals(self) -> bool:
        return (
            self.total_line is not None
            and self.over_odds is not None
            and self.under_odds is not None
        )


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
    one_bet_per_game: bool = True
    elo_per_point: float = 28.0
    margin_sigma: float = 12.0
    flat_unit_frac: Optional[float] = 0.01
    calib_a: float = 1.0
    calib_b: float = 0.0
    # Ensemble knobs (side markets)
    market_shrink: float = 0.35
    elo_blend: float = 0.55
    srs_blend: float = 0.45
    slow_k: float = 12.0
    fast_k: float = 36.0
    slow_weight: float = 0.55
    fast_weight: float = 0.45
    b2b_penalty: float = 2.2
    use_rest: bool = True
    only_fade_favorites: bool = False
    # Totals knobs
    totals_ema_alpha: float = 0.12
    totals_sigma: float = 11.0
    totals_market_shrink: float = 0.25
    totals_home_boost: float = 1.5
    totals_b2b_penalty: float = 2.0
    # Pace / injuries
    use_pace: bool = True
    pace_ema_alpha: float = 0.15
    # Home-court advantage in points (single term, see EnsembleConfig)
    home_advantage_points: float = 2.3
    # Extra shrink of calibrated probs toward the market (applied after
    # outcome calibration). Previously hard-coded at 0.15 in the engine.
    post_calib_shrink: float = 0.15
    # Public-fade (spread): when one side draws >= threshold % of public
    # wagers, the opposite side is bettable even without a model edge.
    use_public_fade: bool = False
    public_fade_threshold: float = 65.0
    # When True, spread bets are restricted to the fade side only.
    public_fade_only: bool = False
    use_injuries: bool = True
    injury_mpg_coeff: float = 4.0
    injury_total_coeff: float = 2.5
    injury_questionable_weight: float = 0.25


def calibrate_probs(model_probs: np.ndarray, outcomes: np.ndarray) -> tuple[float, float]:
    p = np.clip(model_probs.astype(float), 1e-4, 1 - 1e-4)
    y = outcomes.astype(float)
    if len(p) < 40 or y.sum() < 5 or (1 - y).sum() < 5:
        return 1.0, 0.0
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
                "market_shrink": self.config.market_shrink,
                "elo_blend": self.config.elo_blend,
                "srs_blend": self.config.srs_blend,
                "slow_k": self.config.slow_k,
                "fast_k": self.config.fast_k,
                "b2b_penalty": self.config.b2b_penalty,
                "use_rest": self.config.use_rest,
                "use_pace": self.config.use_pace,
                "home_advantage_points": self.config.home_advantage_points,
                "post_calib_shrink": self.config.post_calib_shrink,
                "use_public_fade": self.config.use_public_fade,
                "public_fade_threshold": self.config.public_fade_threshold,
                "public_fade_only": self.config.public_fade_only,
                "use_injuries": self.config.use_injuries,
                "injury_mpg_coeff": self.config.injury_mpg_coeff,
                "pace_ema_alpha": self.config.pace_ema_alpha,
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


def _settle_spread(side: str, line: float, home_score: int, away_score: int) -> Optional[bool]:
    margin = home_score - away_score
    if side == "home":
        adj = margin + line
    else:
        adj = (-margin) + line
    if abs(adj) < 1e-9:
        return None
    return adj > 0


def _settle_total(side: str, line: float, home_score: int, away_score: int) -> Optional[bool]:
    total = home_score + away_score
    diff = total - line
    if abs(diff) < 1e-9:
        return None  # push
    went_over = diff > 0
    return went_over if side == "over" else (not went_over)


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
            "by_moneyline_n": 0.0,
            "by_spread_n": 0.0,
            "log_loss": 0.0,
            "brier": 0.0,
        }
    stakes = np.array([b.stake for b in decided], dtype=float)
    profits = np.array([b.profit for b in decided], dtype=float)
    wins = np.array([1.0 if b.won else 0.0 for b in decided])
    bankrolls = np.array([b.bankroll_after for b in bets], dtype=float)
    peak = np.maximum.accumulate(np.concatenate([[starting], bankrolls]))
    dd = (peak[1:] - bankrolls) / np.clip(peak[1:], 1e-9, None)
    # Probability quality on bet sides
    p = np.clip(np.array([b.model_prob for b in decided]), 1e-6, 1 - 1e-6)
    brier = float(np.mean((p - wins) ** 2))
    log_loss = float(-np.mean(wins * np.log(p) + (1 - wins) * np.log(1 - p)))
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
        "by_total_n": float(sum(1 for b in decided if b.market == "total")),
        "brier": round(brier, 4),
        "log_loss": round(log_loss, 4),
    }


def _ensemble_from_config(cfg: BettingConfig) -> EnsembleModel:
    return EnsembleModel(
        EnsembleConfig(
            slow_k=cfg.slow_k,
            fast_k=cfg.fast_k,
            slow_weight=cfg.slow_weight,
            fast_weight=cfg.fast_weight,
            elo_blend=cfg.elo_blend,
            srs_blend=cfg.srs_blend,
            elo_per_point=cfg.elo_per_point,
            margin_sigma=cfg.margin_sigma,
            b2b_penalty=cfg.b2b_penalty if cfg.use_rest else 0.0,
            rest3_bonus=0.5 if cfg.use_rest else 0.0,
            market_shrink=cfg.market_shrink,
            home_advantage_points=cfg.home_advantage_points,
        )
    )


def _copy_cfg(cfg: BettingConfig, **overrides) -> BettingConfig:
    data = {
        "min_edge": cfg.min_edge,
        "markets": cfg.markets,
        "kelly_fraction": cfg.kelly_fraction,
        "max_bet_frac": cfg.max_bet_frac,
        "starting_bankroll": cfg.starting_bankroll,
        "one_bet_per_game": cfg.one_bet_per_game,
        "elo_per_point": cfg.elo_per_point,
        "margin_sigma": cfg.margin_sigma,
        "flat_unit_frac": cfg.flat_unit_frac,
        "calib_a": cfg.calib_a,
        "calib_b": cfg.calib_b,
        "market_shrink": cfg.market_shrink,
        "elo_blend": cfg.elo_blend,
        "srs_blend": cfg.srs_blend,
        "slow_k": cfg.slow_k,
        "fast_k": cfg.fast_k,
        "slow_weight": cfg.slow_weight,
        "fast_weight": cfg.fast_weight,
        "b2b_penalty": cfg.b2b_penalty,
        "use_rest": cfg.use_rest,
        "only_fade_favorites": cfg.only_fade_favorites,
        "totals_ema_alpha": cfg.totals_ema_alpha,
        "totals_sigma": cfg.totals_sigma,
        "totals_market_shrink": cfg.totals_market_shrink,
        "totals_home_boost": cfg.totals_home_boost,
        "totals_b2b_penalty": cfg.totals_b2b_penalty,
        "use_pace": cfg.use_pace,
        "pace_ema_alpha": cfg.pace_ema_alpha,
        "home_advantage_points": cfg.home_advantage_points,
        "post_calib_shrink": cfg.post_calib_shrink,
        "use_public_fade": cfg.use_public_fade,
        "public_fade_threshold": cfg.public_fade_threshold,
        "public_fade_only": cfg.public_fade_only,
        "use_injuries": cfg.use_injuries,
        "injury_mpg_coeff": cfg.injury_mpg_coeff,
        "injury_total_coeff": cfg.injury_total_coeff,
        "injury_questionable_weight": cfg.injury_questionable_weight,
    }
    data.update(overrides)
    return BettingConfig(**data)


def _seed_injury_minutes(model: InjuryModel, prior_mpg: Optional[dict[str, dict]]) -> None:
    if not prior_mpg:
        return
    for row in prior_mpg.values():
        player = str(row.get("player", "")).strip()
        if not player:
            continue
        try:
            mpg = float(row.get("mpg", 0.0))
        except (TypeError, ValueError):
            continue
        key = normalize_player_name(player)
        if key:
            model.minutes.mpg[key] = mpg
            team_id = str(row.get("team_id", "") or "")
            if team_id:
                model.minutes.team_of[key] = team_id


def _totals_opportunities(
    totals: TotalsModel,
    home_id: str,
    away_id: str,
    line: float,
    over_odds: float,
    under_odds: float,
    *,
    rest=None,
    pace_scale: float = 1.0,
    total_adj: float = 0.0,
) -> list[EdgeOpportunity]:
    raw_over = totals.over_prob(
        home_id,
        away_id,
        line,
        rest=rest,
        pace_scale=pace_scale,
        total_adj=total_adj,
    )
    fair_over, fair_under = fair_probs_from_american(over_odds, under_odds)
    model_over = totals.shrink_to_market(raw_over, fair_over)
    model_under = 1.0 - model_over
    return [
        EdgeOpportunity(
            market="total",
            side="over",
            model_prob=model_over,
            market_prob=fair_over,
            edge=model_over - fair_over,
            american_odds=float(over_odds),
            decimal_odds=american_to_decimal(over_odds),
            line=float(line),
        ),
        EdgeOpportunity(
            market="total",
            side="under",
            model_prob=model_under,
            market_prob=fair_under,
            edge=model_under - fair_under,
            american_odds=float(under_odds),
            decimal_odds=american_to_decimal(under_odds),
            line=float(line),
        ),
    ]


@dataclass
class BettingEngine:
    config: BettingConfig = field(default_factory=BettingConfig)

    def run(
        self,
        prior_schedule: Optional[SeasonSchedule],
        season_schedule: SeasonSchedule,
        odds_lines: Sequence[OddsLine],
        *,
        config: Optional[BettingConfig] = None,
        calibrate_sigma: bool = True,
        pace_observations: Optional[Sequence[PaceObservation]] = None,
        prior_pace_observations: Optional[Sequence[PaceObservation]] = None,
        injuries_by_date: Optional[dict[str, list[InjuryListing]]] = None,
        prior_player_mpg: Optional[dict[str, dict]] = None,
        minute_lines: Optional[Sequence[PlayerMinuteLine]] = None,
    ) -> BettingResult:
        cfg = config or self.config
        need_side = any(m in cfg.markets for m in ("moneyline", "spread"))
        need_total = "total" in cfg.markets
        use_pace = bool(cfg.use_pace and (pace_observations or prior_pace_observations))
        use_injuries = bool(cfg.use_injuries and injuries_by_date is not None)

        model = _ensemble_from_config(cfg) if need_side else None
        totals = TotalsModel(
            TotalsConfig(
                ema_alpha=cfg.totals_ema_alpha,
                total_sigma=cfg.totals_sigma,
                market_shrink=cfg.totals_market_shrink,
                home_total_boost=cfg.totals_home_boost,
                b2b_total_penalty=cfg.totals_b2b_penalty if cfg.use_rest else 0.0,
                rest3_total_boost=0.5 if cfg.use_rest else 0.0,
            )
        )
        pace_model = PaceModel(PaceConfig(ema_alpha=cfg.pace_ema_alpha)) if use_pace else None
        injury_model = (
            InjuryModel(
                InjuryConfig(
                    mpg_coeff=cfg.injury_mpg_coeff,
                    total_coeff=cfg.injury_total_coeff,
                    questionable_weight=cfg.injury_questionable_weight,
                )
            )
            if use_injuries
            else None
        )
        if injury_model is not None:
            _seed_injury_minutes(injury_model, prior_player_mpg)

        pace_by_game = {
            (p.date, p.home_team_id, p.away_team_id): p
            for p in (pace_observations or [])
        }
        # Also index by sorted teams for lookup when home/away encoding differed at fetch
        pace_by_sorted = {
            (p.date, *sorted([p.home_team_id, p.away_team_id])): p
            for p in (pace_observations or [])
        }
        minutes_by_game: dict[tuple, list[PlayerMinuteLine]] = {}
        for ln in minute_lines or []:
            minutes_by_game.setdefault((ln.date, ln.team_id), []).append(ln)

        if prior_schedule is not None:
            team_ids = [t.id for t in prior_schedule.teams]
            if model is not None:
                model.fit(prior_schedule.completed_games(), team_ids)
                model.ensure_teams([t.id for t in season_schedule.teams])
                model.regress_toward_mean([t.id for t in season_schedule.teams])
                if calibrate_sigma:
                    sigma = model.calibrate_sigma_from_prior(prior_schedule.completed_games())
                    cfg = _copy_cfg(cfg, margin_sigma=sigma)
                    model.config.margin_sigma = sigma
            if need_total:
                totals.fit(prior_schedule.completed_games(), team_ids)
                for t in season_schedule.teams:
                    totals.ensure_team(t.id)
                totals.regress_toward_mean([t.id for t in season_schedule.teams])
                if calibrate_sigma:
                    tsigma = totals.calibrate_sigma(prior_schedule.completed_games())
                    cfg = _copy_cfg(cfg, totals_sigma=tsigma)
            if pace_model is not None and prior_pace_observations:
                pace_model.fit(prior_pace_observations, team_ids)
                for t in season_schedule.teams:
                    pace_model.ensure_team(t.id)
                pace_model.regress_toward_mean([t.id for t in season_schedule.teams])
        else:
            if model is not None:
                model.ensure_teams([t.id for t in season_schedule.teams])
            if need_total:
                for t in season_schedule.teams:
                    totals.ensure_team(t.id)
            if pace_model is not None:
                for t in season_schedule.teams:
                    pace_model.ensure_team(t.id)

        from core.rest import RestTracker

        rest_tracker = RestTracker()
        if model is not None:
            model.rest = rest_tracker

        pricer = MarketPricer(model=model) if model is not None else None
        bankroll = cfg.starting_bankroll
        unit = (cfg.flat_unit_frac or 0.0) * cfg.starting_bankroll
        bets: list[Bet] = []
        odds_by_teams_date = {
            (o.date, o.home_team_id, o.away_team_id): o for o in odds_lines
        }

        for g in season_schedule.completed_games():
            rest = rest_tracker.context(g) if cfg.use_rest else None
            pace_scale = 1.0
            if pace_model is not None:
                pace_scale = pace_model.pace_scale(g.home_team_id, g.away_team_id)
            margin_adj = 0.0
            total_adj = 0.0
            if injury_model is not None and injuries_by_date is not None:
                listings = injuries_by_date.get(g.date.isoformat(), [])
                impact = injury_model.impact(g.home_team_id, g.away_team_id, listings)
                margin_adj = impact.margin_adj
                total_adj = impact.total_adj

            o = odds_by_teams_date.get((g.date, g.home_team_id, g.away_team_id))
            opps: list[EdgeOpportunity] = []
            if o is not None:
                if pricer is not None and "moneyline" in cfg.markets:
                    opps.extend(
                        pricer.moneyline_opportunities(
                            g.home_team_id,
                            g.away_team_id,
                            o.home_ml,
                            o.away_ml,
                            is_neutral=g.is_neutral,
                            rest=rest,
                            margin_adj=margin_adj,
                            pace_scale=pace_scale if cfg.use_pace else 1.0,
                        )
                    )
                if pricer is not None and "spread" in cfg.markets:
                    opps.extend(
                        pricer.spread_opportunities(
                            g.home_team_id,
                            g.away_team_id,
                            o.home_spread,
                            o.away_spread,
                            o.home_spread_odds,
                            o.away_spread_odds,
                            is_neutral=g.is_neutral,
                            rest=rest,
                            margin_adj=margin_adj,
                            pace_scale=pace_scale if cfg.use_pace else 1.0,
                        )
                    )
                if need_total and o.has_totals:
                    opps.extend(
                        _totals_opportunities(
                            totals,
                            g.home_team_id,
                            g.away_team_id,
                            float(o.total_line),  # type: ignore[arg-type]
                            float(o.over_odds),  # type: ignore[arg-type]
                            float(o.under_odds),  # type: ignore[arg-type]
                            rest=rest,
                            pace_scale=pace_scale if cfg.use_pace else 1.0,
                            total_adj=total_adj if cfg.use_injuries else 0.0,
                        )
                    )

            calibrated: list[EdgeOpportunity] = []
            for opp in opps:
                p = apply_calibration(opp.model_prob, cfg.calib_a, cfg.calib_b)
                p = (1.0 - cfg.post_calib_shrink) * p + cfg.post_calib_shrink * opp.market_prob
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
            # Public fade: bet against a side drawing an outsized share of
            # public wagers. The fade-side spread candidate is admissible
            # even when the team-strength model sees no edge; with
            # public_fade_only, non-fade spread candidates are dropped.
            if cfg.use_public_fade and o is not None:
                fade_side: Optional[str] = None
                if (
                    o.spread_home_wager_pct is not None
                    and o.spread_home_wager_pct >= cfg.public_fade_threshold
                ):
                    fade_side = "away"
                elif (
                    o.spread_away_wager_pct is not None
                    and o.spread_away_wager_pct >= cfg.public_fade_threshold
                ):
                    fade_side = "home"
                if cfg.public_fade_only:
                    candidates = [
                        x
                        for x in candidates
                        if x.market != "spread" or x.side == fade_side
                    ]
                if fade_side is not None and not any(
                    x.market == "spread" and x.side == fade_side for x in candidates
                ):
                    fade_opp = next(
                        (
                            x
                            for x in calibrated
                            if x.market == "spread" and x.side == fade_side
                        ),
                        None,
                    )
                    if fade_opp is not None:
                        candidates.append(fade_opp)
            if cfg.only_fade_favorites:
                candidates = [
                    x
                    for x in candidates
                    if x.market != "total"
                    and (x.american_odds > 0 or (x.line is not None and x.line > 0))
                ]
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
                    stake_frac = min(
                        kelly_fraction(opp.model_prob, opp.decimal_odds, fraction=cfg.kelly_fraction),
                        cfg.max_bet_frac,
                    )
                    stake = round(bankroll * stake_frac, 2)
                if stake < 1:
                    continue

                won: Optional[bool] = None
                if home_score is not None and away_score is not None:
                    if opp.market == "moneyline":
                        won = _settle_moneyline(opp.side, int(home_score), int(away_score))
                    elif opp.market == "spread":
                        assert opp.line is not None
                        won = _settle_spread(opp.side, float(opp.line), int(home_score), int(away_score))
                    else:
                        assert opp.line is not None
                        won = _settle_total(opp.side, float(opp.line), int(home_score), int(away_score))

                if won is True:
                    profit = stake * (opp.decimal_odds - 1.0)
                elif won is False:
                    profit = -stake
                else:
                    profit = 0.0

                bankroll += profit
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
                rest_tracker.observe(g)
                if model is not None:
                    model.update(g)
                if need_total:
                    totals.update(g)
                if pace_model is not None:
                    obs = pace_by_game.get((g.date, g.home_team_id, g.away_team_id))
                    if obs is None:
                        obs = pace_by_sorted.get(
                            (g.date, *sorted([g.home_team_id, g.away_team_id]))
                        )
                    if obs is not None:
                        # Ensure home/away match the game for bookkeeping
                        pace_model.update(
                            PaceObservation(
                                date=g.date,
                                home_team_id=g.home_team_id,
                                away_team_id=g.away_team_id,
                                pace=obs.pace,
                            )
                        )
                if injury_model is not None:
                    for tid in (g.home_team_id, g.away_team_id):
                        for ln in minutes_by_game.get((g.date, tid), []):
                            injury_model.minutes.update_line(ln)

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
        base: Optional[BettingConfig] = None,
        pace_observations: Optional[Sequence[PaceObservation]] = None,
        prior_pace_observations: Optional[Sequence[PaceObservation]] = None,
        injuries_by_date: Optional[dict[str, list[InjuryListing]]] = None,
        prior_player_mpg: Optional[dict[str, dict]] = None,
        minute_lines: Optional[Sequence[PlayerMinuteLine]] = None,
    ) -> tuple[float, float, float]:
        base = base or self.config
        shadow_cfg = _copy_cfg(
            base,
            min_edge=-1.0,
            markets=markets,
            starting_bankroll=10_000.0,
            flat_unit_frac=0.001,
            one_bet_per_game=False,
            calib_a=1.0,
            calib_b=0.0,
        )
        result = self.run(
            prior_schedule,
            tune_schedule,
            tune_odds,
            config=shadow_cfg,
            pace_observations=pace_observations,
            prior_pace_observations=prior_pace_observations,
            injuries_by_date=injuries_by_date,
            prior_player_mpg=prior_player_mpg,
            minute_lines=minute_lines,
        )
        # Prefer one side per market to avoid double-counting complementary prices.
        preferred_sides = {"moneyline": "home", "spread": "home", "total": "over"}
        sample = [
            bet
            for bet in result.bets
            if bet.won is not None
            and bet.market in markets
            and bet.side == preferred_sides.get(bet.market, bet.side)
        ]
        if len(sample) < 40:
            sample = [bet for bet in result.bets if bet.won is not None]
        probs = np.array([bet.model_prob for bet in sample], dtype=float)
        outs = np.array([1.0 if bet.won else 0.0 for bet in sample], dtype=float)
        a, b = calibrate_probs(probs, outs)
        return a, b, result.config.margin_sigma


def tune_min_edge(
    engine: BettingEngine,
    prior_schedule: SeasonSchedule,
    tune_schedule: SeasonSchedule,
    tune_odds: Sequence[OddsLine],
    *,
    edges: Sequence[float] = (0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10),
    base: Optional[BettingConfig] = None,
    pace_observations: Optional[Sequence[PaceObservation]] = None,
    prior_pace_observations: Optional[Sequence[PaceObservation]] = None,
    injuries_by_date: Optional[dict[str, list[InjuryListing]]] = None,
    prior_player_mpg: Optional[dict[str, dict]] = None,
    minute_lines: Optional[Sequence[PlayerMinuteLine]] = None,
) -> tuple[float, list[dict], BettingConfig]:
    base = base or BettingConfig()
    feature_kwargs = dict(
        pace_observations=pace_observations,
        prior_pace_observations=prior_pace_observations,
        injuries_by_date=injuries_by_date,
        prior_player_mpg=prior_player_mpg,
        minute_lines=minute_lines,
    )
    a, b, sigma = engine.fit_calibration(
        prior_schedule,
        tune_schedule,
        tune_odds,
        markets=base.markets,
        base=base,
        **feature_kwargs,
    )
    fitted = _copy_cfg(
        base,
        margin_sigma=sigma,
        flat_unit_frac=base.flat_unit_frac if base.flat_unit_frac is not None else 0.01,
        calib_a=a,
        calib_b=b,
    )

    rows = []
    best_edge = fitted.min_edge
    best_score = -1e18
    for e in edges:
        cfg = _copy_cfg(fitted, min_edge=float(e))
        result = engine.run(
            prior_schedule,
            tune_schedule,
            tune_odds,
            config=cfg,
            calibrate_sigma=False,
            **feature_kwargs,
        )
        m = result.metrics
        n = m.get("n_decided", 0)
        roi = m.get("roi", 0.0)
        score = roi * np.log1p(n) if n >= 30 else -1e9 + n
        rows.append({"min_edge": e, **m, "score": round(float(score), 4), "calib_a": a, "calib_b": b})
        if score > best_score:
            best_score = score
            best_edge = float(e)
    fitted.min_edge = best_edge
    return best_edge, rows, fitted
