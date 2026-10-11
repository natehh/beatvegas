"""Preseason and rolling backtest runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional

import numpy as np

from core.elo import EloConfig, EloEngine
from core.models import Game, SeasonSchedule, Team
from core.simulate import SeasonSimulator, SimulationResult


Mode = Literal["preseason", "rolling"]


@dataclass
class GamePrediction:
    game_id: str
    date: date
    home_team_id: str
    away_team_id: str
    p_home: float
    home_won: bool


@dataclass
class TeamProjection:
    team_id: str
    team: str
    expected_wins: float
    expected_losses: float
    actual_wins: int
    actual_losses: int
    error: float
    p25_wins: float
    p50_wins: float
    p75_wins: float


@dataclass
class BacktestResult:
    sport: str
    season: str
    mode: Mode
    n_sims: int
    teams: list[TeamProjection]
    metrics: dict[str, float]
    game_predictions: list[GamePrediction] = field(default_factory=list)
    ratings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sport": self.sport,
            "season": self.season,
            "mode": self.mode,
            "n_sims": self.n_sims,
            "metrics": self.metrics,
            "teams": [t.__dict__ for t in self.teams],
            "ratings": self.ratings,
            "n_game_predictions": len(self.game_predictions),
        }


def _games_played_map(schedule: SeasonSchedule) -> dict[str, int]:
    played = {t.id: 0 for t in schedule.teams}
    for g in schedule.completed_games():
        played[g.home_team_id] = played.get(g.home_team_id, 0) + 1
        played[g.away_team_id] = played.get(g.away_team_id, 0) + 1
    return played


def _compute_record_metrics(
    expected_wins: dict[str, float],
    actual_wins: dict[str, int],
    team_ids: list[str],
) -> dict[str, float]:
    errs = np.array([expected_wins.get(tid, 0.0) - float(actual_wins.get(tid, 0)) for tid in team_ids])
    mae = float(np.mean(np.abs(errs))) if len(errs) else 0.0
    rmse = float(np.sqrt(np.mean(errs**2))) if len(errs) else 0.0
    # Spearman-ish: rank correlation via pearson of ranks
    if len(team_ids) >= 2:
        exp_ranks = np.argsort(np.argsort([-expected_wins.get(t, 0.0) for t in team_ids]))
        act_ranks = np.argsort(np.argsort([-float(actual_wins.get(t, 0)) for t in team_ids]))
        if np.std(exp_ranks) > 0 and np.std(act_ranks) > 0:
            rank_corr = float(np.corrcoef(exp_ranks, act_ranks)[0, 1])
        else:
            rank_corr = 0.0
    else:
        rank_corr = 0.0
    return {"mae_wins": round(mae, 3), "rmse_wins": round(rmse, 3), "rank_corr": round(rank_corr, 3)}


def _compute_game_metrics(preds: list[GamePrediction]) -> dict[str, float]:
    if not preds:
        return {"brier": 0.0, "log_loss": 0.0, "accuracy": 0.0, "n_games": 0}
    y = np.array([1.0 if p.home_won else 0.0 for p in preds])
    p = np.clip(np.array([pred.p_home for pred in preds]), 1e-9, 1 - 1e-9)
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    accuracy = float(np.mean(((p >= 0.5) == (y == 1.0)).astype(float)))
    return {
        "brier": round(brier, 4),
        "log_loss": round(log_loss, 4),
        "accuracy": round(accuracy, 4),
        "n_games": len(preds),
    }


@dataclass
class BacktestRunner:
    elo_config: EloConfig = field(default_factory=EloConfig)

    def run_preseason(
        self,
        prior_schedule: SeasonSchedule,
        season_schedule: SeasonSchedule,
        *,
        n_sims: int = 5000,
        seed: int = 42,
    ) -> BacktestResult:
        """Fit on prior season, regress, simulate all target-season games."""
        engine = EloEngine(config=self.elo_config)
        prior_ids = [t.id for t in prior_schedule.teams]
        engine.fit(prior_schedule.completed_games(), team_ids=prior_ids)

        # Ensure current-season teams exist, then regress toward mean.
        current_ids = [t.id for t in season_schedule.teams]
        for tid in current_ids:
            engine.ensure_team(tid)
        engine.regress_toward_mean(current_ids)

        games = season_schedule.completed_games()
        # For preseason we treat the full known schedule as the sim slate.
        sim_games = games if games else [g for g in season_schedule.games]
        sim = SeasonSimulator(engine=engine, update_during_sim=False)
        result = sim.simulate(sim_games, season_schedule.teams, n_sims=n_sims, seed=seed)

        return self._build_result(
            sport=season_schedule.sport,
            season=season_schedule.season,
            mode="preseason",
            schedule=season_schedule,
            sim=result,
            engine=engine,
            game_predictions=[],
        )

    def run_rolling(
        self,
        prior_schedule: Optional[SeasonSchedule],
        season_schedule: SeasonSchedule,
        *,
        n_sims: int = 5000,
        seed: int = 42,
        as_of: Optional[date] = None,
    ) -> BacktestResult:
        """
        Walk chronologically: predict each game, then update Elo.
        Final projected records = wins so far + Monte Carlo of remaining games
        (or, if as_of is None / end of season, use cumulative expected wins from
        one-step predictions + sim of none remaining → actual path scoring via
        expected wins from initial ratings through full schedule resim at end
        using final? Plan: for record projection at end of rolling walk use
        prior_wins + sim remaining from final as-of date.
        For full-season backtest comparison we also report game-level metrics
        and use expected wins = actual prior wins through last game + 0 remaining
        when evaluating end-of-season — better: resimulate entire season from
        opening day ratings many times? Plan says:
        "Chronological walk of S: before each game predict P(home win), then
        update Elo with result; also optionally resim remaining schedule from
        any as-of date"

        For end-of-season team record comparison in rolling mode:
        Use opening ratings (after prior fit+regress), simulate full season
        WITHOUT updating on actual results for the projection table, while
        still collecting rolling one-step game predictions for Brier/log-loss.
        That separates calibration metrics from schedule simulation.
        Alternatively: projected wins = sum of predicted win probs for each team
        across all games (additive expectation) — cleaner for rolling and doesn't
        need Monte Carlo for mean wins.

        We'll do both:
        - game_predictions from rolling walk (for Brier etc.)
        - expected wins via sum of predicted probs (exact expectation under
          independent Bernoulli) AND Monte Carlo of remaining from as_of for
          distributions. When as_of is end / None, MC remaining is empty so
          use probability-sum expected wins for the table.
        """
        engine = EloEngine(config=self.elo_config)
        if prior_schedule is not None:
            engine.fit(prior_schedule.completed_games(), team_ids=[t.id for t in prior_schedule.teams])
        for t in season_schedule.teams:
            engine.ensure_team(t.id)
        if prior_schedule is not None:
            engine.regress_toward_mean([t.id for t in season_schedule.teams])

        opening = engine.snapshot()
        games = season_schedule.completed_games()
        if as_of is not None:
            games = [g for g in games if g.date <= as_of]

        preds: list[GamePrediction] = []
        expected_wins_sum = {t.id: 0.0 for t in season_schedule.teams}
        actual_so_far = {t.id: 0 for t in season_schedule.teams}

        for g in games:
            p_home = engine.predict_proba(g.home_team_id, g.away_team_id, is_neutral=g.is_neutral)
            assert g.home_won is not None
            preds.append(
                GamePrediction(
                    game_id=g.id,
                    date=g.date,
                    home_team_id=g.home_team_id,
                    away_team_id=g.away_team_id,
                    p_home=p_home,
                    home_won=g.home_won,
                )
            )
            expected_wins_sum[g.home_team_id] = expected_wins_sum.get(g.home_team_id, 0.0) + p_home
            expected_wins_sum[g.away_team_id] = expected_wins_sum.get(g.away_team_id, 0.0) + (1.0 - p_home)
            if g.home_won:
                actual_so_far[g.home_team_id] = actual_so_far.get(g.home_team_id, 0) + 1
            else:
                actual_so_far[g.away_team_id] = actual_so_far.get(g.away_team_id, 0) + 1
            engine.update(g)

        # Remaining schedule for MC distributions (games after as_of, or unfinished).
        if as_of is None:
            remaining = [g for g in season_schedule.games if not g.is_final]
        else:
            remaining = [
                g
                for g in season_schedule.games
                if (g.date > as_of) or (not g.is_final and g.date > as_of)
            ]
            # Also include completed games after as_of as if not yet played.
            remaining = [
                g
                for g in season_schedule.games
                if g.date > as_of
            ]

        # For full-season rolling backtest (typical): use probability-sum as expected
        # wins; for mid-season, add MC of remaining onto actual_so_far.
        if remaining:
            sim = SeasonSimulator(engine=engine, update_during_sim=False)
            mc = sim.simulate(
                remaining,
                season_schedule.teams,
                n_sims=n_sims,
                seed=seed,
                prior_wins=actual_so_far,
            )
            expected_wins = mc.expected_wins
            win_samples = mc.win_samples
        else:
            expected_wins = expected_wins_sum
            # Degenerate samples around expectation for percentile columns.
            win_samples = {
                tid: np.full(max(n_sims, 1), expected_wins_sum.get(tid, 0.0))
                for tid in expected_wins_sum
            }

        # Rebuild a SimulationResult-like structure for the table.
        sim_result = SimulationResult(
            n_sims=n_sims,
            expected_wins=expected_wins,
            win_samples=win_samples,
            team_names={t.id: t.name for t in season_schedule.teams},
        )

        # Restore note: ratings at end of walk are in `engine`; opening unused except
        # we could expose it — keep final ratings.
        _ = opening

        return self._build_result(
            sport=season_schedule.sport,
            season=season_schedule.season,
            mode="rolling",
            schedule=season_schedule,
            sim=sim_result,
            engine=engine,
            game_predictions=preds,
            expected_wins_override=expected_wins if remaining else expected_wins_sum,
        )

    def _build_result(
        self,
        *,
        sport: str,
        season: str,
        mode: Mode,
        schedule: SeasonSchedule,
        sim: SimulationResult,
        engine: EloEngine,
        game_predictions: list[GamePrediction],
        expected_wins_override: Optional[dict[str, float]] = None,
    ) -> BacktestResult:
        actual_wins = schedule.actual_wins()
        actual_losses = schedule.actual_losses()
        played = _games_played_map(schedule)
        expected_wins = expected_wins_override or sim.expected_wins
        team_ids = [t.id for t in schedule.teams]
        names = {t.id: t.name for t in schedule.teams}

        metrics = _compute_record_metrics(expected_wins, actual_wins, team_ids)
        metrics.update(_compute_game_metrics(game_predictions))

        teams: list[TeamProjection] = []
        for tid in sorted(team_ids, key=lambda x: -expected_wins.get(x, 0.0)):
            exp_w = float(expected_wins.get(tid, 0.0))
            act_w = int(actual_wins.get(tid, 0))
            act_l = int(actual_losses.get(tid, 0))
            gp = int(played.get(tid, act_w + act_l))
            samples = sim.win_samples.get(tid, np.array([exp_w]))
            teams.append(
                TeamProjection(
                    team_id=tid,
                    team=names.get(tid, tid),
                    expected_wins=round(exp_w, 2),
                    expected_losses=round(float(gp - exp_w), 2),
                    actual_wins=act_w,
                    actual_losses=act_l,
                    error=round(exp_w - act_w, 2),
                    p25_wins=round(float(np.percentile(samples, 25)), 2),
                    p50_wins=round(float(np.percentile(samples, 50)), 2),
                    p75_wins=round(float(np.percentile(samples, 75)), 2),
                )
            )

        return BacktestResult(
            sport=sport,
            season=season,
            mode=mode,
            n_sims=sim.n_sims,
            teams=teams,
            metrics=metrics,
            game_predictions=game_predictions,
            ratings=engine.snapshot(),
        )
