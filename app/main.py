"""Localhost FastAPI app for sports modeling."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.backtest import BacktestRunner
from core.betting import BettingConfig, BettingEngine, tune_min_edge
from core.defaults import DEFAULT_BETTING_CONFIG
from core.elo import EloConfig
from sports.nba.adapter import NbaAdapter, prior_season
from sports.nba.features import features_to_engine_kwargs, load_season_features
from sports.nba.odds import load_odds_for_season, match_rate

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="beatvegas", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_adapter = NbaAdapter()
_runner = BacktestRunner(elo_config=EloConfig())
_betting = BettingEngine()


class SimulateRequest(BaseModel):
    season: str = "2024-25"
    mode: Literal["preseason", "rolling"] = "preseason"
    n_sims: int = Field(default=5000, ge=100, le=50000)
    seed: int = 42


class RefreshRequest(BaseModel):
    season: Optional[str] = None
    seasons: Optional[list[str]] = None


class BettingRequest(BaseModel):
    season: str = "2024-25"
    tune_season: str = "2023-24"
    min_edge: Optional[float] = Field(default=None, ge=0.0, le=0.25)
    markets: list[Literal["moneyline", "spread", "total"]] = Field(
        default_factory=lambda: ["moneyline", "spread"]
    )
    bankroll: float = Field(default=10000.0, ge=100, le=1_000_000)
    kelly_fraction: float = Field(default=0.25, ge=0.05, le=1.0)
    refresh_odds: bool = False
    use_pace: bool = True
    use_injuries: bool = True


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/sports")
def list_sports() -> list[str]:
    return ["nba"]


@app.post("/api/nba/simulate")
def nba_simulate(req: SimulateRequest) -> dict:
    try:
        season_sched = _adapter.fetch_results(req.season, use_cache=True)
        prior = _adapter.fetch_results(prior_season(req.season), use_cache=True)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load NBA data: {exc}") from exc

    if req.mode == "preseason":
        result = _runner.run_preseason(prior, season_sched, n_sims=req.n_sims, seed=req.seed)
    else:
        result = _runner.run_rolling(prior, season_sched, n_sims=req.n_sims, seed=req.seed)

    payload = result.to_dict()
    payload["prior_season"] = prior.season
    payload["n_games"] = len(season_sched.completed_games())
    return payload


@app.post("/api/nba/betting")
def nba_betting(req: BettingRequest) -> dict:
    try:
        tune_sched = _adapter.fetch_results(req.tune_season, use_cache=True)
        holdout_sched = _adapter.fetch_results(req.season, use_cache=True)
        prior_tune = _adapter.fetch_results(prior_season(req.tune_season), use_cache=True)
        prior_hold = _adapter.fetch_results(prior_season(req.season), use_cache=True)
        tune_odds = load_odds_for_season(req.tune_season, force_refresh=req.refresh_odds)
        holdout_odds = load_odds_for_season(req.season, force_refresh=req.refresh_odds)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load betting data: {exc}") from exc

    markets = tuple(req.markets)
    try:
        tune_feat = load_season_features(
            req.tune_season,
            tune_sched,
            prior_schedule=prior_tune,
            load_pace=req.use_pace,
            load_injuries=req.use_injuries,
        )
        hold_feat = load_season_features(
            req.season,
            holdout_sched,
            prior_schedule=prior_hold,
            load_pace=req.use_pace,
            load_injuries=req.use_injuries,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load pace/injury features: {exc}") from exc
    tune_kw = features_to_engine_kwargs(tune_feat)
    hold_kw = features_to_engine_kwargs(hold_feat)

    base = BettingConfig(
        min_edge=DEFAULT_BETTING_CONFIG.min_edge,
        markets=markets,  # type: ignore[arg-type]
        kelly_fraction=req.kelly_fraction,
        starting_bankroll=req.bankroll,
        market_shrink=DEFAULT_BETTING_CONFIG.market_shrink,
        elo_blend=DEFAULT_BETTING_CONFIG.elo_blend,
        srs_blend=DEFAULT_BETTING_CONFIG.srs_blend,
        use_rest=DEFAULT_BETTING_CONFIG.use_rest,
        fast_k=DEFAULT_BETTING_CONFIG.fast_k,
        slow_k=DEFAULT_BETTING_CONFIG.slow_k,
        b2b_penalty=DEFAULT_BETTING_CONFIG.b2b_penalty,
        flat_unit_frac=DEFAULT_BETTING_CONFIG.flat_unit_frac,
        use_pace=req.use_pace,
        use_injuries=req.use_injuries,
        pace_ema_alpha=DEFAULT_BETTING_CONFIG.pace_ema_alpha,
        injury_mpg_coeff=DEFAULT_BETTING_CONFIG.injury_mpg_coeff,
        injury_total_coeff=DEFAULT_BETTING_CONFIG.injury_total_coeff,
        injury_questionable_weight=DEFAULT_BETTING_CONFIG.injury_questionable_weight,
    )

    tune_grid: list[dict] = []
    if req.min_edge is None:
        best_edge, tune_grid, fitted = tune_min_edge(
            _betting, prior_tune, tune_sched, tune_odds, base=base, **tune_kw
        )
        cfg = fitted
        cfg.starting_bankroll = req.bankroll
        cfg.kelly_fraction = req.kelly_fraction
        cfg.min_edge = best_edge
    else:
        best_edge = req.min_edge
        a, b, sigma = _betting.fit_calibration(
            prior_tune,
            tune_sched,
            tune_odds,
            markets=markets,  # type: ignore[arg-type]
            base=base,
            **tune_kw,
        )
        cfg = BettingConfig(
            min_edge=best_edge,
            markets=markets,  # type: ignore[arg-type]
            kelly_fraction=req.kelly_fraction,
            starting_bankroll=req.bankroll,
            flat_unit_frac=0.01,
            calib_a=a,
            calib_b=b,
            margin_sigma=sigma,
            market_shrink=base.market_shrink,
            elo_blend=base.elo_blend,
            srs_blend=base.srs_blend,
            use_rest=base.use_rest,
            use_pace=base.use_pace,
            use_injuries=base.use_injuries,
            pace_ema_alpha=base.pace_ema_alpha,
            injury_mpg_coeff=base.injury_mpg_coeff,
            injury_total_coeff=base.injury_total_coeff,
            injury_questionable_weight=base.injury_questionable_weight,
        )

    result = _betting.run(
        prior_hold,
        holdout_sched,
        holdout_odds,
        config=cfg,
        calibrate_sigma=False,
        **hold_kw,
    )

    hold_keys = {
        (g.date, g.home_team_id, g.away_team_id) for g in holdout_sched.completed_games()
    }
    payload = result.to_dict()
    payload["selected_min_edge"] = best_edge
    payload["tune_season"] = req.tune_season
    payload["tune_grid"] = tune_grid
    payload["odds_match_rate"] = round(match_rate(holdout_odds, hold_keys), 4)
    payload["bets"] = payload["bets"][:100]  # keep response light
    return payload


@app.post("/api/nba/refresh")
def nba_refresh(req: RefreshRequest) -> dict:
    seasons = req.seasons or []
    if req.season:
        seasons.append(req.season)
    if not seasons:
        seasons = ["2022-23", "2023-24", "2024-25"]
    seen = set()
    ordered = []
    for s in seasons:
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    reports = []
    for season in ordered:
        try:
            reports.append(_adapter.refresh(season))
        except Exception as exc:
            reports.append({"season": season, "error": str(exc)})
        try:
            lines = load_odds_for_season(season, force_refresh=True)
            reports.append({"season": season, "odds_lines": len(lines)})
        except Exception as exc:
            reports.append({"season": season, "odds_error": str(exc)})
    return {"refreshed": reports}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
