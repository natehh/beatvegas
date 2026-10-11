"""Localhost FastAPI app for sports modeling."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.backtest import BacktestRunner
from core.elo import EloConfig
from sports.nba.adapter import NbaAdapter, prior_season

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="beatvegas", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_adapter = NbaAdapter()
_runner = BacktestRunner(elo_config=EloConfig())


class SimulateRequest(BaseModel):
    season: str = "2024-25"
    mode: Literal["preseason", "rolling"] = "preseason"
    n_sims: int = Field(default=5000, ge=100, le=50000)
    seed: int = 42


class RefreshRequest(BaseModel):
    season: Optional[str] = None
    seasons: Optional[list[str]] = None


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


@app.post("/api/nba/refresh")
def nba_refresh(req: RefreshRequest) -> dict:
    seasons = req.seasons or []
    if req.season:
        seasons.append(req.season)
    if not seasons:
        seasons = ["2023-24", "2024-25"]
    # unique preserve order
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
    return {"refreshed": reports}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
