# beatvegas

Open sports modeling framework — **NBA team win/loss first**.

Elo ratings + Monte Carlo schedule simulation, with preseason and rolling backtests against completed seasons. Simple localhost HTML UI.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the app

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

- **Preseason:** fit Elo on the prior season, regress toward the mean, simulate the target schedule.
- **Rolling:** walk games chronologically (predict → update), score game-level calibration, project records.

First run fetches free schedule/results and caches under `data/cache/`:

1. Try [`nba_api`](https://github.com/swar/nba_api) → stats.nba.com
2. Fall back to [Basketball-Reference](https://www.basketball-reference.com/) monthly game pages (no API key)

## CLI backtest

```bash
python scripts/backtest_cli.py --season 2024-25 --mode both --n-sims 5000
python scripts/backtest_cli.py --season 2024-25 --refresh   # re-pull data
```

## Betting backtest (moneyline + spread)

Uses free historical odds from [csdurfee/scrape_yahoo_odds](https://github.com/csdurfee/scrape_yahoo_odds) (Yahoo closing-style lines).

Rules:
- Compare model probability vs de-vigged market probability
- Bet when edge ≥ tuned `min_edge` (walk-forward: tune on prior season, evaluate holdout)
- Stake with **quarter-Kelly**, capped at 5% of bankroll
- Spreads: Elo → expected margin → Normal CDF cover probability

```bash
python scripts/betting_cli.py --season 2024-25 --tune-season 2023-24
python scripts/betting_cli.py --season 2024-25 --min-edge 0.03   # skip edge search
python scripts/hillclimb_models.py                               # re-search model knobs
```

API: `POST /api/nba/betting` with `{ season, tune_season, min_edge?, bankroll, markets }`
(`markets` may include `"total"`).

### Totals (v3)

Expected total from EMA team offense/defense + rest, vs Yahoo O/U lines.
Hillclimb (`python scripts/hillclimb_totals.py`) over shrink/α/home-boost/rest:

| Market | Holdout ROI | Bets | Verdict |
|--------|-------------|------|---------|
| ML+spread (locked) | ≈ **+0.4%** | ~680 | best so far |
| Totals (best joint) | ≈ **−10%** | ~800 | worse than sides |

Totals are wired in the UI/CLI but **not** the default bet market.

### Model stack (v2)

Predictions use an **ensemble**:
- Dual Elo (slow K=12 + fast K=36) for recency
- SRS (point differential + SOS)
- Rest / B2B margin adjustments
- Outcome calibration + walk-forward `min_edge` tuning

Hillclimb (`python scripts/hillclimb_models.py`) searched shrink / blend / rest / markets.
Locked defaults in `core/defaults.py` (ML+spread, elo_blend=0.7, rest on, shrink=0):

| Window | ROI | Bets | Notes |
|--------|-----|------|-------|
| 2023–24 tune | ≈ −5.3% | ~660 | used only to pick edge |
| 2024–25 holdout | ≈ **+0.4%** | ~680 | flat 1% units, $10k roll |

A flashier spread-only +1.9% holdout config was **rejected** (tune ROI ≈ −36%, n≈50 — overfit).

### Pace + injuries

Defaults also enable:
- **Pace** — BRef advanced team gamelog EMA; scales expected totals (and slightly stretches margins)
- **Injuries** — official NBA injury reports (`nbainjuries`) weighted by prior-season MPG

```bash
python scripts/hillclimb_pace_injuries.py
python scripts/betting_cli.py --no-pace --no-injuries   # ablation
```

## Tests

```bash
pytest -q
```

## Layout

| Path | Role |
|------|------|
| `core/` | Sport-agnostic models, Elo, simulator, backtests |
| `sports/nba/` | NBA adapter + disk cache |
| `app/` | FastAPI + static HTML |
| `scripts/backtest_cli.py` | Headless eval for tuning |
| `data/cache/` | Cached schedules/results (gitignored) |

## Adding another sport

1. Create `sports/<sport>/adapter.py` implementing the same surface as `NbaAdapter`:
   - `sport: str`
   - `list_teams(season) -> Sequence[Team]`
   - `fetch_schedule(season) -> SeasonSchedule`
   - `fetch_results(season) -> SeasonSchedule`
2. Map free schedule/results into `core.models.Game` / `Team` / `SeasonSchedule`.
3. Wire routes in `app/main.py` (or a thin registry) and add the sport to `GET /api/sports`.
4. Reuse `EloEngine`, `SeasonSimulator`, and `BacktestRunner` unchanged; only tune `EloConfig` if the sport needs different home-court / K.

## API

- `GET /api/sports` → `["nba"]`
- `POST /api/nba/simulate` `{ "season", "mode", "n_sims" }`
- `POST /api/nba/refresh` `{ "seasons": ["2023-24", "2024-25"] }`

## Notes

- Free data only (no paid APIs).
- Default tuning season: **2024-25**, warm-started from **2023-24**.
- v1 is team W/L only — no player injuries, lines, or playoffs yet.
