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
python scripts/betting_cli.py --season 2024-25 --min-edge 0.03   # skip tuning
```

Totals are deferred until an expected-points model exists.

API: `POST /api/nba/betting` with `{ season, tune_season, min_edge?, bankroll, markets }`.

### Honest baseline (2024–25 holdout)

A plain Elo + calibrated edge rule **does not currently beat Yahoo closing-style lines** on 2023–24 tune / 2024–25 holdout (negative ROI). That’s expected: closing markets are strong. The betting stack is there to tune rules and measure progress as the model gets better (injuries, priors, totals, etc.).

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
