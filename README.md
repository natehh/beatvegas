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

### Public fade — the edge that survived walk-forward testing

An extensive sweep (closing ML/spread/total vs Elo/SRS ensemble, rest/travel/
ATS-form, NBA.com advanced stats, line movement, cross-book and cross-market
checks, playoffs, second-half lines, and public betting splits) found closing
team markets to be very efficient: the team model above was **negative** on
the 2022-23 (−0.4%) and 2023-24 (−12.5%) walk-forward holdouts in
`scripts/walkforward_research.py` (no-pace variant). The one rule that was
positive in **every** season tested uses a data field the Yahoo CSV already
carried but the loader discarded — public betting splits:

> When one side draws **65-72% of public wagers**, bet the other side at the
> posted spread, flat 1% units. Splits ≥72% are *not* faded (they tend to
> reflect genuine mismatches), and fading all ≥65% splits was only ~+1% and
> negative in 2024-25.

| Season | Bets | Hit | ROI |
|--------|------|-----|-----|
| 2021-22 | 223 | 54.7% | +4.4% |
| 2022-23 | 196 | 55.1% | +5.5% |
| 2023-24 | 187 | 55.6% | +6.1% |
| 2024-25 | 269 | 54.3% | +3.5% |
| **All** | **875** | **54.9%** | **+4.7%** |

```bash
python scripts/walkforward_research.py                  # model vs public fade, all holdouts
python scripts/betting_cli.py --season 2024-25 --public-fade
```

Also available via the API: `POST /api/nba/betting` with
`{ "season": "2024-25", "public_fade": true }` (returns `"strategy": "public_fade"`).

Caveats: ~875 bets is a modest sample (ROI 95% CI roughly −1% to +11%), Yahoo
split coverage is 75-100% of games depending on season, and the effect is a
market-microstructure signal (book shading of popular sides), so it may decay
if splits reporting or book pricing changes. Two model bugs found along the
way were also fixed: back-to-backs were misclassified (1 day of rest counted
as a B2B), and home court was double-counted in the ensemble margin
(Elo + SRS home edges blended, ~+1 pt home bias); home court is now a single
`home_advantage_points=2.3` term.

### Does the public fade transfer to college sports?

Short answer: **no for college basketball; maybe, weakly, for college
football** (`scripts/college_public_fade_research.py`). No free archive of
historical college splits exists in Yahoo/SBR form, so this uses
VegasInsider's dated/weekly odds pages + per-game "betting trends"
(bets%/money% splits, opening and consensus lines) with finals from ESPN.
VI's splits match Yahoo's NBA splits on identical games (66% vs 66.2%),
so the feed is the same family as the NBA strategy above. Coverage:
NCAAB 2024-25 (from mid-Nov) + 2025-26 (~9,800 settled games), NCAAF
2025 + partial 2026 (~1,240 games).

| Test | Bets | Hit | ROI |
|------|------|-----|-----|
| NCAAB fade, NBA band 65-72% | 1,565 | 50.0% | **−4.4%** (both seasons ≈ −4%) |
| NCAAB fade, all bands (≥55% to ≥80%, money% too) | — | — | all negative |
| NCAAB totals fade (over ≥80% bets) | 2,157 | 52.3% | −0.2% (breakeven at best) |
| NCAAF fade, NBA band 65-72% | 188 | 47.0% | −10.3% |
| NCAAF fade, ≥75% of bets | 256 | 56.6% | +7.6% |
| NCAAF fade, ≥80% of bets | 196 | 58.2% | +10.5% (both seasons positive) |

For NCAAB the negative result is comprehensive, not just the headline
bands: fading was negative for popular home sides and popular away sides,
popular favourites (−1.9%, the least-bad slice) and popular underdogs,
in the tournament window and in the regular season; bets-vs-money
"sharp divergence" sides, open→close line-movement follow *and* fade,
and unconditional home/favourite/underdog baselines (dogs came closest:
51.4% ATS, still −2.0% after the vig) all lost money. Nothing in the
splits/lines data clears the vig in college basketball.

A team-strength model doesn't beat it either — and the choice of model
family doesn't matter. All of the following were built strictly
pregame, calibrated/tuned on 2024-25 and held out on 2025-26 against
the VI consensus line: a 538-style **Elo** (MOV multiplier,
between-season regression) from all ESPN D1 results; **Massey
least-squares** power ratings (weekly ridge refits on margins); an
**adjusted efficiency** model (per-team offence/defence least squares
on scores, KenPom-style); **ridge and gradient boosting** on pregame
features (all three ratings, recent form, rest, scoring), including a
market-residual target; and a rolling-scoring totals model. Betting
*with* any of them lost −3% to −6.4% at every edge threshold on the
holdout; *fading* them (taking the market side whenever they
disagreed) lost −2.8% to −5.3% as well. The gradient booster posted
+16% to +90% on the tuning season and −3% to −5% on the holdout —
textbook overfitting, same as the one positive Elo tuning cell
(edge ≥6 pts, +1.8%, which flipped to −5.8%). Conclusion: the college
consensus line is efficient relative to public-data models of any
family, in both directions. (Model tests:
`scripts/college_public_fade_research.py` for Elo/totals,
`scripts/college_model_families_research.py` for the rest.)

The NCAAF result needs an asterisk the size of a billboard: the sample is
small (bootstrap 95% CI for the ≥80% ROI is roughly −2% to +23%), the
≥72%+ bands were found by scanning, and the effect is mostly *not* the
NBA mechanism. In lopsided college matchups the public takes the huge
underdog (+30), so "fading" usually means laying −30 with the favourite;
unconditionally, NCAAF favourites of 21+ covered 55.5% (+5.7% ROI, n=347)
with no splits filter at all. Meanwhile the money% (handle) version of
the fade is flat, and NCAAB favourites show no such pattern. Verdict:
college basketball definitively rejects the rule; college football has a
suggestive big-favourite/extreme-splits effect worth forward-tracking,
but it is **not** locked in as a preset. Reproduce with
`python scripts/college_public_fade_research.py` (first run scrapes
VegasInsider + ESPN into `data/cache/college_vi/`; later runs are cached).

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

| Feature | Default | 2024–25 holdout (walk-forward) |
|---------|---------|--------------------------------|
| Baseline (no pace/inj) | — | ≈ **+0.4%** ROI (~680 bets) |
| **Pace** (BRef EMA, α=0.15) | **on** | ≈ **+0.6%** ROI (~680 bets) |
| Injuries (NBA reports × prior MPG) | **off** (opt-in) | ≈ −6% to −12% — not locked |

```bash
python scripts/hillclimb_pace_injuries.py
python scripts/betting_cli.py --no-pace          # ablation
python scripts/betting_cli.py                    # pace on, injuries off
```

## Player props — forward collection

Historical prop closes are not free at season scale, so we **record going forward** and settle against box scores.

1. Get a free key at [the-odds-api.com](https://the-odds-api.com/) (~500 credits/month).
2. Export it and snapshot upcoming NBA props (points / rebounds / assists by default):

```bash
export THE_ODDS_API_KEY=your_key
python scripts/collect_prop_odds.py                  # ~3 credits × N games in next 36h
python scripts/collect_prop_odds.py --max-events 5   # credit-safe smoke
python scripts/settle_prop_odds.py                   # grade finished games via ESPN
```

Ledger files (gitignored):
- `data/props/prop_quotes.jsonl` — each book/market/player/line snapshot
- `data/props/prop_settled.jsonl` — joined actuals + over/under results
- `data/props/snapshots/<timestamp>/` — raw API payloads

Suggested cadence: **1× daily pregame** (and optionally again near tip for a crude open→close pair). Extended markets: `--extended` (more credits).

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
| `scripts/collect_prop_odds.py` | Forward prop-odds snapshots |
| `scripts/settle_prop_odds.py` | Settle props vs ESPN box scores |
| `data/cache/` | Cached schedules/results (gitignored) |
| `data/props/` | Prop quote ledger (gitignored) |

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
