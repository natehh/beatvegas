"""Load pace + injury feature packs for NBA betting backtests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.injuries import InjuryListing, PlayerMinuteLine
from core.models import SeasonSchedule
from core.pace import PaceObservation
from sports.nba.adapter import prior_season
from sports.nba.injuries import load_injuries_for_dates
from sports.nba.pace_data import home_away_lookup_from_schedule, load_pace_observations
from sports.nba.player_mpg import load_season_player_mpg


@dataclass
class SeasonFeatures:
    pace: list[PaceObservation]
    prior_pace: list[PaceObservation]
    injuries_by_date: dict[str, list[InjuryListing]]
    prior_player_mpg: dict[str, dict]
    minute_lines: list[PlayerMinuteLine]


def load_season_features(
    season: str,
    schedule: SeasonSchedule,
    *,
    prior_schedule: Optional[SeasonSchedule] = None,
    cache_dir: Optional[Path] = None,
    load_injuries: bool = True,
    load_pace: bool = True,
    force_refresh: bool = False,
) -> SeasonFeatures:
    prior = prior_season(season)
    lookup = home_away_lookup_from_schedule(schedule.games)
    prior_lookup = (
        home_away_lookup_from_schedule(prior_schedule.games) if prior_schedule is not None else None
    )

    pace: list[PaceObservation] = []
    prior_pace: list[PaceObservation] = []
    if load_pace:
        pace = load_pace_observations(
            season,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
            home_away_lookup=lookup,
        )
        prior_pace = load_pace_observations(
            prior,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
            home_away_lookup=prior_lookup,
        )

    injuries: dict[str, list[InjuryListing]] = {}
    if load_injuries:
        dates = sorted({g.date for g in schedule.games})
        injuries = load_injuries_for_dates(
            season,
            dates,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )

    prior_mpg = load_season_player_mpg(prior, cache_dir=cache_dir, force_refresh=force_refresh)

    return SeasonFeatures(
        pace=pace,
        prior_pace=prior_pace,
        injuries_by_date=injuries,
        prior_player_mpg=prior_mpg,
        minute_lines=[],
    )


def features_to_engine_kwargs(features: SeasonFeatures) -> dict:
    return {
        "pace_observations": features.pace,
        "prior_pace_observations": features.prior_pace,
        "injuries_by_date": features.injuries_by_date,
        "prior_player_mpg": features.prior_player_mpg,
        "minute_lines": features.minute_lines,
    }
