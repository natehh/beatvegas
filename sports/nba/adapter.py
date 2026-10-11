"""NBA data adapter with free sources and local cache.

Primary attempt: nba_api (stats.nba.com).
Fallback: Basketball-Reference monthly schedule pages (no API key).
"""

from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from core.models import Game, SeasonSchedule, Team
from sports.nba import cache as nba_cache
from sports.nba.bref import fetch_season_from_bref
from sports.nba.teams import normalize_abbr


def prior_season(season: str) -> str:
    """'2024-25' -> '2023-24'."""
    start = int(season.split("-")[0])
    prev = start - 1
    return f"{prev}-{str(start)[-2:]}"


class NbaAdapter:
    sport = "nba"

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        request_delay_s: float = 0.6,
        nba_api_timeout_s: float = 8.0,
    ) -> None:
        self.cache_dir = cache_dir or nba_cache.DEFAULT_CACHE_DIR
        self.request_delay_s = request_delay_s
        self.nba_api_timeout_s = nba_api_timeout_s

    def list_teams(self, season: str) -> Sequence[Team]:
        schedule = self.fetch_results(season, use_cache=True)
        return schedule.teams

    def fetch_schedule(self, season: str, *, use_cache: bool = True, force_refresh: bool = False) -> SeasonSchedule:
        # For completed seasons, schedule == results for our modeling purposes.
        return self.fetch_results(season, use_cache=use_cache, force_refresh=force_refresh)

    def fetch_results(self, season: str, *, use_cache: bool = True, force_refresh: bool = False) -> SeasonSchedule:
        if use_cache and not force_refresh:
            cached = nba_cache.load_schedule(season, self.cache_dir, kind="results")
            if cached is not None:
                return cached

        errors: list[str] = []
        schedule: Optional[SeasonSchedule] = None

        try:
            schedule = self._fetch_results_from_nba_api(season)
        except Exception as exc:
            errors.append(f"nba_api: {exc}")

        if schedule is None:
            try:
                schedule = fetch_season_from_bref(season, delay_s=self.request_delay_s)
            except Exception as exc:
                errors.append(f"basketball-reference: {exc}")

        if schedule is None:
            raise RuntimeError(
                "Failed to fetch NBA results from all free sources: " + " | ".join(errors)
            )

        nba_cache.save_schedule(schedule, self.cache_dir, kind="results")
        nba_cache.save_schedule(schedule, self.cache_dir, kind="schedule")
        return schedule

    def refresh(self, season: str) -> dict[str, str]:
        results = self.fetch_results(season, use_cache=False, force_refresh=True)
        return {
            "season": season,
            "n_teams": str(len(results.teams)),
            "n_result_games": str(len(results.completed_games())),
            "n_schedule_games": str(len(results.games)),
            "source": "cached-after-fetch",
        }

    def _sleep(self) -> None:
        if self.request_delay_s > 0:
            time.sleep(self.request_delay_s)

    def _fetch_results_from_nba_api(self, season: str) -> SeasonSchedule:
        from nba_api.stats.endpoints import leaguegamelog

        self._sleep()
        log = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="T",
            timeout=self.nba_api_timeout_s,
        )
        df = log.get_data_frames()[0]
        if df.empty:
            raise RuntimeError(f"No NBA game log rows for season {season}")
        return self._schedule_from_game_log(df, season)

    def _schedule_from_game_log(self, df: pd.DataFrame, season: str) -> SeasonSchedule:
        teams: dict[str, Team] = {}
        games_by_id: dict[str, dict] = {}

        for _, row in df.iterrows():
            abbr = normalize_abbr(str(row["TEAM_ABBREVIATION"]))
            teams[abbr] = Team(
                id=abbr,
                name=str(row["TEAM_NAME"]),
                abbreviation=abbr,
            )
            game_id = str(row["GAME_ID"])
            matchup = str(row["MATCHUP"])
            is_home = " vs. " in matchup or " vs " in matchup
            pts = int(row["PTS"]) if pd.notna(row["PTS"]) else None
            game_date = self._parse_date(row["GAME_DATE"])

            entry = games_by_id.setdefault(
                game_id,
                {
                    "id": game_id,
                    "date": game_date,
                    "home_team_id": None,
                    "away_team_id": None,
                    "home_score": None,
                    "away_score": None,
                    "season": season,
                },
            )
            if is_home:
                entry["home_team_id"] = abbr
                entry["home_score"] = pts
            else:
                entry["away_team_id"] = abbr
                entry["away_score"] = pts

        games: list[Game] = []
        for g in games_by_id.values():
            if not g["home_team_id"] or not g["away_team_id"]:
                continue
            games.append(
                Game(
                    id=g["id"],
                    date=g["date"],
                    home_team_id=g["home_team_id"],
                    away_team_id=g["away_team_id"],
                    home_score=g["home_score"],
                    away_score=g["away_score"],
                    season=season,
                )
            )

        games.sort(key=lambda x: (x.date, x.id))
        team_list = sorted(teams.values(), key=lambda t: t.name)
        return SeasonSchedule(sport="nba", season=season, teams=team_list, games=games)

    @staticmethod
    def _parse_date(value) -> date:
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        text = str(value).strip()
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%b %d, %Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(text[:19] if "T" in text else text, fmt).date()
            except ValueError:
                continue
        return pd.to_datetime(value).date()
