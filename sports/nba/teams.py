"""NBA team name and abbreviation helpers."""

from __future__ import annotations

# Canonical abbreviations keyed by lowercase full team name (BRef style).
TEAM_NAME_TO_ABBR: dict[str, str] = {
    "atlanta hawks": "ATL",
    "boston celtics": "BOS",
    "brooklyn nets": "BKN",
    "charlotte hornets": "CHA",
    "chicago bulls": "CHI",
    "cleveland cavaliers": "CLE",
    "dallas mavericks": "DAL",
    "denver nuggets": "DEN",
    "detroit pistons": "DET",
    "golden state warriors": "GSW",
    "houston rockets": "HOU",
    "indiana pacers": "IND",
    "los angeles clippers": "LAC",
    "la clippers": "LAC",
    "los angeles lakers": "LAL",
    "la lakers": "LAL",
    "memphis grizzlies": "MEM",
    "miami heat": "MIA",
    "milwaukee bucks": "MIL",
    "minnesota timberwolves": "MIN",
    "new orleans pelicans": "NOP",
    "new york knicks": "NYK",
    "oklahoma city thunder": "OKC",
    "orlando magic": "ORL",
    "philadelphia 76ers": "PHI",
    "phoenix suns": "PHX",
    "portland trail blazers": "POR",
    "sacramento kings": "SAC",
    "san antonio spurs": "SAS",
    "toronto raptors": "TOR",
    "utah jazz": "UTA",
    "washington wizards": "WAS",
}

# Alternate abbreviations that should normalize to our canonical set.
ABBR_ALIASES: dict[str, str] = {
    "BRK": "BKN",
    "PHO": "PHX",
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
}


def normalize_abbr(abbr: str) -> str:
    a = abbr.strip().upper()
    return ABBR_ALIASES.get(a, a)


def team_id_from_name(name: str) -> str:
    key = name.strip().lower()
    if key in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[key]
    # fallback: first 3 letters
    return normalize_abbr("".join(ch for ch in name if ch.isalpha())[:3].upper())


def display_name(name: str) -> str:
    return name.strip()
