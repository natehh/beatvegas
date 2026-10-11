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
    "CHO": "CHA",
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
}

# Our canonical abbr → Basketball-Reference URL slug.
TO_BREF_ABBR: dict[str, str] = {
    "BKN": "BRK",
    "CHA": "CHO",
    "PHX": "PHO",
}

# Basketball-Reference slug → our canonical abbr.
FROM_BREF_ABBR: dict[str, str] = {v: k for k, v in TO_BREF_ABBR.items()}


def to_bref_abbr(abbr: str) -> str:
    a = normalize_abbr(abbr)
    return TO_BREF_ABBR.get(a, a)


def from_bref_abbr(abbr: str) -> str:
    a = abbr.strip().upper()
    return normalize_abbr(FROM_BREF_ABBR.get(a, a))


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
