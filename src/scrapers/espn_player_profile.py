"""
espn_player_profile.py — Fetch player career statistics from ESPNcricinfo.

ESPNcricinfo player profile pages are Next.js applications that embed all
page data in a ``<script id="__NEXT_DATA__">`` JSON blob — no JavaScript
rendering required.  The URL pattern that reliably redirects to the correct
player page is:

    https://www.espncricinfo.com/player/{espn_player_id}

ESPNcricinfo returns HTTP 301/302 to the canonical slug URL; the
``requests`` session follows the redirect automatically.

Public API
----------
scrape_player_profile(espn_player_id) -> Optional[Dict]
    Scrape career stats for one player.  Returns a dict with T20 / ODI / Test
    sub-dicts, or ``None`` on any failure.

update_player_features_from_profile(player_id, espn_player_id, session) -> bool
    Calls :func:`scrape_player_profile` and writes/updates
    :class:`~src.data.db.PlayerFeature` snapshots for today's date.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from typing import Dict, Optional

from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from src.data.db import PlayerFeature
from src.scrapers.http_client import get_page

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ESPN player profile URL
# ---------------------------------------------------------------------------

_PROFILE_URL = "https://www.espncricinfo.com/player/{player_id}"

# Standard format labels used in our DB
_FORMATS = ("T20", "ODI", "Test")

# Blank career-stats sub-dict — returned when data is absent
_EMPTY_FORMAT_STATS: Dict = {
    "matches": 0,
    "innings": 0,
    "runs": 0,
    "avg": None,
    "sr": None,
    "wickets": 0,
    "bowling_avg": None,
    "economy": None,
}


# ---------------------------------------------------------------------------
# __NEXT_DATA__ helpers (local, not imported, to keep modules independent)
# ---------------------------------------------------------------------------

def _extract_next_data(html: str) -> Optional[Dict]:
    """Parse the ``__NEXT_DATA__`` JSON tag from a Next.js page."""
    try:
        soup = BeautifulSoup(html, "lxml")
        tag = soup.find("script", {"id": "__NEXT_DATA__", "type": "application/json"})
        if tag is None:
            return None
        return json.loads(tag.string)
    except Exception as exc:
        logger.debug("_extract_next_data failed: %s", exc)
        return None


def _dig(obj: object, *keys) -> object:
    """Safely traverse nested dicts/lists; returns ``None`` on any miss."""
    current = obj
    for key in keys:
        try:
            if isinstance(current, dict):
                current = current[key]
            elif isinstance(current, list):
                current = current[int(key)]
            else:
                return None
        except (KeyError, IndexError, TypeError, ValueError):
            return None
    return current


# ---------------------------------------------------------------------------
# Career-stats extraction
# ---------------------------------------------------------------------------

def _build_empty_profile(espn_player_id: int) -> Dict:
    """Return a skeletal profile dict with zeroed stats for all formats."""
    return {
        "espn_player_id": espn_player_id,
        "T20": dict(_EMPTY_FORMAT_STATS),
        "ODI": dict(_EMPTY_FORMAT_STATS),
        "Test": dict(_EMPTY_FORMAT_STATS),
    }


def _extract_career_stats(next_data: Dict, espn_player_id: int) -> Dict:
    """
    Navigate the ``__NEXT_DATA__`` structure to find career stats tables.

    ESPN stores career stats under various paths depending on the page
    version.  This function tries multiple candidate locations and falls back
    gracefully to an empty profile.
    """
    profile = _build_empty_profile(espn_player_id)

    props = _dig(next_data, "props", "pageProps")
    if not props:
        logger.debug("No pageProps found in __NEXT_DATA__")
        return profile

    # ESPN places career averages under different keys across page versions.
    # Try them all before giving up.
    stats_candidates = [
        _dig(props, "playerStats"),
        _dig(props, "career", "averages"),
        _dig(props, "content", "career"),
        _dig(props, "data", "career"),
        _dig(props, "careerAverages"),
        _dig(props, "stats"),
    ]

    raw_stats = next((s for s in stats_candidates if s), None)

    if raw_stats is None:
        # Deep-search for a dict that has format-keyed sub-objects
        raw_stats = _deep_search_stats(props)

    if not raw_stats:
        logger.debug("No career stats structure found in __NEXT_DATA__")
        return profile

    # raw_stats may be a list of format rows or a dict keyed by format
    if isinstance(raw_stats, list):
        profile = _parse_stats_list(raw_stats, profile)
    elif isinstance(raw_stats, dict):
        profile = _parse_stats_dict(raw_stats, profile)

    return profile


def _deep_search_stats(obj: object, depth: int = 0) -> Optional[object]:
    """
    Recursively look for an object that appears to contain career stats.

    Heuristic: a dict with a sub-key whose name looks like a cricket format
    (``t20i``, ``odi``, ``tests``, ``firstClass``) or a list of dicts that
    have both ``matches`` and ``runs`` keys.
    """
    if depth > 7:
        return None

    FORMAT_HINTS = {"t20i", "odi", "tests", "test", "firstclass", "t20", "ipl"}

    if isinstance(obj, dict):
        lower_keys = {k.lower(): k for k in obj}
        if any(k in lower_keys for k in FORMAT_HINTS):
            return obj
        for v in obj.values():
            result = _deep_search_stats(v, depth + 1)
            if result is not None:
                return result

    if isinstance(obj, list) and len(obj) > 0:
        sample = obj[0]
        if isinstance(sample, dict) and "matches" in sample:
            return obj
        for item in obj:
            result = _deep_search_stats(item, depth + 1)
            if result is not None:
                return result

    return None


# Mappings from ESPN format key variants to our standard labels
_FORMAT_KEY_MAP: Dict[str, str] = {
    "t20i": "T20",
    "t20": "T20",
    "odi": "ODI",
    "odis": "ODI",
    "list_a": "ODI",
    "tests": "Test",
    "test": "Test",
    "firstclass": "Test",
    "first_class": "Test",
}


def _parse_stats_dict(raw: Dict, profile: Dict) -> Dict:
    """
    Parse a dict that maps format names to stat sub-dicts.

    Example structure::

        {
            "t20i": {"matches": 85, "innings": 78, "runs": 2100, ...},
            "odi":  {"matches": 102, ...},
            "tests": {"matches": 45, ...},
        }
    """
    for key, value in raw.items():
        fmt = _FORMAT_KEY_MAP.get(key.lower().replace("-", "_").replace(" ", "_"))
        if fmt and isinstance(value, dict):
            profile[fmt] = _normalise_stat_block(value)
    return profile


def _parse_stats_list(raw: list, profile: Dict) -> Dict:
    """
    Parse a list of per-format stat rows.

    Each row is expected to have a ``type`` or ``class_type`` field indicating
    the format, plus numeric stats fields.
    """
    for row in raw:
        if not isinstance(row, dict):
            continue
        # Identify format
        fmt_raw = (
            row.get("type")
            or row.get("class_type")
            or row.get("format")
            or row.get("match_type")
            or ""
        )
        fmt = _FORMAT_KEY_MAP.get(fmt_raw.lower().replace("-", "_").replace(" ", "_"))
        if fmt:
            profile[fmt] = _normalise_stat_block(row)
    return profile


def _normalise_stat_block(raw: Dict) -> Dict:
    """
    Convert a raw ESPN stats sub-dict into our standard format.

    ESPN field names vary between ``Mat``, ``matches``, ``m``, etc.
    """
    def _v(*keys):
        """Return the first non-None value for the given keys."""
        for k in keys:
            v = raw.get(k)
            if v not in (None, "", "-", "–"):
                return v
        return None

    def _float_or_none(val) -> Optional[float]:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _int_or_zero(val) -> int:
        try:
            return int(float(str(val)))
        except (TypeError, ValueError):
            return 0

    return {
        "matches": _int_or_zero(_v("matches", "Mat", "m", "Matches")),
        "innings": _int_or_zero(_v("innings", "Inn", "Inns", "i")),
        "runs": _int_or_zero(_v("runs", "Runs", "run", "r")),
        "avg": _float_or_none(_v("average", "Avg", "ave", "batting_average", "bat_avg")),
        "sr": _float_or_none(_v("strike_rate", "SR", "sr", "bat_sr")),
        "wickets": _int_or_zero(_v("wickets", "Wkts", "wkts", "wkt")),
        "bowling_avg": _float_or_none(_v("bowling_average", "BowlAve", "bowl_avg", "Econ")),
        "economy": _float_or_none(_v("economy", "Econ", "Economy", "econ")),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_player_profile(espn_player_id: int) -> Optional[Dict]:
    """
    Fetch career statistics for one player from ESPNcricinfo.

    The player profile URL ``https://www.espncricinfo.com/player/{id}``
    redirects to the canonical slug URL; the ``requests`` session follows the
    redirect transparently.

    Parameters
    ----------
    espn_player_id:
        ESPN numeric player identifier (e.g. ``28081`` for Sachin Tendulkar).

    Returns
    -------
    dict or None
        Profile dict with structure::

            {
                "espn_player_id": 28081,
                "T20": {"matches": N, "innings": N, "runs": N, "avg": F,
                        "sr": F, "wickets": N, "bowling_avg": F, "economy": F},
                "ODI": { ... },
                "Test": { ... },
            }

        Returns ``None`` on any network, redirect, or parse failure.
    """
    url = _PROFILE_URL.format(player_id=espn_player_id)
    logger.info("Scraping player profile: %s", url)

    try:
        response = get_page(url)
    except Exception as exc:
        logger.warning("Failed to fetch player profile %d: %s", espn_player_id, exc)
        return None

    next_data = _extract_next_data(response.text)
    if not next_data:
        logger.warning(
            "No __NEXT_DATA__ found on player profile page %d", espn_player_id
        )
        return None

    try:
        profile = _extract_career_stats(next_data, espn_player_id)
        logger.info(
            "Player %d: T20 matches=%d, ODI matches=%d, Test matches=%d",
            espn_player_id,
            profile["T20"]["matches"],
            profile["ODI"]["matches"],
            profile["Test"]["matches"],
        )
        return profile
    except Exception as exc:
        logger.warning(
            "Error extracting career stats for player %d: %s", espn_player_id, exc
        )
        return None


def update_player_features_from_profile(
    player_id: int,
    espn_player_id: int,
    session: Session,
) -> bool:
    """
    Scrape ESPN profile and write/update :class:`~src.data.db.PlayerFeature`
    snapshots for today's date.

    One snapshot row is created (or updated) per cricket format (T20, ODI,
    Test).  The composite unique constraint ``(player_id, snapshot_date, format)``
    ensures idempotency — re-running this function on the same day updates
    the existing rows rather than creating duplicates.

    Parameters
    ----------
    player_id:
        Internal DB primary key of the Player record.
    espn_player_id:
        ESPN numeric player identifier used to fetch the profile.
    session:
        Active SQLAlchemy session.  The caller is responsible for committing.

    Returns
    -------
    bool
        ``True`` if at least one PlayerFeature snapshot was persisted;
        ``False`` on any failure.
    """
    logger.info(
        "update_player_features_from_profile: player_id=%d espn_id=%d",
        player_id,
        espn_player_id,
    )

    profile = scrape_player_profile(espn_player_id)
    if not profile:
        logger.warning(
            "No profile data for espn_player_id=%d — skipping feature update",
            espn_player_id,
        )
        return False

    today = date.today().isoformat()
    saved = 0

    try:
        for fmt in _FORMATS:
            stats = profile.get(fmt, {})
            if not stats:
                continue

            # Upsert
            feat: Optional[PlayerFeature] = (
                session.query(PlayerFeature)
                .filter_by(player_id=player_id, snapshot_date=today, format=fmt)
                .first()
            )
            if feat is None:
                feat = PlayerFeature(
                    player_id=player_id,
                    snapshot_date=today,
                    format=fmt,
                )
                session.add(feat)

            feat.n_matches = stats.get("matches", 0)
            feat.n_innings = stats.get("innings", 0)
            feat.batting_avg = stats.get("avg")
            feat.strike_rate = stats.get("sr")
            feat.bowling_avg = stats.get("bowling_avg")
            feat.bowling_econ = stats.get("economy")
            feat.feature_json = json.dumps(stats)
            saved += 1

        session.flush()
        logger.info(
            "Saved %d PlayerFeature snapshots for player_id=%d", saved, player_id
        )
        return saved > 0

    except Exception as exc:
        logger.error(
            "update_player_features_from_profile failed for player_id=%d: %s",
            player_id,
            exc,
            exc_info=True,
        )
        try:
            session.rollback()
        except Exception:
            pass
        return False
