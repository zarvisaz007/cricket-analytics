"""
espn_historical.py — Discover historical match IDs from ESPNcricinfo.

ESPNcricinfo is built with Next.js and embeds ALL page data inside a
``<script id="__NEXT_DATA__" type="application/json">`` tag.  This means
every match-list page can be parsed with plain ``requests`` + ``BeautifulSoup``
— no JavaScript rendering or browser automation is required.

Public API
----------
scrape_match_list(match_type, year) -> List[Dict]
    Scrape one format/year combination from the ESPN records pages and return
    a list of normalised match dicts.

discover_matches(start_year, end_year, formats) -> Iterator[Dict]
    Yield every match dict for the given year range and format list.
    Tracks progress in ``data/scrape_progress.json`` so re-runs skip already-
    processed match IDs.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from bs4 import BeautifulSoup

from src.scrapers.http_client import get_page, scrape_delay

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ESPN_BASE = "https://www.espncricinfo.com"
RECORDS_URL_TEMPLATE = (
    "https://www.espncricinfo.com/records/year/match-results/{year}/{match_type}"
)

DEFAULT_FORMATS: List[str] = [
    "test-matches",
    "one-day-internationals",
    "twenty20-internationals",
]

# Maps ESPN URL slug → normalised match_type label stored in DB
FORMAT_LABEL_MAP: Dict[str, str] = {
    "test-matches": "Test",
    "one-day-internationals": "ODI",
    "twenty20-internationals": "T20",
    "ipl": "T20",
}

PROGRESS_FILE: Path = Path("data/scrape_progress.json")
SAVE_EVERY: int = 10  # persist progress every N matches


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def _load_progress() -> Dict[str, bool]:
    """
    Load the progress dict from ``data/scrape_progress.json``.

    The dict maps ``str(espn_match_id)`` → ``True`` for every match that has
    already been fully processed in a previous run.
    """
    if PROGRESS_FILE.exists():
        try:
            with PROGRESS_FILE.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load progress file %s: %s", PROGRESS_FILE, exc)
    return {}


def _save_progress(progress: Dict[str, bool]) -> None:
    """Persist *progress* dict to disk, creating parent dirs as needed."""
    try:
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with PROGRESS_FILE.open("w", encoding="utf-8") as fh:
            json.dump(progress, fh)
    except OSError as exc:
        logger.warning("Could not save progress file %s: %s", PROGRESS_FILE, exc)


# ---------------------------------------------------------------------------
# __NEXT_DATA__ extraction helpers
# ---------------------------------------------------------------------------

def _extract_next_data(html: str) -> Optional[Dict]:
    """
    Parse the ``__NEXT_DATA__`` JSON blob embedded in a Next.js page.

    Parameters
    ----------
    html:
        Raw HTML string of the page.

    Returns
    -------
    dict or None
        Parsed JSON object, or ``None`` if the tag is absent / unparseable.
    """
    try:
        soup = BeautifulSoup(html, "lxml")
        tag = soup.find("script", {"id": "__NEXT_DATA__", "type": "application/json"})
        if tag is None:
            logger.warning("__NEXT_DATA__ script tag not found in page")
            return None
        return json.loads(tag.string)
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Failed to parse __NEXT_DATA__: %s", exc)
        return None


def _dig(obj: object, *keys) -> object:
    """
    Safely traverse a nested dict/list structure.

    Each key in *keys* may be a string (for dict lookup) or an integer
    (for list index).  Returns ``None`` if any step fails.
    """
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
# Match-list extraction from ESPN __NEXT_DATA__
# ---------------------------------------------------------------------------

def _extract_match_rows(next_data: Dict, match_type: str, year: int) -> List[Dict]:
    """
    Navigate the ``__NEXT_DATA__`` structure to locate the list of matches.

    ESPN's JSON structure changes occasionally; this function attempts several
    known paths and falls back gracefully.

    Parameters
    ----------
    next_data:
        Parsed ``__NEXT_DATA__`` dict.
    match_type:
        ESPN URL slug (e.g. ``"test-matches"``).
    year:
        Calendar year being scraped.

    Returns
    -------
    List[Dict]
        Normalised match dicts.  Empty list if no matches could be extracted.
    """
    matches: List[Dict] = []
    format_label = FORMAT_LABEL_MAP.get(match_type, match_type)

    # Attempt to find the match content under common Next.js paths
    props = _dig(next_data, "props", "pageProps")
    if not props:
        # Some pages nest under dehydratedState
        props = _dig(next_data, "props", "pageProps", "dehydratedState")

    # Try several known keys that ESPN uses for match records tables
    table_data: Optional[list] = None
    candidate_keys = [
        "matchResults",
        "matchList",
        "content",
        "data",
        "matches",
        "results",
    ]
    for key in candidate_keys:
        candidate = _dig(props, key)
        if isinstance(candidate, list) and len(candidate) > 0:
            table_data = candidate
            break

    # Deeper search: walk all values looking for a list of dicts with known fields
    if table_data is None and isinstance(props, dict):
        table_data = _search_for_match_list(props)

    if not table_data:
        logger.warning(
            "No match list found in __NEXT_DATA__ for %s %d", match_type, year
        )
        return []

    for row in table_data:
        if not isinstance(row, dict):
            continue
        try:
            parsed = _parse_match_row(row, match_type, format_label, year)
            if parsed:
                matches.append(parsed)
        except Exception as exc:
            logger.debug("Skipping malformed match row %r: %s", row, exc)

    return matches


def _search_for_match_list(obj: object, depth: int = 0) -> Optional[list]:
    """
    Recursively search *obj* for the first list whose first element looks like
    a match record (has ``id`` or ``objectId`` and at least one team field).
    """
    if depth > 8:
        return None

    if isinstance(obj, list) and len(obj) > 0:
        sample = obj[0]
        if isinstance(sample, dict):
            has_id = any(k in sample for k in ("id", "objectId", "matchId", "match_id"))
            has_team = any(
                k in sample
                for k in ("team1", "team2", "teams", "homeTeam", "awayTeam", "team_a")
            )
            if has_id or has_team:
                return obj

    if isinstance(obj, dict):
        for v in obj.values():
            result = _search_for_match_list(v, depth + 1)
            if result is not None:
                return result

    if isinstance(obj, list):
        for item in obj:
            result = _search_for_match_list(item, depth + 1)
            if result is not None:
                return result

    return None


def _parse_match_row(row: Dict, match_type: str, format_label: str, year: int) -> Optional[Dict]:
    """
    Normalise a raw match-row dict from ESPN's JSON into our standard schema.

    Returns ``None`` if a mandatory field (match ID) cannot be extracted.
    """
    # Extract match ID — ESPN uses several key names
    espn_id: Optional[int] = None
    for id_key in ("id", "objectId", "matchId", "match_id", "contentId"):
        raw_id = row.get(id_key)
        if raw_id is not None:
            try:
                espn_id = int(raw_id)
                break
            except (ValueError, TypeError):
                continue

    if espn_id is None:
        return None

    # Teams — various nested structures ESPN uses
    team_a = _extract_team_name(row, ("team1", "homeTeam", "team_a", "battingTeam"))
    team_b = _extract_team_name(row, ("team2", "awayTeam", "team_b", "fieldingTeam"))

    # Fallback: look inside a "teams" list
    if (team_a is None or team_b is None) and isinstance(row.get("teams"), list):
        teams_list = row["teams"]
        if len(teams_list) >= 1:
            team_a = team_a or _extract_team_name(teams_list[0], ("name", "longName", "shortName"))
        if len(teams_list) >= 2:
            team_b = team_b or _extract_team_name(teams_list[1], ("name", "longName", "shortName"))

    team_a = team_a or "Unknown"
    team_b = team_b or "Unknown"

    # Date
    match_date: Optional[str] = None
    for date_key in ("startDate", "startDateTime", "date", "match_date", "matchDate"):
        raw_date = row.get(date_key)
        if raw_date:
            match_date = str(raw_date)[:10]  # keep YYYY-MM-DD prefix
            break
    if not match_date:
        match_date = f"{year}-01-01"

    # Venue
    venue: str = "Unknown"
    venue_raw = row.get("venue") or row.get("ground") or row.get("stadium")
    if isinstance(venue_raw, dict):
        venue = (
            venue_raw.get("name")
            or venue_raw.get("longName")
            or venue_raw.get("displayName")
            or "Unknown"
        )
    elif isinstance(venue_raw, str):
        venue = venue_raw

    # Tournament / Series
    tournament: str = "Unknown"
    for t_key in ("tournament", "series", "seriesName", "competition", "event"):
        t_raw = row.get(t_key)
        if isinstance(t_raw, dict):
            tournament = t_raw.get("name") or t_raw.get("longName") or "Unknown"
            break
        elif isinstance(t_raw, str) and t_raw:
            tournament = t_raw
            break

    match_key = f"espn_{espn_id}"

    return {
        "espn_match_id": espn_id,
        "match_key": match_key,
        "team_a": team_a,
        "team_b": team_b,
        "match_date": match_date,
        "venue": venue,
        "tournament": tournament,
        "match_type": format_label,
        "source": "espn_scrape",
    }


def _extract_team_name(obj: Dict, keys: tuple) -> Optional[str]:
    """Try each key in *keys* against *obj* and return the first non-empty string."""
    for key in keys:
        raw = obj.get(key)
        if isinstance(raw, dict):
            for sub in ("name", "longName", "shortName", "displayName"):
                val = raw.get(sub)
                if val and isinstance(val, str):
                    return val
        elif isinstance(raw, str) and raw:
            return raw
    return None


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

def _get_pagination_urls(next_data: Dict, base_url: str) -> List[str]:
    """
    Extract any additional page URLs from the ``__NEXT_DATA__`` pagination info.

    Returns a list of URLs to scrape in addition to the first page.
    """
    urls: List[str] = []
    try:
        props = _dig(next_data, "props", "pageProps")
        pagination = None
        for key in ("pagination", "pager", "meta"):
            pagination = _dig(props, key)
            if pagination:
                break
        if not pagination:
            return []

        total_pages = int(
            pagination.get("totalPages")
            or pagination.get("total_pages")
            or pagination.get("pages")
            or 1
        )
        for page_num in range(2, total_pages + 1):
            urls.append(f"{base_url}?page={page_num}")
    except Exception:
        pass
    return urls


# ---------------------------------------------------------------------------
# Public scraper functions
# ---------------------------------------------------------------------------

def scrape_match_list(match_type: str, year: int) -> List[Dict]:
    """
    Scrape the ESPN records page for one format / year combination.

    Parameters
    ----------
    match_type:
        One of ``"test-matches"``, ``"one-day-internationals"``,
        ``"twenty20-internationals"``, ``"ipl"``.
    year:
        Four-digit calendar year.

    Returns
    -------
    List[Dict]
        Normalised match dicts.  Returns an empty list (never raises) on any
        parse or network failure.
    """
    url = RECORDS_URL_TEMPLATE.format(year=year, match_type=match_type)
    logger.info("Scraping ESPN match list: %s", url)

    try:
        response = get_page(url)
        next_data = _extract_next_data(response.text)
        if next_data is None:
            logger.warning("No __NEXT_DATA__ found at %s", url)
            return []

        matches = _extract_match_rows(next_data, match_type, year)
        logger.info(
            "Found %d matches for %s %d", len(matches), match_type, year
        )

        # Handle pagination
        additional_pages = _get_pagination_urls(next_data, url)
        for page_url in additional_pages:
            try:
                logger.info("Fetching paginated URL: %s", page_url)
                page_resp = get_page(page_url)
                page_data = _extract_next_data(page_resp.text)
                if page_data:
                    extra = _extract_match_rows(page_data, match_type, year)
                    logger.info("  + %d matches from %s", len(extra), page_url)
                    matches.extend(extra)
            except Exception as page_exc:
                logger.warning("Failed to scrape paginated page %s: %s", page_url, page_exc)

        return matches

    except Exception as exc:
        logger.warning(
            "scrape_match_list failed for %s %d: %s", match_type, year, exc
        )
        return []


def discover_matches(
    start_year: int = 2014,
    end_year: int = 2026,
    formats: Optional[List[str]] = None,
) -> Iterator[Dict]:
    """
    Yield normalised match dicts for every year/format in the specified range.

    Already-processed match IDs are tracked in ``data/scrape_progress.json``
    so repeated runs skip matches that have been fully ingested.  Progress is
    persisted to disk every :data:`SAVE_EVERY` newly-yielded matches.

    Parameters
    ----------
    start_year:
        First year to scrape (inclusive).  Defaults to 2014.
    end_year:
        Last year to scrape (inclusive).  Defaults to 2026.
    formats:
        List of ESPN format slugs to scrape.  Defaults to
        ``["test-matches", "one-day-internationals", "twenty20-internationals"]``.

    Yields
    ------
    Dict
        One dict per previously-unseen match with keys:
        ``espn_match_id``, ``match_key``, ``team_a``, ``team_b``,
        ``match_date``, ``venue``, ``tournament``, ``match_type``, ``source``.
    """
    if formats is None:
        formats = list(DEFAULT_FORMATS)

    progress: Dict[str, bool] = _load_progress()
    newly_yielded: int = 0

    for year in range(start_year, end_year + 1):
        for fmt in formats:
            batch = scrape_match_list(fmt, year)
            for match in batch:
                match_id_str = str(match["espn_match_id"])
                if progress.get(match_id_str):
                    logger.debug("Skipping already-processed match %s", match_id_str)
                    continue

                yield match

                # Mark as processed and periodically flush
                progress[match_id_str] = True
                newly_yielded += 1
                if newly_yielded % SAVE_EVERY == 0:
                    _save_progress(progress)
                    logger.info(
                        "Progress saved — %d new matches yielded so far",
                        newly_yielded,
                    )

    # Final flush
    _save_progress(progress)
    logger.info(
        "discover_matches complete — %d new matches yielded total", newly_yielded
    )
