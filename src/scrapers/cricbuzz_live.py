"""
cricbuzz_live.py — Live match poller for Cricbuzz.

Cricbuzz uses a mix of server-side HTML rendering and internal JSON
commentary endpoints.  This module extracts live match data via two
complementary strategies:

1. **Match listing**: Parse the live-scores HTML page
   (``https://www.cricbuzz.com/cricket-match/live-scores``) to find active
   match IDs and basic team information.

2. **Scorecard**: Attempt the internal commentary JSON API first
   (``https://www.cricbuzz.com/api/cricket-match/{id}/full-commentary/1``),
   then fall back to parsing the live-scores HTML page.

Public API
----------
get_live_matches() -> List[Dict]
    Return a list of currently live match summaries.

fetch_live_scorecard(cricbuzz_match_id) -> Optional[Dict]
    Fetch current innings state for one live match.

poll_live_match(cricbuzz_match_id, match_db_id, session) -> int
    Upsert live innings data into the database and return the count of new
    deliveries inserted.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from src.data.db import Innings
from src.scrapers.http_client import get_page, scrape_delay

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cricbuzz URLs
# ---------------------------------------------------------------------------

_LIVE_SCORES_URL = "https://www.cricbuzz.com/cricket-match/live-scores"
_COMMENTARY_URL = (
    "https://www.cricbuzz.com/api/cricket-match/{match_id}/full-commentary/1"
)
_MATCH_SCORES_URL = "https://www.cricbuzz.com/live-cricket-scores/{match_id}"

# ---------------------------------------------------------------------------
# Match ID patterns — Cricbuzz embeds IDs in URLs and data attributes
# ---------------------------------------------------------------------------

# e.g. /live-cricket-scores/12345/india-vs-australia
_MATCH_ID_RE = re.compile(r"/live-cricket-scores/(\d+)")
# Fallback: data-match-id attribute
_DATA_MATCH_RE = re.compile(r'data-match-id="(\d+)"')
# JSON embedded in <script> tags on the live page
_JSON_MATCH_RE = re.compile(r'"matchId"\s*:\s*(\d+)')


# ---------------------------------------------------------------------------
# get_live_matches
# ---------------------------------------------------------------------------

def get_live_matches() -> List[Dict]:
    """
    Scrape the Cricbuzz live-scores landing page and return all active matches.

    The function extracts match IDs from anchor ``href`` attributes matching
    ``/live-cricket-scores/{id}``, then reads team names and status from the
    surrounding ``div.cb-mtch-lst`` elements.

    Returns
    -------
    List[Dict]
        Each dict has keys: ``cricbuzz_id`` (int), ``team_a`` (str),
        ``team_b`` (str), ``match_type`` (str), ``status`` (str).
        Returns an empty list on any failure — never raises.
    """
    logger.info("Fetching live matches from %s", _LIVE_SCORES_URL)
    try:
        response = get_page(_LIVE_SCORES_URL)
        return _parse_live_scores_page(response.text)
    except Exception as exc:
        logger.warning("get_live_matches failed: %s", exc)
        return []


def _parse_live_scores_page(html: str) -> List[Dict]:
    """
    Extract live match metadata from the Cricbuzz live-scores HTML.

    Cricbuzz renders server-side HTML with class-based selectors.  The match
    list container uses ``div.cb-mtch-lst`` or ``div.cb-col-100.cb-ltst-wgt-hdr``.
    Each match has an anchor with the URL pattern
    ``/live-cricket-scores/{id}/{slug}``.
    """
    soup = BeautifulSoup(html, "lxml")
    matches: List[Dict] = []
    seen_ids: set = set()

    # Primary selector — actual live match blocks
    match_cards = soup.select("div.cb-mtch-lst.cb-col.cb-col-100.cb-tms-itm")
    if not match_cards:
        # Fallback: any div that contains a live-cricket-scores link
        match_cards = soup.select("div.cb-col-100")

    for card in match_cards:
        link = card.find("a", href=_MATCH_ID_RE)
        if not link:
            continue

        m = _MATCH_ID_RE.search(link["href"])
        if not m:
            continue
        match_id = int(m.group(1))

        if match_id in seen_ids:
            continue
        seen_ids.add(match_id)

        # Teams from h3/h2 or span elements
        team_a, team_b = _extract_teams_from_card(card)

        # Status text
        status_el = card.find(class_=re.compile(r"cb-text-(live|complete|preview)"))
        status = status_el.get_text(strip=True) if status_el else "Unknown"

        # Match type (T20, ODI, Test) — usually in a span or the card heading
        match_type = _extract_match_type_from_card(card)

        matches.append(
            {
                "cricbuzz_id": match_id,
                "team_a": team_a,
                "team_b": team_b,
                "match_type": match_type,
                "status": status,
            }
        )

    if not matches:
        # Last-resort: regex scan the raw HTML for match IDs
        matches = _fallback_parse_live_html(html, seen_ids)

    logger.info("Found %d live matches on Cricbuzz", len(matches))
    return matches


def _extract_teams_from_card(card) -> tuple:
    """Return ``(team_a, team_b)`` from a Cricbuzz match card element."""
    team_elements = card.select("div.cb-hmscg-tm-nm, span.cb-team-name, h3.cb-lv-scrs-well-batsmen")
    names = [el.get_text(strip=True) for el in team_elements if el.get_text(strip=True)]

    if len(names) >= 2:
        return names[0], names[1]

    # Fallback: first two <a> tag texts after the card link
    anchors = card.find_all("a", limit=3)
    texts = [a.get_text(strip=True) for a in anchors if a.get_text(strip=True)]
    if len(texts) >= 2:
        return texts[0], texts[1]

    return "Team A", "Team B"


def _extract_match_type_from_card(card) -> str:
    """Guess match type (T20 / ODI / Test) from card text."""
    text = card.get_text(" ", strip=True).upper()
    if "TEST" in text:
        return "Test"
    if "ODI" in text or "ONE DAY" in text:
        return "ODI"
    if "T20" in text or "IPL" in text or "TWENTY20" in text:
        return "T20"
    return "T20"  # Cricbuzz predominantly hosts T20 live


def _fallback_parse_live_html(html: str, seen_ids: set) -> List[Dict]:
    """Last-resort: extract match IDs via regex and return minimal dicts."""
    matches: List[Dict] = []
    for m in _MATCH_ID_RE.finditer(html):
        mid = int(m.group(1))
        if mid not in seen_ids:
            seen_ids.add(mid)
            matches.append(
                {
                    "cricbuzz_id": mid,
                    "team_a": "Unknown",
                    "team_b": "Unknown",
                    "match_type": "T20",
                    "status": "Live",
                }
            )
    return matches


# ---------------------------------------------------------------------------
# fetch_live_scorecard
# ---------------------------------------------------------------------------

def fetch_live_scorecard(cricbuzz_match_id: int) -> Optional[Dict]:
    """
    Fetch the current scorecard for one live Cricbuzz match.

    Attempts the internal commentary JSON API first.  On failure (or if the
    API returns no innings data), falls back to parsing the live-scores page
    HTML.

    Parameters
    ----------
    cricbuzz_match_id:
        Cricbuzz numeric match identifier.

    Returns
    -------
    dict or None
        Scorecard dict with structure::

            {
                "team_a": str,
                "team_b": str,
                "innings": [
                    {
                        "batting_team": str,
                        "total_runs": int,
                        "total_wickets": int,
                        "total_overs": float,
                        "current_batsmen": [str, ...],
                        "current_bowler": str,
                        "recent_balls": [str, ...],
                    },
                    ...
                ]
            }

        Returns ``None`` on failure — never raises.
    """
    logger.info("fetch_live_scorecard: cricbuzz_match_id=%d", cricbuzz_match_id)

    # Strategy 1: internal commentary JSON API
    scorecard = _fetch_from_commentary_api(cricbuzz_match_id)
    if scorecard:
        return scorecard

    # Strategy 2: parse the live-scores HTML page
    logger.debug("Falling back to HTML parse for match %d", cricbuzz_match_id)
    return _fetch_from_html(cricbuzz_match_id)


def _fetch_from_commentary_api(cricbuzz_match_id: int) -> Optional[Dict]:
    """
    Try to fetch scorecard from the Cricbuzz internal commentary API.

    The API endpoint sometimes requires additional headers or session cookies
    and may return 403 / 404 for certain matches.  All failures are handled
    gracefully.
    """
    url = _COMMENTARY_URL.format(match_id=cricbuzz_match_id)
    logger.debug("Trying commentary API: %s", url)
    try:
        response = get_page(url)
        if response.status_code != 200:
            return None
        data = response.json()
        return _parse_commentary_api_response(data)
    except Exception as exc:
        logger.debug("Commentary API failed for %d: %s", cricbuzz_match_id, exc)
        return None


def _parse_commentary_api_response(data: Dict) -> Optional[Dict]:
    """
    Normalise the Cricbuzz commentary API JSON into our scorecard schema.

    The API response structure::

        {
            "matchHeader": {
                "team1": {"name": "India"}, "team2": {"name": "Australia"}
            },
            "miniscore": {
                "inningsId": 1,
                "batTeam": {"teamScore": {...}},
                "batsmanStriker": {"batName": "Virat Kohli", ...},
                "batsmanNonStriker": {...},
                "bowlerStriker": {"bowlName": "Mitchell Starc", ...},
                "recentOvsStats": "0 4 W 6 1 .",
            }
        }
    """
    try:
        header = data.get("matchHeader", {})
        team_a = header.get("team1", {}).get("name", "Team A")
        team_b = header.get("team2", {}).get("name", "Team B")

        miniscore = data.get("miniscore", {})
        if not miniscore:
            return None

        bat_team_data = miniscore.get("batTeam", {})
        team_score = bat_team_data.get("teamScore", {})

        innings_summary = {
            "batting_team": bat_team_data.get("teamName", ""),
            "total_runs": int(team_score.get("runs", 0)),
            "total_wickets": int(team_score.get("wickets", 0)),
            "total_overs": float(team_score.get("overs", 0.0)),
            "current_batsmen": [],
            "current_bowler": "",
            "recent_balls": [],
        }

        # Current batsmen
        for batter_key in ("batsmanStriker", "batsmanNonStriker"):
            batter = miniscore.get(batter_key, {})
            if batter.get("batName"):
                innings_summary["current_batsmen"].append(batter["batName"])

        # Current bowler
        bowler = miniscore.get("bowlerStriker", {})
        innings_summary["current_bowler"] = bowler.get("bowlName", "")

        # Recent balls
        recent_str = miniscore.get("recentOvsStats", "")
        innings_summary["recent_balls"] = recent_str.split() if recent_str else []

        return {
            "team_a": team_a,
            "team_b": team_b,
            "innings": [innings_summary],
        }
    except Exception as exc:
        logger.debug("_parse_commentary_api_response failed: %s", exc)
        return None


def _fetch_from_html(cricbuzz_match_id: int) -> Optional[Dict]:
    """Parse live scorecard by scraping the match-specific HTML page."""
    url = _MATCH_SCORES_URL.format(match_id=cricbuzz_match_id)
    logger.debug("Fetching match HTML: %s", url)
    try:
        response = get_page(url)
        return _parse_match_html(response.text)
    except Exception as exc:
        logger.warning(
            "_fetch_from_html failed for match %d: %s", cricbuzz_match_id, exc
        )
        return None


def _parse_match_html(html: str) -> Optional[Dict]:
    """
    Extract current innings state from a Cricbuzz match HTML page.

    Cricbuzz uses ``div.cb-lv-scrs-well`` for current innings scores and
    ``div.cb-lv-scrs-well-batsmen`` for the batting table.
    """
    soup = BeautifulSoup(html, "lxml")
    innings_list: List[Dict] = []

    # Team names from the header
    team_els = soup.select("a.cb-nav-sublinks-itm") or soup.select("span.cb-team-name")
    team_names = [el.get_text(strip=True) for el in team_els[:2]]
    team_a = team_names[0] if len(team_names) > 0 else "Team A"
    team_b = team_names[1] if len(team_names) > 1 else "Team B"

    # Score blocks
    score_blocks = soup.select("div.cb-lv-scrs-well, div.cb-col-100.cb-ltst-wgt-hdr")
    for block in score_blocks:
        score_text = block.get_text(" ", strip=True)
        # Look for pattern like "India 245/8 (45.2 Ov)"
        score_match = re.search(
            r"(\d{1,4})/(\d{1,2})\s*\(([0-9.]+)\s*[Oo]v",
            score_text,
        )
        if score_match:
            team_name_el = block.find(class_=re.compile(r"cb-team|cb-nav"))
            batting_team = team_name_el.get_text(strip=True) if team_name_el else team_a

            innings_list.append(
                {
                    "batting_team": batting_team,
                    "total_runs": int(score_match.group(1)),
                    "total_wickets": int(score_match.group(2)),
                    "total_overs": float(score_match.group(3)),
                    "current_batsmen": [],
                    "current_bowler": "",
                    "recent_balls": [],
                }
            )

    if not innings_list:
        logger.debug("No innings score blocks found in match HTML")
        return None

    return {"team_a": team_a, "team_b": team_b, "innings": innings_list}


# ---------------------------------------------------------------------------
# poll_live_match
# ---------------------------------------------------------------------------

def poll_live_match(
    cricbuzz_match_id: int,
    match_db_id: int,
    session: Session,
) -> int:
    """
    Fetch the current live scorecard and upsert Innings rows into the DB.

    This function is designed to be called on a schedule (e.g. every 60
    seconds during a live match).  It is fully idempotent — calling it
    multiple times with the same match state produces the same DB state.

    Parameters
    ----------
    cricbuzz_match_id:
        Cricbuzz numeric match identifier.
    match_db_id:
        Internal DB primary key of the corresponding Match record.
    session:
        Active SQLAlchemy session.  The caller is responsible for committing.

    Returns
    -------
    int
        Count of new Innings rows inserted (usually 0 after the first call for
        a given innings, since subsequent calls update existing rows).
        Returns 0 on any failure — never raises.
    """
    logger.info(
        "poll_live_match: cricbuzz_id=%d db_match_id=%d",
        cricbuzz_match_id,
        match_db_id,
    )
    try:
        scorecard = fetch_live_scorecard(cricbuzz_match_id)
        if not scorecard:
            logger.debug(
                "No scorecard returned for cricbuzz match %d", cricbuzz_match_id
            )
            return 0

        new_innings_count = 0

        for idx, inn_data in enumerate(scorecard.get("innings", []), start=1):
            existing: Optional[Innings] = (
                session.query(Innings)
                .filter_by(match_id=match_db_id, innings_number=idx)
                .first()
            )

            if existing is None:
                innings_obj = Innings(
                    match_id=match_db_id,
                    innings_number=idx,
                    batting_team=inn_data.get("batting_team", "Unknown"),
                    bowling_team=_infer_bowling_team(
                        inn_data.get("batting_team", ""),
                        scorecard.get("team_a", ""),
                        scorecard.get("team_b", ""),
                    ),
                    total_runs=inn_data.get("total_runs", 0),
                    total_wickets=inn_data.get("total_wickets", 0),
                    total_overs=inn_data.get("total_overs", 0.0),
                )
                session.add(innings_obj)
                new_innings_count += 1
                logger.info(
                    "New innings row created: match_id=%d innings=%d runs=%d/%d",
                    match_db_id,
                    idx,
                    inn_data.get("total_runs", 0),
                    inn_data.get("total_wickets", 0),
                )
            else:
                # Update running totals
                existing.total_runs = inn_data.get("total_runs", existing.total_runs)
                existing.total_wickets = inn_data.get("total_wickets", existing.total_wickets)
                existing.total_overs = inn_data.get("total_overs", existing.total_overs)
                logger.debug(
                    "Updated innings %d: %d/%d in %.1f ov",
                    idx,
                    existing.total_runs,
                    existing.total_wickets,
                    existing.total_overs,
                )

        session.flush()
        return new_innings_count

    except Exception as exc:
        logger.error(
            "poll_live_match failed for cricbuzz_id=%d: %s",
            cricbuzz_match_id,
            exc,
            exc_info=True,
        )
        try:
            session.rollback()
        except Exception:
            pass
        return 0


def _infer_bowling_team(batting_team: str, team_a: str, team_b: str) -> str:
    """Return the team that is not batting, using string matching."""
    if not batting_team:
        return team_b or "Unknown"
    if batting_team.lower() == team_a.lower():
        return team_b
    if batting_team.lower() == team_b.lower():
        return team_a
    # Fallback: return whichever team name is not the batting team
    return team_b if team_a.lower() in batting_team.lower() else team_a
