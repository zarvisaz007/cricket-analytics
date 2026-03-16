"""
espn_scorecard.py — Fetch and parse detailed scorecard data from ESPNcricinfo.

ESPNcricinfo exposes a legacy JSON API that returns complete, structured
batting/bowling data without any HTML parsing:

  Scorecard  : https://www.espncricinfo.com/matches/engine/match/{id}.json
  Ball-by-ball: https://www.espncricinfo.com/matches/engine/match/{id}.json?type=comms

Both endpoints return plain JSON and work with a standard ``requests`` GET.

Public API
----------
scrape_scorecard(espn_match_id) -> Optional[Dict]
    Fetch full scorecard data: match_info, innings summaries, batting/bowling
    line-by-line stats.

scrape_ball_by_ball(espn_match_id) -> Optional[List[Dict]]
    Fetch delivery-level commentary.  Returns ``None`` if unavailable.

parse_scorecard_to_db(espn_match_id, session) -> bool
    Orchestrates both fetches and upserts Match, Innings, PlayerStat, and
    Delivery rows into the database.  Returns ``True`` on success.
"""

from __future__ import annotations

import difflib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from src.data.db import Innings, Match, Player, PlayerStat
from src.scrapers.http_client import get_page

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ESPN JSON API endpoints
# ---------------------------------------------------------------------------

_ENGINE_URL = "https://www.espncricinfo.com/matches/engine/match/{match_id}.json"
_COMMS_URL = "https://www.espncricinfo.com/matches/engine/match/{match_id}.json?type=comms"

# ---------------------------------------------------------------------------
# Scorecard scraper
# ---------------------------------------------------------------------------

def scrape_scorecard(espn_match_id: int) -> Optional[Dict]:
    """
    Fetch the full scorecard for *espn_match_id* from the ESPN engine API.

    The response JSON contains ``match``, ``innings``, ``team`` and related
    sub-objects.  This function normalises the raw payload into a flat,
    predictable structure that the rest of the pipeline can consume directly.

    Parameters
    ----------
    espn_match_id:
        ESPN numeric match identifier.

    Returns
    -------
    dict or None
        Structured scorecard dict with keys:

        - ``match_info``: date, venue, result, toss, teams
        - ``innings``: list of innings summary dicts
        - ``batting``: dict keyed by innings index → list of batting line dicts
        - ``bowling``: dict keyed by innings index → list of bowling line dicts

        Returns ``None`` on any network or parse failure (logs WARNING).
    """
    url = _ENGINE_URL.format(match_id=espn_match_id)
    logger.info("Fetching scorecard: %s", url)

    try:
        response = get_page(url)
        data = response.json()
    except Exception as exc:
        logger.warning(
            "Failed to fetch/parse scorecard for match %d: %s", espn_match_id, exc
        )
        return None

    try:
        return _parse_scorecard_json(data, espn_match_id)
    except Exception as exc:
        logger.warning(
            "Error parsing scorecard JSON for match %d: %s", espn_match_id, exc
        )
        return None


def _parse_scorecard_json(data: Dict, espn_match_id: int) -> Dict:
    """Transform raw ESPN engine JSON into normalised scorecard structure."""
    match_node: Dict = data.get("match", {})
    innings_list: list = data.get("innings", [])

    # ---- Match info -------------------------------------------------------
    match_info: Dict = {
        "espn_match_id": espn_match_id,
        "date": match_node.get("start_date_raw") or match_node.get("date", ""),
        "venue": match_node.get("ground_name", "Unknown"),
        "result": match_node.get("result", ""),
        "toss_winner": match_node.get("toss", {}).get("team", ""),
        "toss_decision": match_node.get("toss", {}).get("elected", ""),
        "team_a": "",
        "team_b": "",
        "match_type": _normalise_format(match_node.get("international_class_card", "")),
        "tournament": match_node.get("series_name", "Unknown"),
    }

    # Teams from match node
    teams: list = data.get("team", [])
    if len(teams) >= 1:
        match_info["team_a"] = teams[0].get("team_name", "") or teams[0].get("team_abbreviation", "")
    if len(teams) >= 2:
        match_info["team_b"] = teams[1].get("team_name", "") or teams[1].get("team_abbreviation", "")

    # ---- Innings summary + batting/bowling --------------------------------
    innings_summaries: List[Dict] = []
    batting_by_innings: Dict[int, List[Dict]] = {}
    bowling_by_innings: Dict[int, List[Dict]] = {}

    for idx, inn in enumerate(innings_list):
        inn_num = idx + 1

        batting_team = inn.get("batting_team_name", "") or inn.get("batting_team", "")
        bowling_team = inn.get("bowling_team_name", "") or inn.get("bowling_team", "")
        total_runs = _safe_int(inn.get("runs", 0))
        total_wickets = _safe_int(inn.get("wickets", 0))
        total_overs_str = inn.get("overs", "0")

        innings_summaries.append(
            {
                "innings_number": inn_num,
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "total_runs": total_runs,
                "total_wickets": total_wickets,
                "total_overs": _parse_overs(str(total_overs_str)),
                "extras": _safe_int(inn.get("extras", {}).get("total", 0)),
            }
        )

        # Batting
        batting_rows: List[Dict] = []
        for batter in inn.get("bat", []):
            batting_rows.append(
                {
                    "name": batter.get("name", ""),
                    "runs": _safe_int(batter.get("runs", 0)),
                    "balls": _safe_int(batter.get("balls_faced", 0)),
                    "fours": _safe_int(batter.get("fours", 0)),
                    "sixes": _safe_int(batter.get("sixes", 0)),
                    "strike_rate": _safe_float(batter.get("strike_rate", 0.0)),
                    "dismissal": batter.get("dismissal", "") or batter.get("how_out", ""),
                    "not_out": batter.get("how_out", "") in ("not out", ""),
                    "batting_position": _safe_int(batter.get("bat_order", 0)),
                }
            )
        batting_by_innings[inn_num] = batting_rows

        # Bowling
        bowling_rows: List[Dict] = []
        for bowler in inn.get("bowl", []):
            overs_raw = str(bowler.get("overs", "0"))
            bowling_rows.append(
                {
                    "name": bowler.get("name", ""),
                    "overs": _parse_overs(overs_raw),
                    "maidens": _safe_int(bowler.get("maidens", 0)),
                    "runs": _safe_int(bowler.get("runs", 0)),
                    "wickets": _safe_int(bowler.get("wickets", 0)),
                    "economy": _safe_float(bowler.get("economy_rate", 0.0)),
                }
            )
        bowling_by_innings[inn_num] = bowling_rows

    return {
        "match_info": match_info,
        "innings": innings_summaries,
        "batting": batting_by_innings,
        "bowling": bowling_by_innings,
    }


# ---------------------------------------------------------------------------
# Ball-by-ball scraper
# ---------------------------------------------------------------------------

def scrape_ball_by_ball(espn_match_id: int) -> Optional[List[Dict]]:
    """
    Fetch delivery-level commentary for *espn_match_id*.

    Ball-by-ball data is not available for all matches (older fixtures or
    certain formats may lack it).  This function returns ``None`` rather than
    raising when the data is unavailable.

    Parameters
    ----------
    espn_match_id:
        ESPN numeric match identifier.

    Returns
    -------
    list of dict or None
        Each dict represents one delivery with keys:
        ``innings_number``, ``over``, ``ball``, ``batsman_name``,
        ``bowler_name``, ``runs_scored``, ``extras``, ``extra_type``,
        ``wicket_type``, ``fielder_name``.
        Returns ``None`` on failure or if no commentary exists.
    """
    url = _COMMS_URL.format(match_id=espn_match_id)
    logger.info("Fetching ball-by-ball: %s", url)

    try:
        response = get_page(url)
        data = response.json()
    except Exception as exc:
        logger.warning(
            "Failed to fetch ball-by-ball for match %d: %s", espn_match_id, exc
        )
        return None

    try:
        return _parse_ball_by_ball_json(data)
    except Exception as exc:
        logger.warning(
            "Error parsing ball-by-ball JSON for match %d: %s", espn_match_id, exc
        )
        return None


def _parse_ball_by_ball_json(data: Dict) -> Optional[List[Dict]]:
    """
    Extract delivery records from the ESPN ``comms`` response.

    The response contains an ``innings_list`` array; each element has a
    ``ball_comms`` array of ball objects.
    """
    innings_list = data.get("innings_list") or data.get("innings", [])
    if not innings_list:
        logger.debug("No innings_list found in comms response")
        return None

    deliveries: List[Dict] = []

    for inn_idx, innings in enumerate(innings_list):
        inn_num = inn_idx + 1
        ball_comms = innings.get("ball_comms") or innings.get("comms", [])

        for ball in ball_comms:
            if not isinstance(ball, dict):
                continue
            delivery = _parse_single_delivery(ball, inn_num)
            if delivery:
                deliveries.append(delivery)

    if not deliveries:
        logger.debug("No deliveries extracted from comms response")
        return None

    logger.info("Extracted %d deliveries from ball-by-ball feed", len(deliveries))
    return deliveries


def _parse_single_delivery(ball: Dict, inn_num: int) -> Optional[Dict]:
    """Normalise one ball_comms entry into our delivery schema."""
    try:
        # Over string is typically "12.3" (over 12, ball 3)
        over_str = str(ball.get("over_number", ball.get("over", "0.1")))
        over_num, ball_num = _split_over_ball(over_str)

        # Runs
        runs_off_bat = _safe_int(
            ball.get("runs_off_bat")
            or ball.get("batsman_runs")
            or ball.get("runs", 0)
        )
        extras = _safe_int(ball.get("extras", 0))

        # Extra type
        extra_type: Optional[str] = None
        for et in ("wide", "no_ball", "bye", "leg_bye", "penalty"):
            if _safe_int(ball.get(et, 0)) > 0:
                extra_type = et.replace("_", " ")
                break
        if not extra_type and extras > 0:
            extra_type = ball.get("extra_type") or ball.get("extras_type")

        # Wicket
        wicket_type: Optional[str] = ball.get("wicket_type") or ball.get("out_type") or None
        fielder_name: Optional[str] = ball.get("fielder_name") or ball.get("fielder") or None

        # Player names (comms uses various key names)
        batsman_name = (
            ball.get("batsman_name")
            or ball.get("batsman")
            or ball.get("striker")
            or ""
        )
        bowler_name = (
            ball.get("bowler_name")
            or ball.get("bowler")
            or ""
        )

        return {
            "innings_number": inn_num,
            "over": over_num,
            "ball": ball_num,
            "batsman_name": batsman_name.strip(),
            "bowler_name": bowler_name.strip(),
            "runs_scored": runs_off_bat,
            "extras": extras,
            "extra_type": extra_type,
            "wicket_type": wicket_type,
            "fielder_name": fielder_name,
        }
    except Exception as exc:
        logger.debug("Skipping malformed delivery dict: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Database upsert
# ---------------------------------------------------------------------------

def parse_scorecard_to_db(espn_match_id: int, session: Session) -> bool:
    """
    Fetch and persist full scorecard data for *espn_match_id*.

    Orchestrates :func:`scrape_scorecard` and :func:`scrape_ball_by_ball`,
    then upserts the results into Match, Innings, PlayerStat, and Delivery
    rows.  Player names are fuzzy-matched against existing Player records;
    unmatched names create minimal new records.

    Parameters
    ----------
    espn_match_id:
        ESPN numeric match identifier.
    session:
        Active SQLAlchemy session.  The caller is responsible for committing
        or rolling back.

    Returns
    -------
    bool
        ``True`` if at least the Match and Innings rows were persisted
        successfully; ``False`` on any critical failure.
    """
    logger.info("parse_scorecard_to_db: match_id=%d", espn_match_id)

    scorecard = scrape_scorecard(espn_match_id)
    if not scorecard:
        logger.warning("No scorecard data for match %d — skipping DB upsert", espn_match_id)
        return False

    ball_by_ball = scrape_ball_by_ball(espn_match_id)  # may be None

    try:
        match_key = f"espn_{espn_match_id}"
        mi = scorecard["match_info"]

        # ---- Upsert Match -------------------------------------------------
        match_obj: Optional[Match] = (
            session.query(Match).filter_by(match_key=match_key).first()
        )
        if match_obj is None:
            match_obj = Match(match_key=match_key)
            session.add(match_obj)

        match_obj.team_a = mi.get("team_a", "Unknown")
        match_obj.team_b = mi.get("team_b", "Unknown")
        match_obj.venue = mi.get("venue", "Unknown")
        match_obj.match_date = mi.get("date", "")
        match_obj.match_type = mi.get("match_type", "")
        match_obj.tournament = mi.get("tournament", "")
        match_obj.toss_winner = mi.get("toss_winner", "")
        match_obj.toss_decision = mi.get("toss_decision", "")
        match_obj.winner = _extract_winner(mi.get("result", ""), mi.get("team_a", ""), mi.get("team_b", ""))
        match_obj.result_margin = mi.get("result", "")
        match_obj.source = "espn_scrape"
        match_obj.innings_complete = True
        match_obj.updated_at = _now()

        session.flush()  # get match_obj.id

        # ---- Upsert Innings -----------------------------------------------
        innings_id_map: Dict[int, int] = {}  # innings_number → Innings.id

        for inn_summary in scorecard["innings"]:
            inn_num = inn_summary["innings_number"]
            innings_obj: Optional[Innings] = (
                session.query(Innings)
                .filter_by(match_id=match_obj.id, innings_number=inn_num)
                .first()
            )
            if innings_obj is None:
                innings_obj = Innings(
                    match_id=match_obj.id, innings_number=inn_num
                )
                session.add(innings_obj)

            innings_obj.batting_team = inn_summary.get("batting_team", "")
            innings_obj.bowling_team = inn_summary.get("bowling_team", "")
            innings_obj.total_runs = inn_summary.get("total_runs", 0)
            innings_obj.total_wickets = inn_summary.get("total_wickets", 0)
            innings_obj.total_overs = inn_summary.get("total_overs", 0.0)
            innings_obj.extras = inn_summary.get("extras", 0)
            session.flush()
            innings_id_map[inn_num] = innings_obj.id

        # Build player name cache to avoid N+1 queries
        all_players: List[Player] = session.query(Player).all()
        player_name_cache: Dict[str, Player] = {p.name: p for p in all_players}

        # ---- Upsert PlayerStat (batting) -----------------------------------
        for inn_num, batting_rows in scorecard["batting"].items():
            innings_db_id = innings_id_map.get(inn_num)
            batting_team = next(
                (s["batting_team"] for s in scorecard["innings"] if s["innings_number"] == inn_num),
                "",
            )
            for pos, bat in enumerate(batting_rows, start=1):
                player = _resolve_player(
                    bat["name"], batting_team, player_name_cache, session
                )
                if player is None:
                    continue

                stat: Optional[PlayerStat] = (
                    session.query(PlayerStat)
                    .filter_by(
                        player_id=player.id,
                        match_id=match_obj.id,
                        innings_id=innings_db_id,
                    )
                    .first()
                )
                if stat is None:
                    stat = PlayerStat(
                        player_id=player.id,
                        match_id=match_obj.id,
                        innings_id=innings_db_id,
                    )
                    session.add(stat)

                stat.team = batting_team
                stat.runs = bat.get("runs", 0)
                stat.balls_faced = bat.get("balls", 0)
                stat.fours = bat.get("fours", 0)
                stat.sixes = bat.get("sixes", 0)
                stat.strike_rate = bat.get("strike_rate", 0.0)
                stat.not_out = bat.get("not_out", False)
                stat.batting_position = pos

        # ---- Upsert PlayerStat (bowling) -----------------------------------
        for inn_num, bowling_rows in scorecard["bowling"].items():
            innings_db_id = innings_id_map.get(inn_num)
            bowling_team = next(
                (s["bowling_team"] for s in scorecard["innings"] if s["innings_number"] == inn_num),
                "",
            )
            for slot, bowl in enumerate(bowling_rows, start=1):
                player = _resolve_player(
                    bowl["name"], bowling_team, player_name_cache, session
                )
                if player is None:
                    continue

                stat = (
                    session.query(PlayerStat)
                    .filter_by(
                        player_id=player.id,
                        match_id=match_obj.id,
                        innings_id=innings_db_id,
                    )
                    .first()
                )
                if stat is None:
                    stat = PlayerStat(
                        player_id=player.id,
                        match_id=match_obj.id,
                        innings_id=innings_db_id,
                    )
                    session.add(stat)

                stat.team = bowling_team
                stat.wickets = bowl.get("wickets", 0)
                stat.overs_bowled = bowl.get("overs", 0.0)
                stat.runs_conceded = bowl.get("runs", 0)
                stat.economy_rate = bowl.get("economy", 0.0)
                stat.bowling_slot = slot

        # ---- Upsert Deliveries --------------------------------------------
        if ball_by_ball:
            _upsert_deliveries(
                ball_by_ball, innings_id_map, player_name_cache, session
            )

        session.flush()
        logger.info(
            "parse_scorecard_to_db success: match_key=%s innings=%d",
            match_key,
            len(innings_id_map),
        )
        return True

    except Exception as exc:
        logger.error(
            "parse_scorecard_to_db failed for match %d: %s", espn_match_id, exc,
            exc_info=True,
        )
        try:
            session.rollback()
        except Exception:
            pass
        return False


def _upsert_deliveries(
    deliveries: List[Dict],
    innings_id_map: Dict[int, int],
    player_name_cache: Dict[str, "Player"],
    session: Session,
) -> int:
    """
    Insert Delivery rows for each delivery in *deliveries*.

    Skips rows with unknown innings numbers or that already exist in the DB.
    Returns the count of new rows inserted.
    """
    from src.data.db import Delivery

    inserted = 0
    for d in deliveries:
        inn_num = d.get("innings_number", 1)
        innings_db_id = innings_id_map.get(inn_num)
        if innings_db_id is None:
            continue

        over = d.get("over", 0)
        ball = d.get("ball", 1)

        exists = (
            session.query(Delivery)
            .filter_by(innings_id=innings_db_id, over_number=over, ball_number=ball)
            .first()
        )
        if exists:
            continue

        batsman = player_name_cache.get(d.get("batsman_name", ""))
        bowler = player_name_cache.get(d.get("bowler_name", ""))

        delivery = Delivery(
            innings_id=innings_db_id,
            over_number=over,
            ball_number=ball,
            batsman_id=batsman.id if batsman else None,
            bowler_id=bowler.id if bowler else None,
            batsman_name=d.get("batsman_name", ""),
            bowler_name=d.get("bowler_name", ""),
            runs_scored=d.get("runs_scored", 0),
            extras=d.get("extras", 0),
            extra_type=d.get("extra_type"),
            wicket_type=d.get("wicket_type"),
            fielder_name=d.get("fielder_name"),
            is_boundary=d.get("runs_scored", 0) == 4,
            is_six=d.get("runs_scored", 0) == 6,
        )
        session.add(delivery)
        inserted += 1

    session.flush()
    logger.info("Inserted %d new delivery rows", inserted)
    return inserted


# ---------------------------------------------------------------------------
# Player resolution (fuzzy matching)
# ---------------------------------------------------------------------------

def _resolve_player(
    name: str,
    team: str,
    cache: Dict[str, "Player"],
    session: Session,
) -> Optional["Player"]:
    """
    Return the Player record for *name*, creating one if necessary.

    Lookup order:
    1. Exact name match in *cache*.
    2. Fuzzy match via ``difflib.get_close_matches`` at cutoff=0.8.
    3. Create a new minimal Player record.

    Parameters
    ----------
    name:
        Raw player name from scraped data.
    team:
        Team name, used as country hint when creating a new Player.
    cache:
        Dict mapping player name → Player ORM object (mutated in place when
        new players are created).
    session:
        Active SQLAlchemy session.

    Returns
    -------
    Player or None
        The resolved or newly-created Player, or ``None`` if *name* is blank.
    """
    if not name or not name.strip():
        return None

    name = name.strip()

    # 1. Exact match
    if name in cache:
        return cache[name]

    # 2. Fuzzy match
    close = difflib.get_close_matches(name, cache.keys(), n=1, cutoff=0.8)
    if close:
        matched = close[0]
        logger.debug("Fuzzy-matched player '%s' → '%s'", name, matched)
        player = cache[matched]
        # Also register under the raw name for subsequent lookups
        cache[name] = player
        return player

    # 3. Create new minimal player
    logger.info("Creating new Player record for '%s' (team: %s)", name, team)
    player = Player(
        name=name,
        country=team,
        created_at=_now(),
    )
    session.add(player)
    session.flush()
    cache[name] = player
    return player


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _normalise_format(raw: str) -> str:
    """Map ESPN class strings to our standard T20 / ODI / Test labels."""
    raw_lower = raw.lower()
    if "test" in raw_lower:
        return "Test"
    if "one-day" in raw_lower or "odi" in raw_lower or "list a" in raw_lower:
        return "ODI"
    if "twenty20" in raw_lower or "t20" in raw_lower or "ipl" in raw_lower:
        return "T20"
    return raw or "Unknown"


def _parse_overs(overs_str: str) -> float:
    """
    Convert an ESPN overs string (e.g. ``"19.4"``) to a float.

    ESPN represents ``20 overs`` as ``"20.0"``; ``19 overs and 4 balls`` as
    ``"19.4"``.  We preserve this as a float for storage.
    """
    try:
        return float(overs_str)
    except (ValueError, TypeError):
        return 0.0


def _split_over_ball(over_str: str):
    """
    Split an over string like ``"12.3"`` into ``(over_number, ball_number)``.

    Returns ``(0, 1)`` on parse failure.
    """
    try:
        parts = str(over_str).split(".")
        over = int(parts[0])
        ball = int(parts[1]) if len(parts) > 1 else 1
        return over, ball
    except (ValueError, IndexError):
        return 0, 1


def _safe_int(val) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _safe_float(val) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _extract_winner(result: str, team_a: str, team_b: str) -> str:
    """Attempt to identify the winning team from a free-text result string."""
    if not result:
        return ""
    result_lower = result.lower()
    if "won" in result_lower:
        if team_a.lower() in result_lower:
            return team_a
        if team_b.lower() in result_lower:
            return team_b
    if "tie" in result_lower or "draw" in result_lower or "no result" in result_lower:
        return "Draw/No Result"
    return ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
