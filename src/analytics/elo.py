"""
elo.py — Elo rating system for cricket teams.

Implements a standard Elo system where each team carries a rating per format
(T20, ODI, Test).  After every completed match, both teams' ratings are updated
using the classic Elo formula with a K-factor that scales with the importance
of the match / tournament.

Key constants
-------------
- Starting Elo: 1500 for any team with no history.
- K-factor varies by tournament tier (see `get_k_factor`).
- Expected score: 1 / (1 + 10 ** ((opponent - own) / 400))
- Rating update: new = old + K * (result - expected)
  where result = 1.0 (win), 0.5 (tie/draw), 0.0 (loss).

Usage
-----
    from src.data.db import get_session
    from src.analytics.elo import replay_all_elo, update_elo_for_match

    session = get_session()
    final_ratings = replay_all_elo(session, format="T20")
    session.close()
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from src.data.db import EloRating, Match, get_session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_ELO: float = 1500.0
ELO_DIVISOR: float = 400.0
ALL_FORMATS: List[str] = ["T20", "ODI", "Test"]


# ---------------------------------------------------------------------------
# K-factor
# ---------------------------------------------------------------------------

def get_k_factor(tournament: str) -> float:
    """Return the K-factor for a match based on the tournament name.

    Rules (evaluated in order):
    - Practice / warm-up matches        → 20
    - Tournament contains "Final" or "Semi" → 50
    - Tournament contains "ICC" or "World Cup" → 40
    - Anything else                     → 32
    """
    if tournament is None:
        return 32.0
    t_lower = tournament.lower()
    if any(kw in t_lower for kw in ("warm-up", "warmup", "practice", "unofficial")):
        return 20.0
    if any(kw in t_lower for kw in ("final", "semi")):
        return 50.0
    if any(kw in t_lower for kw in ("icc", "world cup")):
        return 40.0
    return 32.0


# ---------------------------------------------------------------------------
# Single-team Elo lookup
# ---------------------------------------------------------------------------

def get_current_elo(team_name: str, format: str, session: Session) -> float:
    """Return the most recent Elo rating for *team_name* in *format*.

    Falls back to DEFAULT_ELO (1500) when the team has no history.
    """
    row = (
        session.query(EloRating)
        .filter(EloRating.team_name == team_name, EloRating.format == format)
        .order_by(desc(EloRating.match_date), desc(EloRating.id))
        .first()
    )
    return row.rating if row else DEFAULT_ELO


# ---------------------------------------------------------------------------
# Internal Elo maths
# ---------------------------------------------------------------------------

def _expected_score(own_rating: float, opponent_rating: float) -> float:
    """Elo expected score for the team with *own_rating*."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - own_rating) / ELO_DIVISOR))


def _result_for_team(team: str, match: Match) -> Optional[float]:
    """Return Elo result (1.0 / 0.5 / 0.0) for *team* in *match*.

    Returns None for matches with no winner recorded (implies tie/no result
    treated as 0.5).
    """
    if match.winner is None or match.winner.strip() == "":
        return 0.5  # no result / tie
    if match.winner.strip().lower() in ("tie", "draw", "no result"):
        return 0.5
    return 1.0 if match.winner == team else 0.0


def _upsert_elo(
    team_name: str,
    format: str,
    rating: float,
    match: Match,
    session: Session,
) -> None:
    """Insert or update an EloRating row for *team_name* after *match*."""
    existing = (
        session.query(EloRating)
        .filter(
            EloRating.team_name == team_name,
            EloRating.format == format,
            EloRating.match_id == match.id,
        )
        .first()
    )
    if existing:
        existing.rating = rating
        existing.match_date = match.match_date
    else:
        row = EloRating(
            team_name=team_name,
            format=format,
            rating=rating,
            match_id=match.id,
            match_date=match.match_date,
        )
        session.add(row)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def update_elo_for_match(match_id: int, session: Session) -> None:
    """Compute and persist Elo updates for a single completed match.

    Fetches current ratings for both teams, applies the standard Elo formula,
    and upserts one EloRating row per team.
    """
    match: Optional[Match] = session.get(Match, match_id)
    if match is None:
        logger.warning("update_elo_for_match: match_id=%s not found", match_id)
        return
    if not match.match_type:
        logger.warning("update_elo_for_match: match_id=%s has no match_type", match_id)
        return

    fmt = match.match_type  # T20 | ODI | Test
    team_a = match.team_a
    team_b = match.team_b
    k = get_k_factor(match.tournament or "")

    ra = get_current_elo(team_a, fmt, session)
    rb = get_current_elo(team_b, fmt, session)

    ea = _expected_score(ra, rb)
    eb = _expected_score(rb, ra)

    result_a = _result_for_team(team_a, match)
    if result_a is None:
        result_a = 0.5
    result_b = 1.0 - result_a if result_a != 0.5 else 0.5

    new_ra = ra + k * (result_a - ea)
    new_rb = rb + k * (result_b - eb)

    _upsert_elo(team_a, fmt, new_ra, match, session)
    _upsert_elo(team_b, fmt, new_rb, match, session)

    try:
        session.commit()
        logger.debug(
            "Elo updated for match %s (%s vs %s, fmt=%s): %.1f→%.1f / %.1f→%.1f",
            match_id, team_a, team_b, fmt, ra, new_ra, rb, new_rb,
        )
    except Exception:
        session.rollback()
        raise


def replay_all_elo(
    session: Session,
    format: Optional[str] = None,
) -> Dict[str, float]:
    """Replay Elo ratings chronologically for all completed matches.

    Deletes existing EloRating rows for the affected format(s), then processes
    every match in ascending date order, creating one EloRating row per team
    per match.

    Parameters
    ----------
    session:
        Active SQLAlchemy session.
    format:
        If given, only matches of that format are processed.  If None, all
        three formats (T20, ODI, Test) are processed independently.

    Returns
    -------
    Dict[str, float]
        Final ratings keyed by team name.  When *format* is None the dict
        contains ratings from the last-processed format iteration; callers
        that need per-format results should call this function once per format.
    """
    formats_to_run = [format] if format else ALL_FORMATS
    final_ratings: Dict[str, float] = {}

    for fmt in formats_to_run:
        logger.info("replay_all_elo: processing format=%s", fmt)

        # --- Delete existing rows for this format so we start fresh ----------
        session.query(EloRating).filter(EloRating.format == fmt).delete()
        session.commit()

        # --- Fetch all matches for this format in chronological order --------
        matches: List[Match] = (
            session.query(Match)
            .filter(Match.match_type == fmt)
            .filter(Match.winner.isnot(None))
            .order_by(Match.match_date.asc(), Match.id.asc())
            .all()
        )

        logger.info("replay_all_elo: %d matches to process for %s", len(matches), fmt)

        # Running ratings dict (in-memory cache so we don't query the DB on
        # every match during the replay — only committed at the end)
        ratings: Dict[str, float] = {}

        for match in matches:
            team_a = match.team_a
            team_b = match.team_b
            k = get_k_factor(match.tournament or "")

            ra = ratings.get(team_a, DEFAULT_ELO)
            rb = ratings.get(team_b, DEFAULT_ELO)

            ea = _expected_score(ra, rb)
            eb = _expected_score(rb, ra)

            result_a = _result_for_team(team_a, match)
            if result_a is None:
                result_a = 0.5
            result_b = 1.0 - result_a if result_a != 0.5 else 0.5

            new_ra = ra + k * (result_a - ea)
            new_rb = rb + k * (result_b - eb)

            ratings[team_a] = new_ra
            ratings[team_b] = new_rb

            _upsert_elo(team_a, fmt, new_ra, match, session)
            _upsert_elo(team_b, fmt, new_rb, match, session)

        try:
            session.commit()
        except Exception:
            session.rollback()
            raise

        final_ratings.update(ratings)
        logger.info(
            "replay_all_elo: completed format=%s, %d teams rated", fmt, len(ratings)
        )

    return final_ratings
