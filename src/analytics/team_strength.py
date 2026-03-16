"""
team_strength.py — Team composite strength ratings.

Aggregates individual player ratings (from PlayerFeature) into a single team
strength score that also accounts for recent form, venue conditions, and
head-to-head history.

Algorithm
---------
1. Identify the top-11 rated players who have appeared for the team in the
   chosen format (via PlayerStat.team), using their latest PlayerFeature
   snapshot rating.
2. Apply positional weights:
   - T20/ODI: batting positions 1–6 get 1.2×, bowling slots 7–11 get 1.0×.
   - Test: reversed (bowlers weighted higher).
3. Base team score = weighted mean of the 11 player ratings.
4. Form factor: wins in last 10 matches → [0.8, 1.2] via ``0.8 + win_rate*0.4``.
5. Venue factor: ``Venue.batting_factor`` if a venue name is supplied, else 1.0.
6. H2H factor: win rate in last 20 H2H matches → [0.85, 1.15] via
   ``0.85 + h2h_win_rate * 0.30``.
7. final_strength = base × form_factor × venue_factor × h2h_factor.

Public API
----------
- get_recent_win_pct(team_name, format, n, session) -> float
- get_h2h_win_pct(team_a, team_b, format, n, session) -> float
- get_team_top11_ratings(team_name, format, session) -> List[float]
- compute_team_strength(team_name, format, session, opponent, venue_name) -> Dict
- run_all_team_strengths(session, snapshot_date) -> int
"""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import Dict, List, Optional, Tuple

from sqlalchemy import desc, func, or_
from sqlalchemy.orm import Session

from src.data.db import (
    Match,
    Player,
    PlayerFeature,
    PlayerStat,
    TeamFeature,
    Venue,
    get_session,
)

logger = logging.getLogger(__name__)

_DEFAULT_POSITIONAL_WEIGHT_BATTING = 1.2
_DEFAULT_POSITIONAL_WEIGHT_BOWLING = 1.0


# ---------------------------------------------------------------------------
# Recent win percentage
# ---------------------------------------------------------------------------

def get_recent_win_pct(
    team_name: str,
    format: str,
    n: int,
    session: Session,
) -> float:
    """Return the win percentage of *team_name* in its last *n* matches.

    Only matches where the team actually played (as team_a or team_b) are
    considered.  Returns 0.5 when there is no history.
    """
    query = (
        session.query(Match)
        .filter(
            Match.match_type == format,
            Match.winner.isnot(None),
            or_(Match.team_a == team_name, Match.team_b == team_name),
        )
        .order_by(desc(Match.match_date), desc(Match.id))
        .limit(n)
    )
    matches: List[Match] = query.all()
    if not matches:
        return 0.5
    wins = sum(1 for m in matches if m.winner == team_name)
    return wins / len(matches)


# ---------------------------------------------------------------------------
# Head-to-head win percentage
# ---------------------------------------------------------------------------

def get_h2h_win_pct(
    team_a: str,
    team_b: str,
    format: str,
    n: int,
    session: Session,
) -> float:
    """Return *team_a*'s win percentage in the last *n* H2H matches vs *team_b*.

    Considers both orderings (team_a vs team_b and team_b vs team_a).
    Returns 0.5 when there is no H2H history.
    """
    query = (
        session.query(Match)
        .filter(
            Match.match_type == format,
            Match.winner.isnot(None),
            or_(
                (Match.team_a == team_a) & (Match.team_b == team_b),
                (Match.team_a == team_b) & (Match.team_b == team_a),
            ),
        )
        .order_by(desc(Match.match_date), desc(Match.id))
        .limit(n)
    )
    matches: List[Match] = query.all()
    if not matches:
        return 0.5
    wins_a = sum(1 for m in matches if m.winner == team_a)
    return wins_a / len(matches)


# ---------------------------------------------------------------------------
# Top-11 player ratings for a team
# ---------------------------------------------------------------------------

def get_team_top11_ratings(
    team_name: str,
    format: str,
    session: Session,
) -> List[float]:
    """Return a list of up to 11 player ratings for *team_name* in *format*.

    Strategy:
    1. Find all distinct player_ids who have a PlayerStat row with
       ``team == team_name`` in the given format.
    2. For each such player, find their latest PlayerFeature.rating.
    3. Sort descending by rating, return top 11.

    Falls back to an empty list when no data is available.
    """
    # Distinct player ids for this team+format
    player_id_rows = (
        session.query(PlayerStat.player_id)
        .join(Match, PlayerStat.match_id == Match.id)
        .filter(
            PlayerStat.team == team_name,
            Match.match_type == format,
        )
        .distinct()
        .all()
    )
    player_ids = [r[0] for r in player_id_rows]
    if not player_ids:
        return []

    # Latest snapshot date per player
    ratings: List[float] = []
    for pid in player_ids:
        latest = (
            session.query(PlayerFeature.rating)
            .filter(
                PlayerFeature.player_id == pid,
                PlayerFeature.format == format,
                PlayerFeature.rating.isnot(None),
            )
            .order_by(desc(PlayerFeature.snapshot_date))
            .first()
        )
        if latest and latest[0] is not None:
            ratings.append(latest[0])

    ratings.sort(reverse=True)
    return ratings[:11]


# ---------------------------------------------------------------------------
# Core team strength computation
# ---------------------------------------------------------------------------

def compute_team_strength(
    team_name: str,
    format: str,
    session: Session,
    opponent: Optional[str] = None,
    venue_name: Optional[str] = None,
) -> Dict:
    """Compute a composite team strength score.

    Parameters
    ----------
    team_name:
        Team whose strength to compute.
    format:
        Match format: T20 | ODI | Test.
    session:
        Active SQLAlchemy session.
    opponent:
        If provided, H2H factor is computed against this opponent.
    venue_name:
        If provided, venue batting factor is looked up from the Venue table.

    Returns
    -------
    Dict with keys: base_strength, form_factor, venue_factor, h2h_factor,
                    final_strength, n_players.
    """
    player_ratings = get_team_top11_ratings(team_name, format, session)
    n_players = len(player_ratings)

    if n_players == 0:
        logger.debug("compute_team_strength: no players for %s/%s", team_name, format)
        return {
            "base_strength": 0.0,
            "form_factor": 1.0,
            "venue_factor": 1.0,
            "h2h_factor": 1.0,
            "final_strength": 0.0,
            "n_players": 0,
        }

    # -- Positional weights --------------------------------------------------
    # Assign 1-based positions to players sorted desc by rating
    # (best players bat at top — 1–6 are batters; 7–11 are bowlers)
    weights: List[float] = []
    for i, _ in enumerate(player_ratings, start=1):
        if format == "Test":
            # Test: bowlers (7–11) weighted higher
            w = _DEFAULT_POSITIONAL_WEIGHT_BATTING if i > 6 else _DEFAULT_POSITIONAL_WEIGHT_BOWLING
        else:
            # T20/ODI: batters (1–6) weighted higher
            w = _DEFAULT_POSITIONAL_WEIGHT_BATTING if i <= 6 else _DEFAULT_POSITIONAL_WEIGHT_BOWLING
        weights.append(w)

    total_weight = sum(weights)
    base_strength = (
        sum(r * w for r, w in zip(player_ratings, weights)) / total_weight
        if total_weight > 0 else 0.0
    )

    # -- Form factor ---------------------------------------------------------
    win_rate = get_recent_win_pct(team_name, format, n=10, session=session)
    form_factor = 0.8 + win_rate * 0.4   # [0.8, 1.2]

    # -- Venue factor --------------------------------------------------------
    venue_factor = 1.0
    if venue_name:
        venue_row: Optional[Venue] = (
            session.query(Venue)
            .filter(Venue.name == venue_name)
            .first()
        )
        if venue_row and venue_row.batting_factor is not None:
            venue_factor = venue_row.batting_factor

    # -- H2H factor ----------------------------------------------------------
    h2h_factor = 1.0
    if opponent:
        h2h_win_rate = get_h2h_win_pct(team_name, opponent, format, n=20, session=session)
        h2h_factor = 0.85 + h2h_win_rate * 0.30   # [0.85, 1.15]

    # -- Final ---------------------------------------------------------------
    final_strength = base_strength * form_factor * venue_factor * h2h_factor

    return {
        "base_strength": base_strength,
        "form_factor": form_factor,
        "venue_factor": venue_factor,
        "h2h_factor": h2h_factor,
        "final_strength": final_strength,
        "n_players": n_players,
    }


# ---------------------------------------------------------------------------
# Upsert helper
# ---------------------------------------------------------------------------

def _upsert_team_feature(
    team_name: str,
    format: str,
    snapshot_date: str,
    result: Dict,
    session: Session,
) -> None:
    feature_blob = json.dumps(result)
    existing = (
        session.query(TeamFeature)
        .filter(
            TeamFeature.team_name == team_name,
            TeamFeature.format == format,
            TeamFeature.snapshot_date == snapshot_date,
        )
        .first()
    )
    if existing:
        existing.rating = result.get("final_strength")
        existing.recent_win_pct = result.get("form_factor")
        existing.expected_xi_strength = result.get("base_strength")
        existing.feature_json = feature_blob
    else:
        row = TeamFeature(
            team_name=team_name,
            snapshot_date=snapshot_date,
            format=format,
            rating=result.get("final_strength"),
            recent_win_pct=result.get("form_factor"),
            expected_xi_strength=result.get("base_strength"),
            feature_json=feature_blob,
        )
        session.add(row)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_all_team_strengths(
    session: Session,
    snapshot_date: Optional[str] = None,
) -> int:
    """Compute and persist TeamFeature rows for every known team and format.

    Teams are discovered by scanning the distinct ``team_a`` and ``team_b``
    values in the matches table.

    Parameters
    ----------
    session:
        Active SQLAlchemy session.
    snapshot_date:
        Snapshot date string (``YYYY-MM-DD``).  Defaults to today.

    Returns
    -------
    int
        Number of (team, format) combinations processed.
    """
    if snapshot_date is None:
        snapshot_date = date.today().strftime("%Y-%m-%d")

    # Gather all known teams from match records
    team_a_rows = session.query(Match.team_a).distinct().all()
    team_b_rows = session.query(Match.team_b).distinct().all()
    teams = set(r[0] for r in team_a_rows if r[0]) | set(r[0] for r in team_b_rows if r[0])

    formats = ["T20", "ODI", "Test"]
    processed = 0

    for team_name in sorted(teams):
        for fmt in formats:
            try:
                result = compute_team_strength(team_name, fmt, session)
                _upsert_team_feature(team_name, fmt, snapshot_date, result, session)
                processed += 1
            except Exception as exc:
                logger.warning(
                    "run_all_team_strengths: error for team=%s format=%s: %s",
                    team_name, fmt, exc,
                )

    try:
        session.commit()
        logger.info(
            "run_all_team_strengths: committed %d team+format entries for %s",
            processed, snapshot_date,
        )
    except Exception:
        session.rollback()
        raise

    return processed
