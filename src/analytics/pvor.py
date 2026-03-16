"""
pvor.py — Player Value Over Replacement (PVOR) calculation.

PVOR measures how much more (or less) value a player provides compared with a
"replacement-level" player — i.e. the 25th-percentile performer at the same
batting position or bowling slot across all matches in the given format.

Formulae
--------
Batting PVOR  = (player_runs  − replacement_avg_runs) / max(replacement_std, 1)
Bowling PVOR  = (replacement_wpm − player_wpm)        / max(replacement_std, 0.1)
                  ↑ inverted: scoring fewer wickets than replacement is negative
Fielding PVOR = (catches + stumpings + run_outs) × 3
                  (run_outs are not stored on PlayerStat; treated as 0)

Role weights (total = 1.0)::

    batsman      → batting=0.8, bowling=0.1, fielding=0.1
    bowler       → batting=0.1, bowling=0.8, fielding=0.1
    all-rounder  → batting=0.5, bowling=0.5, fielding=0.1  (re-normalised)
    wicket-keeper→ batting=0.7, bowling=0.0, fielding=0.2  (re-normalised with 0.1 extra)

Total PVOR = batting_pvor × batting_w + bowling_pvor × bowling_w
           + fielding_pvor × fielding_w

Replacement level caching
--------------------------
``compute_replacement_levels`` results are cached in a module-level dict keyed
by format so that a full replay only hits the DB once per format.

Public API
----------
- compute_replacement_levels(format, session) -> Dict
- compute_match_pvor(match_id, session) -> int
- compute_player_agg_pvor(player_id, format, session) -> None
- run_all_pvor(session) -> int
"""
from __future__ import annotations

import logging
import statistics
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy import desc, func, or_
from sqlalchemy.orm import Session

from src.data.db import (
    Match,
    Player,
    PlayerStat,
    PVORMatch,
    PVORPlayerAgg,
    get_session,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache for replacement levels
# ---------------------------------------------------------------------------
_replacement_cache: Dict[str, Dict] = {}


# ---------------------------------------------------------------------------
# Role weight lookup
# ---------------------------------------------------------------------------

def _role_weights(role: str) -> Tuple[float, float, float]:
    """Return (batting_w, bowling_w, fielding_w) for *role*."""
    r = (role or "batsman").lower()
    if "all" in r:
        return (0.5, 0.5, 0.1)   # will be used as proportional weights
    if "keep" in r or "wicket" in r:
        return (0.7, 0.0, 0.2)
    if "bowl" in r:
        return (0.1, 0.8, 0.1)
    # Default: batsman
    return (0.8, 0.1, 0.1)


# ---------------------------------------------------------------------------
# Replacement level computation
# ---------------------------------------------------------------------------

def compute_replacement_levels(format: str, session: Session) -> Dict:
    """Compute 25th-percentile replacement statistics per batting position and bowling slot.

    Results are cached in ``_replacement_cache`` to avoid repeated DB queries.

    Returns
    -------
    Dict of the form::

        {
            "batting": {
                1: {"mean": float, "std": float},
                2: {"mean": float, "std": float},
                ...
            },
            "bowling": {
                1: {"mean": float, "std": float},
                ...
            }
        }
    """
    if format in _replacement_cache:
        return _replacement_cache[format]

    # ---- Batting replacement levels ----------------------------------------
    batting_data: Dict[int, List[float]] = {}
    bat_rows = (
        session.query(PlayerStat.batting_position, PlayerStat.runs)
        .join(Match, PlayerStat.match_id == Match.id)
        .filter(
            Match.match_type == format,
            PlayerStat.batting_position.isnot(None),
            PlayerStat.balls_faced > 0,
        )
        .all()
    )
    for pos, runs in bat_rows:
        if pos is None:
            continue
        batting_data.setdefault(pos, []).append(float(runs or 0))

    batting_levels: Dict[int, Dict] = {}
    for pos, run_list in batting_data.items():
        run_list_sorted = sorted(run_list)
        n = len(run_list_sorted)
        p25_idx = max(0, int(n * 0.25) - 1)
        replacement_mean = run_list_sorted[p25_idx]
        std = statistics.stdev(run_list) if len(run_list) >= 2 else 1.0
        batting_levels[pos] = {"mean": replacement_mean, "std": std}

    # ---- Bowling replacement levels ----------------------------------------
    bowling_data: Dict[int, List[float]] = {}
    bowl_rows = (
        session.query(PlayerStat.bowling_slot, PlayerStat.wickets, PlayerStat.overs_bowled)
        .join(Match, PlayerStat.match_id == Match.id)
        .filter(
            Match.match_type == format,
            PlayerStat.bowling_slot.isnot(None),
            PlayerStat.overs_bowled > 0,
        )
        .all()
    )
    for slot, wickets, overs in bowl_rows:
        if slot is None:
            continue
        bowling_data.setdefault(slot, []).append(float(wickets or 0))

    bowling_levels: Dict[int, Dict] = {}
    for slot, wkt_list in bowling_data.items():
        wkt_sorted = sorted(wkt_list)
        n = len(wkt_sorted)
        p25_idx = max(0, int(n * 0.25) - 1)
        replacement_mean = wkt_sorted[p25_idx]
        std = statistics.stdev(wkt_list) if len(wkt_list) >= 2 else 0.1
        bowling_levels[slot] = {"mean": replacement_mean, "std": std}

    result = {"batting": batting_levels, "bowling": bowling_levels}
    _replacement_cache[format] = result
    return result


# ---------------------------------------------------------------------------
# Per-match PVOR
# ---------------------------------------------------------------------------

def _upsert_pvor_match(
    player_id: int,
    match_id: int,
    format: str,
    batting_pvor: float,
    bowling_pvor: float,
    fielding_pvor: float,
    total_pvor: float,
    session: Session,
) -> None:
    existing = (
        session.query(PVORMatch)
        .filter(PVORMatch.player_id == player_id, PVORMatch.match_id == match_id)
        .first()
    )
    if existing:
        existing.format = format
        existing.batting_pvor = batting_pvor
        existing.bowling_pvor = bowling_pvor
        existing.fielding_pvor = fielding_pvor
        existing.total_pvor = total_pvor
    else:
        row = PVORMatch(
            player_id=player_id,
            match_id=match_id,
            format=format,
            batting_pvor=batting_pvor,
            bowling_pvor=bowling_pvor,
            fielding_pvor=fielding_pvor,
            total_pvor=total_pvor,
        )
        session.add(row)


def compute_match_pvor(match_id: int, session: Session) -> int:
    """Compute PVOR for every player who featured in *match_id*.

    Upserts PVORMatch rows and returns the count of rows written.
    """
    match: Optional[Match] = session.get(Match, match_id)
    if match is None:
        logger.warning("compute_match_pvor: match_id=%s not found", match_id)
        return 0
    if not match.match_type:
        return 0

    fmt = match.match_type
    rep = compute_replacement_levels(fmt, session)
    batting_rep = rep.get("batting", {})
    bowling_rep = rep.get("bowling", {})

    stats: List[PlayerStat] = (
        session.query(PlayerStat)
        .filter(PlayerStat.match_id == match_id)
        .all()
    )
    if not stats:
        return 0

    count = 0
    for stat in stats:
        player: Optional[Player] = session.get(Player, stat.player_id)
        role = (player.role if player else None) or "batsman"
        bat_w, bowl_w, field_w = _role_weights(role)

        # ---- Batting PVOR --------------------------------------------------
        batting_pvor = 0.0
        if (stat.batting_position is not None) and ((stat.balls_faced or 0) > 0):
            pos = stat.batting_position
            rep_data = batting_rep.get(pos, {"mean": 0.0, "std": 1.0})
            rep_mean = rep_data["mean"]
            rep_std = max(rep_data["std"], 1.0)
            batting_pvor = (float(stat.runs or 0) - rep_mean) / rep_std

        # ---- Bowling PVOR --------------------------------------------------
        bowling_pvor = 0.0
        if (stat.bowling_slot is not None) and ((stat.overs_bowled or 0) > 0):
            slot = stat.bowling_slot
            rep_data = bowling_rep.get(slot, {"mean": 0.0, "std": 0.1})
            rep_mean = rep_data["mean"]
            rep_std = max(rep_data["std"], 0.1)
            player_wpm = float(stat.wickets or 0)
            # Inverted: replacement_wpm - player_wpm => fewer wickets = negative
            bowling_pvor = (rep_mean - player_wpm) / rep_std
            # Flip sign: positive PVOR means player took MORE wickets than replacement
            bowling_pvor = -bowling_pvor

        # ---- Fielding PVOR -------------------------------------------------
        fielding_pvor = float((stat.catches or 0) + (stat.stumpings or 0)) * 3.0

        # ---- Total PVOR ----------------------------------------------------
        total_pvor = (
            batting_pvor * bat_w
            + bowling_pvor * bowl_w
            + fielding_pvor * field_w
        )

        _upsert_pvor_match(
            stat.player_id,
            match_id,
            fmt,
            batting_pvor,
            bowling_pvor,
            fielding_pvor,
            total_pvor,
            session,
        )
        count += 1

    return count


# ---------------------------------------------------------------------------
# Per-player aggregate PVOR
# ---------------------------------------------------------------------------

def _upsert_pvor_agg(
    player_id: int,
    format: str,
    period: str,
    snapshot_date: str,
    batting_avg: float,
    bowling_avg: float,
    total_avg: float,
    n_matches: int,
    session: Session,
) -> None:
    existing = (
        session.query(PVORPlayerAgg)
        .filter(
            PVORPlayerAgg.player_id == player_id,
            PVORPlayerAgg.format == format,
            PVORPlayerAgg.period == period,
            PVORPlayerAgg.snapshot_date == snapshot_date,
        )
        .first()
    )
    if existing:
        existing.batting_pvor_avg = batting_avg
        existing.bowling_pvor_avg = bowling_avg
        existing.total_pvor_avg = total_avg
        existing.n_matches = n_matches
    else:
        row = PVORPlayerAgg(
            player_id=player_id,
            format=format,
            period=period,
            snapshot_date=snapshot_date,
            batting_pvor_avg=batting_avg,
            bowling_pvor_avg=bowling_avg,
            total_pvor_avg=total_avg,
            n_matches=n_matches,
        )
        session.add(row)


def compute_player_agg_pvor(
    player_id: int,
    format: str,
    session: Session,
) -> None:
    """Compute last30d, last90d, and career PVOR aggregates for a player.

    Upserts PVORPlayerAgg rows (does NOT commit — caller is responsible).
    """
    today = date.today()
    snapshot_date = today.strftime("%Y-%m-%d")

    cutoffs: Dict[str, Optional[date]] = {
        "last30d": today - timedelta(days=30),
        "last90d": today - timedelta(days=90),
        "career": None,
    }

    for period, cutoff in cutoffs.items():
        query = (
            session.query(PVORMatch)
            .join(Match, PVORMatch.match_id == Match.id)
            .filter(
                PVORMatch.player_id == player_id,
                PVORMatch.format == format,
            )
        )
        if cutoff is not None:
            query = query.filter(Match.match_date >= cutoff.strftime("%Y-%m-%d"))

        rows: List[PVORMatch] = query.all()
        n = len(rows)
        if n == 0:
            _upsert_pvor_agg(
                player_id, format, period, snapshot_date,
                0.0, 0.0, 0.0, 0,
                session,
            )
            continue

        bat_avg = sum(r.batting_pvor or 0 for r in rows) / n
        bowl_avg = sum(r.bowling_pvor or 0 for r in rows) / n
        total_avg = sum(r.total_pvor or 0 for r in rows) / n

        _upsert_pvor_agg(
            player_id, format, period, snapshot_date,
            bat_avg, bowl_avg, total_avg, n,
            session,
        )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_all_pvor(session: Session) -> int:
    """Run PVOR computation for every match, then aggregate per player.

    Steps:
    1. Clear the replacement-level cache so a fresh computation is done.
    2. Compute match-level PVOR for every match with a known type.
    3. Commit match-level rows.
    4. Aggregate per player (last30d, last90d, career) for every format they
       have PVOR data in.
    5. Commit aggregates.

    Returns
    -------
    int
        Total number of matches processed.
    """
    global _replacement_cache
    _replacement_cache = {}

    matches: List[Match] = (
        session.query(Match)
        .filter(Match.match_type.isnot(None))
        .order_by(Match.match_date.asc(), Match.id.asc())
        .all()
    )

    total_matches = 0
    for match in matches:
        try:
            compute_match_pvor(match.id, session)
            total_matches += 1
        except Exception as exc:
            logger.warning("run_all_pvor: error for match_id=%s: %s", match.id, exc)

    try:
        session.commit()
        logger.info("run_all_pvor: committed match-level PVOR for %d matches", total_matches)
    except Exception:
        session.rollback()
        raise

    # ---- Per-player aggregations -------------------------------------------
    # Find all distinct (player_id, format) pairs in pvor_match
    pid_format_rows = (
        session.query(PVORMatch.player_id, PVORMatch.format)
        .distinct()
        .all()
    )

    for player_id, fmt in pid_format_rows:
        if not fmt:
            continue
        try:
            compute_player_agg_pvor(player_id, fmt, session)
        except Exception as exc:
            logger.warning(
                "run_all_pvor: agg error for player_id=%s format=%s: %s",
                player_id, fmt, exc,
            )

    try:
        session.commit()
        logger.info(
            "run_all_pvor: committed PVOR aggregates for %d player+format pairs",
            len(pid_format_rows),
        )
    except Exception:
        session.rollback()
        raise

    return total_matches
