"""
player_rating.py — Role-specific player rating engine.

All composite ratings are normalized to the range [0, 100].

Rating methodology
------------------
Ratings are built from several components whose raw values are first normalized
to [0, 1] with `normalize()`, then Bayesian-smoothed with a global population
mean to handle players with few innings/matches, and finally scaled to 0–100.

**Bayesian smoothing**::

    smoothed = (prior_weight * global_mean + n * player_value) / (prior_weight + n)

prior_weight is 15 for T20/ODI and 20 for Test.

**Recency weighting** uses exponential decay::

    weight_i = exp(-lambda * days_ago_i / 365)   (lambda = 1.0)

Weights are normalised so they sum to 1 before computing the weighted average.

Batsman composite (weights 0.40 / 0.30 / 0.20 / 0.10)::

    0.40 * avg_norm + 0.30 * sr_norm + 0.20 * form_norm + 0.10 * consistency_norm

Bowler composite (weights 0.35 / 0.30 / 0.20 / 0.15)::

    0.35 * (1 - econ_norm) + 0.30 * (1 - bowling_sr_norm) + 0.20 * (1 - avg_norm) + 0.15 * form_norm

All-rounder::

    alpha * batting_rating + (1 - alpha) * bowling_rating
    alpha = 0.6 if batting contribution > bowling, else 0.4

Wicket-keeper::

    0.7 * batting_rating + 0.3 * fielding_score
    fielding_score = (catches + stumpings) per match * 15, capped at 100

Public API
----------
- normalize(value, min_val, max_val) -> float
- bayesian_smooth(value, n, global_mean, prior_weight) -> float
- recency_weighted(values_with_dates, lambda_=1.0) -> float
- compute_batsman_rating(player_id, format, session) -> Dict
- compute_bowler_rating(player_id, format, session) -> Dict
- compute_player_rating(player_id, format, session) -> Dict
- run_all_ratings(session, snapshot_date=None) -> int
"""
from __future__ import annotations

import json
import logging
import math
import statistics
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.data.db import Match, Player, PlayerFeature, PlayerStat, get_session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format-specific normalisation ranges
# ---------------------------------------------------------------------------
_BATTING_AVG_RANGES: Dict[str, Tuple[float, float]] = {
    "T20": (10.0, 60.0),
    "ODI": (15.0, 65.0),
    "Test": (20.0, 75.0),
}
_STRIKE_RATE_RANGES: Dict[str, Optional[Tuple[float, float]]] = {
    "T20": (80.0, 200.0),
    "ODI": (60.0, 110.0),
    "Test": None,  # SR not used for Test batting rating
}
_ECON_RANGES: Dict[str, Tuple[float, float]] = {
    "T20": (4.0, 10.0),
    "ODI": (3.5, 7.0),
    "Test": (2.0, 5.0),
}
_BOWLING_AVG_RANGE: Tuple[float, float] = (10.0, 45.0)
_WPM_RANGE: Tuple[float, float] = (0.0, 4.0)

_PRIOR_WEIGHT: Dict[str, float] = {
    "T20": 15.0,
    "ODI": 15.0,
    "Test": 20.0,
}
_DEFAULT_PRIOR = 15.0

_MIN_INNINGS = 1   # minimum innings before we include in population means
_RECENT_FORM_INNINGS = 10
_RECENT_FORM_MATCHES = 10


# ---------------------------------------------------------------------------
# Core maths helpers
# ---------------------------------------------------------------------------

def normalize(value: float, min_val: float, max_val: float) -> float:
    """Clamp *value* to [min_val, max_val] and scale to [0, 1]."""
    if max_val <= min_val:
        return 0.0
    clamped = max(min_val, min(max_val, value))
    return (clamped - min_val) / (max_val - min_val)


def bayesian_smooth(
    value: float,
    n: int,
    global_mean: float,
    prior_weight: float,
) -> float:
    """Apply Bayesian shrinkage toward *global_mean*.

    smoothed = (prior_weight * global_mean + n * value) / (prior_weight + n)
    """
    return (prior_weight * global_mean + n * value) / (prior_weight + n)


def recency_weighted(
    values_with_dates: List[Tuple[float, str]],
    lambda_: float = 1.0,
) -> float:
    """Exponential-decay recency-weighted mean.

    Parameters
    ----------
    values_with_dates:
        List of (value, date_str) where date_str is ``YYYY-MM-DD``.
    lambda_:
        Decay constant (default 1.0; a match 365 days ago gets weight e^-1).

    Returns
    -------
    float
        Weighted mean, or 0.0 if the list is empty.
    """
    if not values_with_dates:
        return 0.0

    today = date.today()
    weights: List[float] = []
    vals: List[float] = []

    for value, date_str in values_with_dates:
        try:
            d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
            days_ago = max((today - d).days, 0)
        except (ValueError, TypeError):
            days_ago = 365  # treat unparseable dates as 1 year old

        w = math.exp(-lambda_ * days_ago / 365.0)
        weights.append(w)
        vals.append(value)

    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    return sum(v * w for v, w in zip(vals, weights)) / total_w


# ---------------------------------------------------------------------------
# Population means (cached per format to avoid repeated DB round-trips)
# ---------------------------------------------------------------------------

_global_batting_avg_cache: Dict[str, float] = {}
_global_sr_cache: Dict[str, float] = {}
_global_bowling_avg_cache: Dict[str, float] = {}
_global_econ_cache: Dict[str, float] = {}


def _get_global_batting_avg(format: str, session: Session) -> float:
    if format in _global_batting_avg_cache:
        return _global_batting_avg_cache[format]

    # Compute from PlayerFeature rows
    rows = (
        session.query(PlayerFeature.batting_avg)
        .filter(
            PlayerFeature.format == format,
            PlayerFeature.batting_avg.isnot(None),
            PlayerFeature.n_innings >= _MIN_INNINGS,
        )
        .all()
    )
    vals = [r[0] for r in rows if r[0] is not None and r[0] > 0]
    mean = statistics.mean(vals) if vals else 30.0
    _global_batting_avg_cache[format] = mean
    return mean


def _get_global_sr(format: str, session: Session) -> float:
    if format in _global_sr_cache:
        return _global_sr_cache[format]

    rows = (
        session.query(PlayerFeature.strike_rate)
        .filter(
            PlayerFeature.format == format,
            PlayerFeature.strike_rate.isnot(None),
        )
        .all()
    )
    vals = [r[0] for r in rows if r[0] is not None and r[0] > 0]
    mean = statistics.mean(vals) if vals else 120.0
    _global_sr_cache[format] = mean
    return mean


def _get_global_bowling_avg(format: str, session: Session) -> float:
    if format in _global_bowling_avg_cache:
        return _global_bowling_avg_cache[format]

    rows = (
        session.query(PlayerFeature.bowling_avg)
        .filter(
            PlayerFeature.format == format,
            PlayerFeature.bowling_avg.isnot(None),
        )
        .all()
    )
    vals = [r[0] for r in rows if r[0] is not None and r[0] > 0]
    mean = statistics.mean(vals) if vals else 30.0
    _global_bowling_avg_cache[format] = mean
    return mean


def _get_global_econ(format: str, session: Session) -> float:
    if format in _global_econ_cache:
        return _global_econ_cache[format]

    rows = (
        session.query(PlayerFeature.bowling_econ)
        .filter(
            PlayerFeature.format == format,
            PlayerFeature.bowling_econ.isnot(None),
        )
        .all()
    )
    vals = [r[0] for r in rows if r[0] is not None and r[0] > 0]
    mean = statistics.mean(vals) if vals else 7.0
    _global_econ_cache[format] = mean
    return mean


def _invalidate_global_caches() -> None:
    """Clear all cached global means (call before a fresh run_all_ratings)."""
    _global_batting_avg_cache.clear()
    _global_sr_cache.clear()
    _global_bowling_avg_cache.clear()
    _global_econ_cache.clear()


# ---------------------------------------------------------------------------
# Per-player stat aggregation helpers
# ---------------------------------------------------------------------------

def _get_player_stats(
    player_id: int,
    format: str,
    session: Session,
) -> List[PlayerStat]:
    """Return all PlayerStat rows for a player in the given format, ordered by match date."""
    return (
        session.query(PlayerStat)
        .join(Match, PlayerStat.match_id == Match.id)
        .filter(
            PlayerStat.player_id == player_id,
            Match.match_type == format,
        )
        .order_by(Match.match_date.asc())
        .all()
    )


def _batting_avg(stats: List[PlayerStat]) -> Tuple[float, int]:
    """Return (batting_average, n_innings).

    Innings where player did not bat (balls_faced == 0 and runs == 0) are
    excluded.  Not-out innings are excluded from the dismissal denominator.
    """
    batting_innings = [s for s in stats if s.balls_faced is not None and s.balls_faced > 0]
    n = len(batting_innings)
    if n == 0:
        return 0.0, 0
    total_runs = sum(s.runs or 0 for s in batting_innings)
    dismissals = sum(1 for s in batting_innings if not s.not_out)
    if dismissals == 0:
        return float(total_runs) / max(n, 1), n
    return float(total_runs) / dismissals, n


def _strike_rate(stats: List[PlayerStat]) -> float:
    """Overall batting strike rate across all innings."""
    batting_innings = [s for s in stats if s.balls_faced is not None and s.balls_faced > 0]
    total_runs = sum(s.runs or 0 for s in batting_innings)
    total_balls = sum(s.balls_faced or 0 for s in batting_innings)
    if total_balls == 0:
        return 0.0
    return (total_runs / total_balls) * 100.0


def _bowling_avg(stats: List[PlayerStat]) -> Tuple[float, float, int]:
    """Return (bowling_average, economy, n_matches_bowled)."""
    bowling = [s for s in stats if s.overs_bowled is not None and s.overs_bowled > 0]
    n = len(bowling)
    if n == 0:
        return 0.0, 0.0, 0
    total_wickets = sum(s.wickets or 0 for s in bowling)
    total_runs = sum(s.runs_conceded or 0 for s in bowling)
    total_balls = sum(int((s.overs_bowled or 0) * 6) for s in bowling)
    avg = total_runs / max(total_wickets, 1)
    econ = (total_runs / total_balls * 6) if total_balls else 0.0
    bowling_sr = total_balls / max(total_wickets, 1)
    return avg, econ, n


# ---------------------------------------------------------------------------
# Batsman rating
# ---------------------------------------------------------------------------

def compute_batsman_rating(
    player_id: int,
    format: str,
    session: Session,
) -> Dict:
    """Compute a batsman composite rating (0–100).

    Returns
    -------
    Dict with keys: raw_avg, strike_rate, recent_form, consistency, rating, n_innings
    """
    stats = _get_player_stats(player_id, format, session)
    avg_val, n_innings = _batting_avg(stats)
    sr_val = _strike_rate(stats)

    if n_innings == 0:
        return {
            "raw_avg": 0.0,
            "strike_rate": 0.0,
            "recent_form": 0.0,
            "consistency": 0.0,
            "rating": 0.0,
            "n_innings": 0,
        }

    # -- Normalise average ---------------------------------------------------
    avg_min, avg_max = _BATTING_AVG_RANGES.get(format, (10.0, 60.0))
    avg_norm = normalize(avg_val, avg_min, avg_max)

    # -- Normalise strike rate -----------------------------------------------
    sr_range = _STRIKE_RATE_RANGES.get(format)
    if sr_range is not None:
        sr_norm = normalize(sr_val, sr_range[0], sr_range[1])
    else:
        sr_norm = 0.5  # neutral for Test where SR is not the primary factor

    # -- Recent form (recency-weighted runs, last 10 innings) ----------------
    batting_innings = [s for s in stats if (s.balls_faced or 0) > 0]
    recent = batting_innings[-_RECENT_FORM_INNINGS:]

    form_pairs: List[Tuple[float, str]] = []
    for s in recent:
        match_date = s.match.match_date if s.match else None
        form_pairs.append((float(s.runs or 0), match_date or "2000-01-01"))

    form_raw = recency_weighted(form_pairs)
    form_norm = normalize(form_raw, 0.0, 80.0)  # 80 runs per innings ≈ excellent

    # -- Consistency: 1 / (1 + std) ------------------------------------------
    run_list = [float(s.runs or 0) for s in batting_innings]
    if len(run_list) >= 2:
        std_runs = statistics.stdev(run_list)
    else:
        std_runs = 0.0
    consistency_raw = 1.0 / (1.0 + std_runs)
    consistency_norm = normalize(consistency_raw, 0.05, 1.0)

    # -- Composite -----------------------------------------------------------
    composite = (
        0.40 * avg_norm
        + 0.30 * sr_norm
        + 0.20 * form_norm
        + 0.10 * consistency_norm
    )

    # -- Bayesian smoothing --------------------------------------------------
    prior_w = _PRIOR_WEIGHT.get(format, _DEFAULT_PRIOR)
    global_mean_avg = _get_global_batting_avg(format, session)
    global_mean_norm = normalize(global_mean_avg, avg_min, avg_max)
    smoothed = bayesian_smooth(composite, n_innings, global_mean_norm, prior_w)

    rating = max(0.0, min(100.0, smoothed * 100.0))

    return {
        "raw_avg": avg_val,
        "strike_rate": sr_val,
        "recent_form": form_raw,
        "consistency": consistency_raw,
        "rating": rating,
        "n_innings": n_innings,
    }


# ---------------------------------------------------------------------------
# Bowler rating
# ---------------------------------------------------------------------------

def compute_bowler_rating(
    player_id: int,
    format: str,
    session: Session,
) -> Dict:
    """Compute a bowler composite rating (0–100).

    Returns
    -------
    Dict with keys: bowling_avg, economy, bowling_sr, recent_form,
                    wickets_per_match, rating, n_matches
    """
    stats = _get_player_stats(player_id, format, session)
    bowling_stats = [s for s in stats if (s.overs_bowled or 0) > 0]
    n_matches = len(bowling_stats)

    if n_matches == 0:
        return {
            "bowling_avg": 0.0,
            "economy": 0.0,
            "bowling_sr": 0.0,
            "recent_form": 0.0,
            "wickets_per_match": 0.0,
            "rating": 0.0,
            "n_matches": 0,
        }

    total_wickets = sum(s.wickets or 0 for s in bowling_stats)
    total_runs_c = sum(s.runs_conceded or 0 for s in bowling_stats)
    total_balls = sum(int((s.overs_bowled or 0) * 6) for s in bowling_stats)

    bowling_avg_val = total_runs_c / max(total_wickets, 1)
    econ_val = (total_runs_c / total_balls * 6) if total_balls else 0.0
    bowling_sr_val = total_balls / max(total_wickets, 1)
    wpm = total_wickets / n_matches

    # -- Normalise economy (inverted: lower = better) -----------------------
    econ_min, econ_max = _ECON_RANGES.get(format, (4.0, 10.0))
    econ_norm = normalize(econ_val, econ_min, econ_max)        # 0 = good, 1 = bad
    econ_component = 1.0 - econ_norm                           # inverted

    # -- Normalise bowling average (inverted) --------------------------------
    avg_norm = normalize(bowling_avg_val, _BOWLING_AVG_RANGE[0], _BOWLING_AVG_RANGE[1])
    avg_component = 1.0 - avg_norm                             # lower avg = higher component

    # -- Bowling strike rate (inverted, fewer balls per wicket = better) -----
    # Use a reasonable range: 6 balls (exceptional) to 60 balls per wicket
    sr_norm = normalize(bowling_sr_val, 6.0, 60.0)
    sr_component = 1.0 - sr_norm

    # -- Wickets per match ---------------------------------------------------
    wpm_norm = normalize(wpm, _WPM_RANGE[0], _WPM_RANGE[1])

    # -- Recent form: recency-weighted wickets per match ---------------------
    recent = bowling_stats[-_RECENT_FORM_MATCHES:]
    form_pairs: List[Tuple[float, str]] = []
    for s in recent:
        match_date = s.match.match_date if s.match else None
        form_pairs.append((float(s.wickets or 0), match_date or "2000-01-01"))
    form_raw = recency_weighted(form_pairs)
    form_norm = normalize(form_raw, 0.0, 4.0)

    # -- Composite -----------------------------------------------------------
    # 0.35*(1-econ) + 0.30*(1-sr_norm) + 0.20*(1-avg_norm) + 0.15*form
    composite = (
        0.35 * econ_component
        + 0.30 * sr_component
        + 0.20 * avg_component
        + 0.15 * form_norm
    )

    # -- Bayesian smoothing --------------------------------------------------
    prior_w = _PRIOR_WEIGHT.get(format, _DEFAULT_PRIOR)
    global_mean_econ = _get_global_econ(format, session)
    global_econ_norm = normalize(global_mean_econ, econ_min, econ_max)
    global_composite = 1.0 - global_econ_norm   # global mean component (economy-only proxy)
    smoothed = bayesian_smooth(composite, n_matches, global_composite, prior_w)

    rating = max(0.0, min(100.0, smoothed * 100.0))

    return {
        "bowling_avg": bowling_avg_val,
        "economy": econ_val,
        "bowling_sr": bowling_sr_val,
        "recent_form": form_raw,
        "wickets_per_match": wpm,
        "rating": rating,
        "n_matches": n_matches,
    }


# ---------------------------------------------------------------------------
# Fielding score helper
# ---------------------------------------------------------------------------

def _fielding_score(stats: List[PlayerStat]) -> float:
    """Compute fielding contribution score (0–100) for a wicket-keeper."""
    n = len(stats)
    if n == 0:
        return 0.0
    total_ct = sum(s.catches or 0 for s in stats)
    total_st = sum(s.stumpings or 0 for s in stats)
    rate = (total_ct + total_st) / n
    return min(100.0, rate * 15.0)


# ---------------------------------------------------------------------------
# Unified player rating
# ---------------------------------------------------------------------------

def compute_player_rating(
    player_id: int,
    format: str,
    session: Session,
) -> Dict:
    """Determine a player's role and compute the appropriate composite rating.

    Returns a dict containing at minimum ``rating`` and role information, plus
    all sub-component keys from the underlying rating function(s).

    All-rounder alpha:
        0.6 if batting stats suggest the player is primarily a batter, else 0.4.

    Wicket-keeper:
        0.7 * batting_rating + 0.3 * fielding_score
    """
    player: Optional[Player] = session.get(Player, player_id)
    if player is None:
        logger.warning("compute_player_rating: player_id=%s not found", player_id)
        return {"rating": 0.0, "role": "unknown"}

    role = (player.role or "batsman").lower()

    bat_result = compute_batsman_rating(player_id, format, session)
    bowl_result = compute_bowler_rating(player_id, format, session)

    if "all" in role:
        # All-rounder: alpha depends on which contribution dominates
        bat_rating = bat_result["rating"]
        bowl_rating = bowl_result["rating"]
        alpha = 0.6 if bat_rating >= bowl_rating else 0.4
        combined_rating = alpha * bat_rating + (1.0 - alpha) * bowl_rating
        result = {**bat_result, **bowl_result}
        result["rating"] = combined_rating
        result["alpha"] = alpha
        result["role"] = role
        return result

    if "keeper" in role or "wicket" in role:
        stats = _get_player_stats(player_id, format, session)
        fs = _fielding_score(stats)
        combined_rating = 0.7 * bat_result["rating"] + 0.3 * fs
        result = {**bat_result}
        result["rating"] = combined_rating
        result["fielding_score"] = fs
        result["role"] = role
        return result

    if "bowl" in role:
        bowl_result["role"] = role
        return bowl_result

    # Default: batsman
    bat_result["role"] = role
    return bat_result


# ---------------------------------------------------------------------------
# Aggregate statistics helpers for PlayerFeature upsert
# ---------------------------------------------------------------------------

def _aggregate_stats(
    stats: List[PlayerStat],
) -> Dict:
    """Compute aggregate batting and bowling stats for PlayerFeature fields."""
    batting_innings = [s for s in stats if (s.balls_faced or 0) > 0]
    bowling_matches = [s for s in stats if (s.overs_bowled or 0) > 0]

    # Batting
    total_bat_runs = sum(s.runs or 0 for s in batting_innings)
    dismissals = sum(1 for s in batting_innings if not s.not_out)
    batting_avg = total_bat_runs / max(dismissals, 1) if batting_innings else None
    total_balls = sum(s.balls_faced or 0 for s in batting_innings)
    sr = (total_bat_runs / total_balls * 100.0) if total_balls else None

    # Bowling
    total_wickets = sum(s.wickets or 0 for s in bowling_matches)
    total_runs_c = sum(s.runs_conceded or 0 for s in bowling_matches)
    total_bowl_balls = sum(int((s.overs_bowled or 0) * 6) for s in bowling_matches)
    bowling_avg = total_runs_c / max(total_wickets, 1) if bowling_matches else None
    bowling_econ = (total_runs_c / total_bowl_balls * 6) if total_bowl_balls else None
    bowling_sr = total_bowl_balls / max(total_wickets, 1) if bowling_matches else None

    return {
        "n_matches": len(stats),
        "n_innings": len(batting_innings),
        "batting_avg": batting_avg,
        "strike_rate": sr,
        "bowling_avg": bowling_avg,
        "bowling_econ": bowling_econ,
        "bowling_sr": bowling_sr,
    }


def _upsert_player_feature(
    player_id: int,
    format: str,
    snapshot_date: str,
    stats: List[PlayerStat],
    rating_dict: Dict,
    session: Session,
) -> None:
    """Insert or update a PlayerFeature row for the given player/format/date."""
    agg = _aggregate_stats(stats)

    # Recent form values from rating_dict
    recent_form_batting = rating_dict.get("recent_form")
    recent_form_bowling = rating_dict.get("recent_form")  # overloaded key
    if "bowling_avg" in rating_dict and "raw_avg" not in rating_dict:
        # Pure bowler — batting form is not relevant
        recent_form_batting = None
    if "raw_avg" in rating_dict and "bowling_avg" not in rating_dict:
        # Pure batter — bowling form is not relevant
        recent_form_bowling = None

    feature_blob = json.dumps({k: v for k, v in rating_dict.items() if isinstance(v, (int, float, str, type(None)))})

    existing = (
        session.query(PlayerFeature)
        .filter(
            PlayerFeature.player_id == player_id,
            PlayerFeature.format == format,
            PlayerFeature.snapshot_date == snapshot_date,
        )
        .first()
    )

    if existing:
        existing.n_matches = agg["n_matches"]
        existing.n_innings = agg["n_innings"]
        existing.batting_avg = agg["batting_avg"]
        existing.strike_rate = agg["strike_rate"]
        existing.bowling_avg = agg["bowling_avg"]
        existing.bowling_econ = agg["bowling_econ"]
        existing.bowling_sr = agg["bowling_sr"]
        existing.recent_form_batting = recent_form_batting
        existing.recent_form_bowling = recent_form_bowling
        existing.rating = rating_dict.get("rating")
        existing.feature_json = feature_blob
    else:
        row = PlayerFeature(
            player_id=player_id,
            snapshot_date=snapshot_date,
            format=format,
            n_matches=agg["n_matches"],
            n_innings=agg["n_innings"],
            batting_avg=agg["batting_avg"],
            strike_rate=agg["strike_rate"],
            bowling_avg=agg["bowling_avg"],
            bowling_econ=agg["bowling_econ"],
            bowling_sr=agg["bowling_sr"],
            recent_form_batting=recent_form_batting,
            recent_form_bowling=recent_form_bowling,
            rating=rating_dict.get("rating"),
            feature_json=feature_blob,
        )
        session.add(row)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_all_ratings(
    session: Session,
    snapshot_date: Optional[str] = None,
) -> int:
    """Compute and persist ratings for every player in the DB, all formats.

    Parameters
    ----------
    session:
        Active SQLAlchemy session.
    snapshot_date:
        The date string (``YYYY-MM-DD``) to tag the snapshot.  Defaults to
        today's date.

    Returns
    -------
    int
        Number of players processed.
    """
    if snapshot_date is None:
        snapshot_date = date.today().strftime("%Y-%m-%d")

    # Invalidate cached population means so we compute fresh values
    _invalidate_global_caches()

    players: List[Player] = session.query(Player).all()
    n_players = len(players)
    logger.info("run_all_ratings: processing %d players, snapshot=%s", n_players, snapshot_date)

    processed = 0
    for player in players:
        for fmt in ["T20", "ODI", "Test"]:
            try:
                stats = _get_player_stats(player.id, fmt, session)
                rating_dict = compute_player_rating(player.id, fmt, session)
                _upsert_player_feature(
                    player.id, fmt, snapshot_date, stats, rating_dict, session
                )
            except Exception as exc:
                logger.warning(
                    "run_all_ratings: error for player_id=%s format=%s: %s",
                    player.id, fmt, exc,
                )
        processed += 1

    try:
        session.commit()
        logger.info("run_all_ratings: committed %d players", processed)
    except Exception:
        session.rollback()
        raise

    return processed
