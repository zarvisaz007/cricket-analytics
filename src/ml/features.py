"""
features.py — Feature engineering pipeline for match prediction.

Single source of truth for the 18-feature vector used by the XGBoost model.
All feature computation is centralised here to ensure consistency between
training (build_training_dataset) and inference (build_feature_vector).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature definitions — canonical 18-feature list
# ---------------------------------------------------------------------------

FEATURE_COLS: List[str] = [
    "team_a_elo",
    "team_b_elo",
    "elo_diff",
    "team_a_strength",
    "team_b_strength",
    "team_a_form_last10",
    "team_b_form_last10",
    "h2h_team_a_win_rate",
    "h2h_total",
    "venue_batting_factor",
    "venue_spin_factor",
    "toss_winner_batting",
    "toss_winner_is_team_a",
    "match_type_encoded",
    "team_a_top_batsman_rating",
    "team_b_top_batsman_rating",
    "team_a_top_bowler_rating",
    "team_b_top_bowler_rating",
]

DEFAULT_FEATURES: Dict[str, float] = {
    "team_a_elo": 1500.0,
    "team_b_elo": 1500.0,
    "elo_diff": 0.0,
    "team_a_strength": 50.0,
    "team_b_strength": 50.0,
    "team_a_form_last10": 0.5,
    "team_b_form_last10": 0.5,
    "h2h_team_a_win_rate": 0.5,
    "h2h_total": 10.0,
    "venue_batting_factor": 1.0,
    "venue_spin_factor": 1.0,
    "toss_winner_batting": 0.5,
    "toss_winner_is_team_a": 0.5,
    "match_type_encoded": 0.0,
    "team_a_top_batsman_rating": 50.0,
    "team_b_top_batsman_rating": 50.0,
    "team_a_top_bowler_rating": 50.0,
    "team_b_top_bowler_rating": 50.0,
}


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_match_type(mt: str) -> float:
    """Encode match type as a numeric value: T20=0, ODI=1, Test=2."""
    return float({"T20": 0, "ODI": 1, "TEST": 2}.get(mt.upper(), 0))


# ---------------------------------------------------------------------------
# Player rating helpers
# ---------------------------------------------------------------------------

def _get_top_batsman_rating(team_name: str, match_type: str, session: Session) -> float:
    """Return the highest batting-oriented player rating for the team."""
    try:
        from src.data.db import PlayerFeature, PlayerStat, Match as MatchModel

        player_id_rows = (
            session.query(PlayerStat.player_id)
            .join(MatchModel, PlayerStat.match_id == MatchModel.id)
            .filter(
                PlayerStat.team == team_name,
                MatchModel.match_type == match_type,
            )
            .distinct()
            .all()
        )
        player_ids = [r[0] for r in player_id_rows if r[0] is not None]
        if not player_ids:
            return DEFAULT_FEATURES["team_a_top_batsman_rating"]

        best_rating: Optional[float] = None
        for pid in player_ids:
            row = (
                session.query(PlayerFeature)
                .filter(
                    PlayerFeature.player_id == pid,
                    PlayerFeature.format == match_type,
                    PlayerFeature.batting_avg.isnot(None),
                    PlayerFeature.rating.isnot(None),
                )
                .order_by(PlayerFeature.snapshot_date.desc())
                .first()
            )
            if row and row.rating is not None:
                if best_rating is None or row.rating > best_rating:
                    best_rating = row.rating

        return best_rating if best_rating is not None else DEFAULT_FEATURES["team_a_top_batsman_rating"]
    except Exception as exc:
        logger.debug("_get_top_batsman_rating failed for %s: %s", team_name, exc)
        return DEFAULT_FEATURES["team_a_top_batsman_rating"]


def _get_top_bowler_rating(team_name: str, match_type: str, session: Session) -> float:
    """Return the highest bowling-oriented player rating for the team."""
    try:
        from src.data.db import PlayerFeature, PlayerStat, Match as MatchModel

        player_id_rows = (
            session.query(PlayerStat.player_id)
            .join(MatchModel, PlayerStat.match_id == MatchModel.id)
            .filter(
                PlayerStat.team == team_name,
                MatchModel.match_type == match_type,
            )
            .distinct()
            .all()
        )
        player_ids = [r[0] for r in player_id_rows if r[0] is not None]
        if not player_ids:
            return DEFAULT_FEATURES["team_a_top_bowler_rating"]

        best_rating: Optional[float] = None
        for pid in player_ids:
            row = (
                session.query(PlayerFeature)
                .filter(
                    PlayerFeature.player_id == pid,
                    PlayerFeature.format == match_type,
                    PlayerFeature.bowling_avg.isnot(None),
                    PlayerFeature.rating.isnot(None),
                )
                .order_by(PlayerFeature.snapshot_date.desc())
                .first()
            )
            if row and row.rating is not None:
                if best_rating is None or row.rating > best_rating:
                    best_rating = row.rating

        return best_rating if best_rating is not None else DEFAULT_FEATURES["team_a_top_bowler_rating"]
    except Exception as exc:
        logger.debug("_get_top_bowler_rating failed for %s: %s", team_name, exc)
        return DEFAULT_FEATURES["team_a_top_bowler_rating"]


# ---------------------------------------------------------------------------
# Core feature builder
# ---------------------------------------------------------------------------

def build_feature_vector(
    match_dict: Dict[str, Any],
    session: Session,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build an 18-feature vector for a match.

    Parameters
    ----------
    match_dict:
        Keys: team_a, team_b, match_type, venue (optional),
              toss_winner (optional), toss_decision (optional),
              match_date (optional).
    session:
        Active SQLAlchemy session used for DB lookups.

    Returns
    -------
    (feature_array_shape_1xN, feature_names_list)
    """
    team_a: str = match_dict.get("team_a", "")
    team_b: str = match_dict.get("team_b", "")
    match_type: str = match_dict.get("match_type", "T20") or "T20"
    venue: str = match_dict.get("venue", "") or ""
    toss_winner: str = match_dict.get("toss_winner", "") or ""
    toss_decision: str = match_dict.get("toss_decision", "") or ""

    features: Dict[str, float] = dict(DEFAULT_FEATURES)  # start with defaults

    # -- Elo ratings ----------------------------------------------------------
    try:
        from src.analytics.elo import get_current_elo
        features["team_a_elo"] = get_current_elo(team_a, match_type, session)
        features["team_b_elo"] = get_current_elo(team_b, match_type, session)
        features["elo_diff"] = features["team_a_elo"] - features["team_b_elo"]
    except Exception as exc:
        logger.debug("Elo lookup failed: %s", exc)

    # -- Team strength --------------------------------------------------------
    try:
        from src.analytics.team_strength import compute_team_strength
        strength_a = compute_team_strength(
            team_a, match_type, session,
            opponent=team_b, venue_name=venue or None,
        )
        strength_b = compute_team_strength(
            team_b, match_type, session,
            opponent=team_a, venue_name=venue or None,
        )
        features["team_a_strength"] = float(strength_a.get("final_strength") or 0.0) or DEFAULT_FEATURES["team_a_strength"]
        features["team_b_strength"] = float(strength_b.get("final_strength") or 0.0) or DEFAULT_FEATURES["team_b_strength"]
    except Exception as exc:
        logger.debug("Team strength lookup failed: %s", exc)

    # -- Recent form (last 10) ------------------------------------------------
    try:
        from src.analytics.team_strength import get_recent_win_pct
        features["team_a_form_last10"] = get_recent_win_pct(team_a, match_type, n=10, session=session)
        features["team_b_form_last10"] = get_recent_win_pct(team_b, match_type, n=10, session=session)
    except Exception as exc:
        logger.debug("Form lookup failed: %s", exc)

    # -- Head-to-head ---------------------------------------------------------
    try:
        from src.data.db import Match as MatchModel
        from sqlalchemy import or_

        h2h_matches = (
            session.query(MatchModel)
            .filter(
                MatchModel.winner.isnot(None),
                MatchModel.match_type == match_type,
                or_(
                    (MatchModel.team_a == team_a) & (MatchModel.team_b == team_b),
                    (MatchModel.team_a == team_b) & (MatchModel.team_b == team_a),
                ),
            )
            .all()
        )
        h2h_total = len(h2h_matches)
        if h2h_total > 0:
            wins_a = sum(1 for m in h2h_matches if m.winner == team_a)
            features["h2h_team_a_win_rate"] = wins_a / h2h_total
            features["h2h_total"] = float(h2h_total)
    except Exception as exc:
        logger.debug("H2H lookup failed: %s", exc)

    # -- Venue factors --------------------------------------------------------
    try:
        from src.data.db import Venue as VenueModel

        if venue:
            venue_row = (
                session.query(VenueModel)
                .filter(VenueModel.name.ilike(f"%{venue}%"))
                .first()
            )
            if venue_row:
                if venue_row.batting_factor is not None:
                    features["venue_batting_factor"] = float(venue_row.batting_factor)
                if venue_row.spin_factor is not None:
                    features["venue_spin_factor"] = float(venue_row.spin_factor)
    except Exception as exc:
        logger.debug("Venue lookup failed: %s", exc)

    # -- Toss features --------------------------------------------------------
    try:
        features["toss_winner_batting"] = 1.0 if toss_decision.lower() == "bat" else 0.0
        features["toss_winner_is_team_a"] = 1.0 if toss_winner == team_a else 0.0
    except Exception as exc:
        logger.debug("Toss feature failed: %s", exc)

    # -- Match type encoding --------------------------------------------------
    try:
        features["match_type_encoded"] = encode_match_type(match_type)
    except Exception as exc:
        logger.debug("Match type encoding failed: %s", exc)

    # -- Player ratings -------------------------------------------------------
    try:
        features["team_a_top_batsman_rating"] = _get_top_batsman_rating(team_a, match_type, session)
        features["team_b_top_batsman_rating"] = _get_top_batsman_rating(team_b, match_type, session)
    except Exception as exc:
        logger.debug("Batsman rating lookup failed: %s", exc)

    try:
        features["team_a_top_bowler_rating"] = _get_top_bowler_rating(team_a, match_type, session)
        features["team_b_top_bowler_rating"] = _get_top_bowler_rating(team_b, match_type, session)
    except Exception as exc:
        logger.debug("Bowler rating lookup failed: %s", exc)

    # Build ordered array
    arr = np.array([[features[col] for col in FEATURE_COLS]], dtype=np.float64)
    return arr, FEATURE_COLS


# ---------------------------------------------------------------------------
# Training dataset builder
# ---------------------------------------------------------------------------

def build_training_dataset(session: Session) -> Optional[pd.DataFrame]:
    """
    Load all completed matches from the DB and build the training DataFrame.

    Returns
    -------
    DataFrame with FEATURE_COLS + 'label' column, or None if fewer than 20
    labelled matches are available.
    """
    try:
        from src.data.db import Match as MatchModel

        matches = (
            session.query(MatchModel)
            .filter(MatchModel.winner.isnot(None))
            .all()
        )

        if len(matches) < 20:
            logger.info(
                "[features] Only %d labelled matches — insufficient for training",
                len(matches),
            )
            return None

        rows = []
        for m in matches:
            try:
                match_dict = {
                    "team_a": m.team_a,
                    "team_b": m.team_b,
                    "match_type": m.match_type or "T20",
                    "venue": m.venue or "",
                    "toss_winner": m.toss_winner or "",
                    "toss_decision": m.toss_decision or "",
                    "match_date": m.match_date or "",
                }
                arr, cols = build_feature_vector(match_dict, session)
                label = 1 if m.winner == m.team_a else 0
                row = {col: float(arr[0, i]) for i, col in enumerate(cols)}
                row["label"] = label
                rows.append(row)
            except Exception as exc:
                logger.debug("[features] Skipping match %s: %s", m.id, exc)
                continue

        if not rows:
            logger.warning("[features] No rows could be built from matches")
            return None

        df = pd.DataFrame(rows)
        logger.info("[features] Built training dataset: %d rows", len(df))
        return df

    except Exception as exc:
        logger.warning("[features] build_training_dataset failed: %s", exc)
        return None
