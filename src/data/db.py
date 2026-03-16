"""
db.py — SQLAlchemy ORM models and session factory.

Models: Team, Player, Match, PlayerStat, Prediction, TelegramUser (unchanged),
plus Phase-2 additions: Venue, Tournament, Innings, Delivery, PlayerFeature,
TeamFeature, ModelRecord, EloRating, PVORMatch, PVORPlayerAgg.

Privacy note: TelegramUser stores only telegram_id (an integer) and
preferences — no names, usernames, or PII are persisted.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean, Column, Float, ForeignKey, Index, Integer, String, Text,
    UniqueConstraint, create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

load_dotenv()

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/cricket.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    __allow_unmapped__ = True


# ---------------------------------------------------------------------------
# Original ORM Models (unchanged)
# ---------------------------------------------------------------------------

class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    country = Column(String)
    team_type = Column(String, default="international")
    created_at = Column(String, default=lambda: _now())


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    country = Column(String)
    role = Column(String)           # batsman | bowler | all-rounder | wicket-keeper
    batting_style = Column(String)
    bowling_style = Column(String)
    dob = Column(String)
    espn_id = Column(Integer, nullable=True, unique=True)
    cricbuzz_id = Column(Integer, nullable=True)
    created_at = Column(String, default=lambda: _now())

    stats: List["PlayerStat"] = relationship("PlayerStat", back_populates="player")
    features: List["PlayerFeature"] = relationship("PlayerFeature", back_populates="player")


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_key = Column(String, unique=True)            # external identifier
    team_a = Column(String, nullable=False)
    team_b = Column(String, nullable=False)
    venue = Column(String)
    venue_id = Column(Integer, ForeignKey("venues.id"), nullable=True)
    match_date = Column(String)
    match_type = Column(String)                        # T20 | ODI | Test
    tournament = Column(String)
    tournament_id = Column(Integer, ForeignKey("tournaments.id"), nullable=True)
    winner = Column(String)
    result_margin = Column(String)
    toss_winner = Column(String)
    toss_decision = Column(String)
    scorecard_json = Column(Text)                      # full scorecard as JSON
    source = Column(String, default="mock")            # mock | cricapi | espn_scrape | cricbuzz
    cricbuzz_match_id = Column(Integer, nullable=True)
    innings_complete = Column(Boolean, default=False)
    created_at = Column(String, default=lambda: _now())
    updated_at = Column(String, default=lambda: _now())

    predictions: List["Prediction"] = relationship("Prediction", back_populates="match")
    player_stats: List["PlayerStat"] = relationship("PlayerStat", back_populates="match")
    innings_list: List["Innings"] = relationship("Innings", back_populates="match")
    elo_ratings: List["EloRating"] = relationship("EloRating", back_populates="match")

    def get_scorecard(self) -> Optional[Dict]:
        if self.scorecard_json:
            try:
                return json.loads(self.scorecard_json)
            except Exception:
                return None
        return None


class PlayerStat(Base):
    __tablename__ = "player_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"))
    match_id = Column(Integer, ForeignKey("matches.id"))
    innings_id = Column(Integer, ForeignKey("innings.id"), nullable=True)
    team = Column(String)
    runs = Column(Integer, default=0)
    balls_faced = Column(Integer, default=0)
    fours = Column(Integer, default=0)
    sixes = Column(Integer, default=0)
    strike_rate = Column(Float, default=0.0)
    wickets = Column(Integer, default=0)
    overs_bowled = Column(Float, default=0.0)
    runs_conceded = Column(Integer, default=0)
    economy_rate = Column(Float, default=0.0)
    catches = Column(Integer, default=0)
    stumpings = Column(Integer, default=0)
    not_out = Column(Boolean, default=False)
    batting_position = Column(Integer, nullable=True)  # 1-11
    bowling_slot = Column(Integer, nullable=True)      # order in which bowler bowled
    created_at = Column(String, default=lambda: _now())

    player: "Player" = relationship("Player", back_populates="stats")
    match: "Match" = relationship("Match", back_populates="player_stats")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"))
    model_version = Column(String)
    team_a_win_prob = Column(Float)
    team_b_win_prob = Column(Float)
    predicted_winner = Column(String)
    confidence = Column(Float)
    key_features_json = Column(Text)   # JSON dict of feature importances
    explanation = Column(Text)         # LLM or fallback explanation
    sim_win_prob_a = Column(Float, nullable=True)   # Monte Carlo win prob
    sim_score_p50_a = Column(Float, nullable=True)  # MC median score team_a
    sim_score_p50_b = Column(Float, nullable=True)
    created_at = Column(String, default=lambda: _now())

    match: "Match" = relationship("Match", back_populates="predictions")

    def get_key_features(self) -> Dict:
        if self.key_features_json:
            try:
                return json.loads(self.key_features_json)
            except Exception:
                return {}
        return {}


class TelegramUser(Base):
    """
    Privacy note: only telegram_id (integer) and preferences are stored.
    No usernames, first names, or other PII.
    """
    __tablename__ = "telegram_users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_id = Column(Integer, nullable=False, unique=True)
    notify_enabled = Column(Integer, default=1)
    language = Column(String, default="en")
    created_at = Column(String, default=lambda: _now())
    updated_at = Column(String, default=lambda: _now())


# ---------------------------------------------------------------------------
# Phase 2 — New ORM Models
# ---------------------------------------------------------------------------

class Venue(Base):
    __tablename__ = "venues"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    city = Column(String)
    country = Column(String)
    # Pitch/conditions factors (1.0 = neutral)
    batting_factor = Column(Float, default=1.0)   # > 1 favours batting
    spin_factor = Column(Float, default=1.0)      # > 1 favours spin bowlers
    pace_factor = Column(Float, default=1.0)      # > 1 favours pace bowlers
    dew_factor = Column(Float, default=0.0)       # 0-1 probability of dew
    avg_first_innings_t20 = Column(Float, nullable=True)
    avg_first_innings_odi = Column(Float, nullable=True)
    avg_first_innings_test = Column(Float, nullable=True)
    home_team = Column(String, nullable=True)
    source = Column(String, default="manual")
    created_at = Column(String, default=lambda: _now())

    matches: List["Match"] = relationship("Match", foreign_keys=[Match.venue_id])


class Tournament(Base):
    __tablename__ = "tournaments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    year = Column(Integer)
    format = Column(String)           # T20 | ODI | Test | Mixed
    host_country = Column(String)
    start_date = Column(String)
    end_date = Column(String)
    importance = Column(String, default="bilateral")  # bilateral | icc_group | icc_knockout
    created_at = Column(String, default=lambda: _now())

    __table_args__ = (UniqueConstraint("name", "year", name="uq_tournament_name_year"),)


class Innings(Base):
    __tablename__ = "innings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    innings_number = Column(Integer, nullable=False)  # 1 or 2 (limited overs); 1-4 (Test)
    batting_team = Column(String, nullable=False)
    bowling_team = Column(String, nullable=False)
    total_runs = Column(Integer, default=0)
    total_wickets = Column(Integer, default=0)
    total_overs = Column(Float, default=0.0)
    extras = Column(Integer, default=0)
    result = Column(String, default="normal")  # normal | D/L | forfeit
    created_at = Column(String, default=lambda: _now())

    match: "Match" = relationship("Match", back_populates="innings_list")
    deliveries: List["Delivery"] = relationship("Delivery", back_populates="innings")

    __table_args__ = (UniqueConstraint("match_id", "innings_number", name="uq_innings_match_num"),)


class Delivery(Base):
    __tablename__ = "deliveries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    innings_id = Column(Integer, ForeignKey("innings.id"), nullable=False)
    over_number = Column(Integer, nullable=False)    # 0-based
    ball_number = Column(Integer, nullable=False)    # 1-6 within over
    batsman_id = Column(Integer, ForeignKey("players.id"), nullable=True)
    bowler_id = Column(Integer, ForeignKey("players.id"), nullable=True)
    batsman_name = Column(String)   # raw scraped name (fallback)
    bowler_name = Column(String)
    runs_scored = Column(Integer, default=0)
    extras = Column(Integer, default=0)
    extra_type = Column(String, nullable=True)  # wide | nb | bye | lb | null
    wicket_type = Column(String, nullable=True) # null | bowled | caught | lbw | run out | etc.
    fielder_name = Column(String, nullable=True)
    is_boundary = Column(Boolean, default=False)
    is_six = Column(Boolean, default=False)
    created_at = Column(String, default=lambda: _now())

    innings: "Innings" = relationship("Innings", back_populates="deliveries")

    __table_args__ = (
        Index("idx_delivery_innings_over", "innings_id", "over_number", "ball_number"),
    )


class PlayerFeature(Base):
    """Daily snapshots of computed player features and ratings."""
    __tablename__ = "player_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    snapshot_date = Column(String, nullable=False)    # YYYY-MM-DD
    format = Column(String, nullable=False)           # T20 | ODI | Test | ALL
    n_matches = Column(Integer, default=0)
    n_innings = Column(Integer, default=0)
    batting_avg = Column(Float, nullable=True)
    strike_rate = Column(Float, nullable=True)
    bowling_avg = Column(Float, nullable=True)
    bowling_econ = Column(Float, nullable=True)
    bowling_sr = Column(Float, nullable=True)
    recent_form_batting = Column(Float, nullable=True)  # exp-decay weighted avg
    recent_form_bowling = Column(Float, nullable=True)
    rating = Column(Float, nullable=True)               # composite 0-100 score
    feature_json = Column(Text, nullable=True)          # full feature blob
    created_at = Column(String, default=lambda: _now())

    player: "Player" = relationship("Player", back_populates="features")

    __table_args__ = (
        UniqueConstraint("player_id", "snapshot_date", "format", name="uq_player_feature"),
    )

    def get_features(self) -> Dict:
        if self.feature_json:
            try:
                return json.loads(self.feature_json)
            except Exception:
                return {}
        return {}


class TeamFeature(Base):
    """Daily snapshots of team strength ratings."""
    __tablename__ = "team_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    team_name = Column(String, nullable=False)
    snapshot_date = Column(String, nullable=False)
    format = Column(String, nullable=False)
    rating = Column(Float, nullable=True)             # aggregated from top-11
    recent_win_pct = Column(Float, nullable=True)     # last 10 matches
    home_advantage = Column(Float, default=0.0)
    expected_xi_strength = Column(Float, nullable=True)
    feature_json = Column(Text, nullable=True)
    created_at = Column(String, default=lambda: _now())

    __table_args__ = (
        UniqueConstraint("team_name", "snapshot_date", "format", name="uq_team_feature"),
    )

    def get_features(self) -> Dict:
        if self.feature_json:
            try:
                return json.loads(self.feature_json)
            except Exception:
                return {}
        return {}


class ModelRecord(Base):
    """Tracks ML model versions and their evaluation metrics."""
    __tablename__ = "model_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String, nullable=False, unique=True)
    trained_at = Column(String, nullable=False)
    accuracy = Column(Float, nullable=True)
    log_loss = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)
    n_train = Column(Integer, nullable=True)
    n_test = Column(Integer, nullable=True)
    feature_cols_json = Column(Text, nullable=True)
    model_path = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(String, default=lambda: _now())


class EloRating(Base):
    """Per-team Elo rating history, one row per match."""
    __tablename__ = "elo_ratings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    team_name = Column(String, nullable=False)
    format = Column(String, nullable=False)   # T20 | ODI | Test
    rating = Column(Float, nullable=False, default=1500.0)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=True)
    match_date = Column(String, nullable=True)
    created_at = Column(String, default=lambda: _now())

    match: "Match" = relationship("Match", back_populates="elo_ratings")

    __table_args__ = (
        UniqueConstraint("team_name", "format", "match_id", name="uq_elo_team_match"),
        Index("idx_elo_team_format_date", "team_name", "format", "match_date"),
    )


class PVORMatch(Base):
    """Player Value Over Replacement for each match."""
    __tablename__ = "pvor_match"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    format = Column(String, nullable=True)
    batting_pvor = Column(Float, nullable=True)   # runs above replacement batting
    bowling_pvor = Column(Float, nullable=True)   # wickets above replacement bowling
    fielding_pvor = Column(Float, nullable=True)
    total_pvor = Column(Float, nullable=True)
    created_at = Column(String, default=lambda: _now())

    __table_args__ = (
        UniqueConstraint("player_id", "match_id", name="uq_pvor_player_match"),
    )


class PVORPlayerAgg(Base):
    """Aggregated PVOR per player over different time windows."""
    __tablename__ = "pvor_player_agg"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    format = Column(String, nullable=False)
    period = Column(String, nullable=False)        # last30d | last90d | career
    batting_pvor_avg = Column(Float, nullable=True)
    bowling_pvor_avg = Column(Float, nullable=True)
    total_pvor_avg = Column(Float, nullable=True)
    n_matches = Column(Integer, default=0)
    snapshot_date = Column(String, nullable=False)
    created_at = Column(String, default=lambda: _now())

    __table_args__ = (
        UniqueConstraint(
            "player_id", "format", "period", "snapshot_date",
            name="uq_pvor_agg_player_period"
        ),
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    """Return a new SQLAlchemy session. Caller is responsible for closing."""
    return SessionLocal()
