"""
db.py — SQLAlchemy ORM models and session factory.

Models: Team, Player, Match, PlayerStat, Prediction, TelegramUser.
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
    Column, Float, ForeignKey, Index, Integer, String, Text,
    create_engine, text,
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
    pass


# ---------------------------------------------------------------------------
# ORM Models
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
    created_at = Column(String, default=lambda: _now())

    stats: List["PlayerStat"] = relationship("PlayerStat", back_populates="player")


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_key = Column(String, unique=True)
    team_a = Column(String, nullable=False)
    team_b = Column(String, nullable=False)
    venue = Column(String)
    match_date = Column(String)
    match_type = Column(String)
    tournament = Column(String)
    winner = Column(String)
    result_margin = Column(String)
    toss_winner = Column(String)
    toss_decision = Column(String)
    scorecard_json = Column(Text)
    source = Column(String, default="mock")
    created_at = Column(String, default=lambda: _now())
    updated_at = Column(String, default=lambda: _now())

    predictions: List["Prediction"] = relationship("Prediction", back_populates="match")
    player_stats: List["PlayerStat"] = relationship("PlayerStat", back_populates="match")

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
    key_features_json = Column(Text)
    explanation = Column(Text)
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
