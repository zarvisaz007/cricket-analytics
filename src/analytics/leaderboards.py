"""
leaderboards.py — Query helpers for the Telegram bot and REST API.

Provides functions to retrieve ranked leaderboards and detailed player /
team profiles from the analytics tables populated by elo.py,
player_rating.py, team_strength.py, and pvor.py.

All functions accept an optional ``session`` parameter.  When ``None``, a
fresh session is created and closed automatically before returning.

Leaderboard functions return plain Python lists of dicts so that callers are
not tightly coupled to SQLAlchemy row objects.

Text-formatting helpers return Markdown strings suitable for sending via the
Telegram Bot API (``parse_mode=Markdown``).

Public API
----------
- get_batting_leaderboard(format, limit, session) -> List[Dict]
- get_bowling_leaderboard(format, limit, session) -> List[Dict]
- get_elo_leaderboard(format, limit, session) -> List[Dict]
- get_player_profile(player_id, session) -> Optional[Dict]
- get_h2h_summary(team_a, team_b, format, n, session) -> Dict
- format_batting_leaderboard_text(leaders) -> str
- format_bowling_leaderboard_text(leaders) -> str
- format_elo_leaderboard_text(leaders) -> str
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import date, datetime
from typing import Dict, Generator, List, Optional

from sqlalchemy import desc, func, or_
from sqlalchemy.orm import Session

from src.data.db import (
    EloRating,
    Match,
    Player,
    PlayerFeature,
    PlayerStat,
    PVORPlayerAgg,
    get_session,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session context manager
# ---------------------------------------------------------------------------

@contextmanager
def _managed_session(session: Optional[Session]) -> Generator[Session, None, None]:
    """Yield *session* if provided, otherwise create and close one."""
    if session is not None:
        yield session
    else:
        s = get_session()
        try:
            yield s
        finally:
            s.close()


# ---------------------------------------------------------------------------
# Batting leaderboard
# ---------------------------------------------------------------------------

def get_batting_leaderboard(
    format: str,
    limit: int = 10,
    session: Optional[Session] = None,
) -> List[Dict]:
    """Return the top *limit* players ordered by batting rating.

    Parameters
    ----------
    format:
        One of ``T20``, ``ODI``, ``Test``, or ``ALL``.  ``ALL`` returns the
        highest rating across any format.
    limit:
        Maximum number of entries to return.
    session:
        Optional existing session.

    Returns
    -------
    List[Dict] with keys: rank, player_id, name, country, role, rating,
                          batting_avg, strike_rate, n_matches.
    """
    with _managed_session(session) as s:
        try:
            # Subquery: latest snapshot per (player_id, format)
            subq = (
                s.query(
                    PlayerFeature.player_id,
                    PlayerFeature.format,
                    func.max(PlayerFeature.snapshot_date).label("latest_date"),
                )
                .group_by(PlayerFeature.player_id, PlayerFeature.format)
                .subquery()
            )

            if format == "ALL":
                # Best rating across any format
                best_subq = (
                    s.query(
                        PlayerFeature.player_id,
                        func.max(PlayerFeature.rating).label("best_rating"),
                    )
                    .join(
                        subq,
                        (PlayerFeature.player_id == subq.c.player_id)
                        & (PlayerFeature.format == subq.c.format)
                        & (PlayerFeature.snapshot_date == subq.c.latest_date),
                    )
                    .group_by(PlayerFeature.player_id)
                    .subquery()
                )
                rows = (
                    s.query(
                        Player,
                        PlayerFeature,
                    )
                    .join(best_subq, Player.id == best_subq.c.player_id)
                    .join(
                        PlayerFeature,
                        (PlayerFeature.player_id == Player.id)
                        & (PlayerFeature.rating == best_subq.c.best_rating),
                    )
                    .join(
                        subq,
                        (PlayerFeature.player_id == subq.c.player_id)
                        & (PlayerFeature.format == subq.c.format)
                        & (PlayerFeature.snapshot_date == subq.c.latest_date),
                    )
                    .filter(PlayerFeature.rating.isnot(None))
                    .order_by(desc(PlayerFeature.rating))
                    .limit(limit)
                    .all()
                )
            else:
                rows = (
                    s.query(Player, PlayerFeature)
                    .join(PlayerFeature, Player.id == PlayerFeature.player_id)
                    .join(
                        subq,
                        (PlayerFeature.player_id == subq.c.player_id)
                        & (PlayerFeature.format == subq.c.format)
                        & (PlayerFeature.snapshot_date == subq.c.latest_date),
                    )
                    .filter(
                        PlayerFeature.format == format,
                        PlayerFeature.rating.isnot(None),
                    )
                    .order_by(desc(PlayerFeature.rating))
                    .limit(limit)
                    .all()
                )

            result = []
            for rank, (player, pf) in enumerate(rows, start=1):
                result.append({
                    "rank": rank,
                    "player_id": player.id,
                    "name": player.name,
                    "country": player.country,
                    "role": player.role,
                    "rating": round(pf.rating or 0.0, 2),
                    "batting_avg": round(pf.batting_avg or 0.0, 2),
                    "strike_rate": round(pf.strike_rate or 0.0, 2),
                    "n_matches": pf.n_matches or 0,
                })
            return result

        except Exception as exc:
            logger.error("get_batting_leaderboard: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Bowling leaderboard
# ---------------------------------------------------------------------------

def get_bowling_leaderboard(
    format: str,
    limit: int = 10,
    session: Optional[Session] = None,
) -> List[Dict]:
    """Return the top *limit* players by bowling component.

    Players are ranked using the stored ``rating`` on their latest
    PlayerFeature snapshot.  Only players whose role contains "bowl" or
    "all" are included (pure batsmen are filtered out by requiring bowling
    stats to be non-null).

    Returns
    -------
    List[Dict] with keys: rank, player_id, name, country, role, rating,
                          bowling_avg, bowling_econ, wickets.
    """
    with _managed_session(session) as s:
        try:
            subq = (
                s.query(
                    PlayerFeature.player_id,
                    PlayerFeature.format,
                    func.max(PlayerFeature.snapshot_date).label("latest_date"),
                )
                .group_by(PlayerFeature.player_id, PlayerFeature.format)
                .subquery()
            )

            if format == "ALL":
                rows = (
                    s.query(Player, PlayerFeature)
                    .join(PlayerFeature, Player.id == PlayerFeature.player_id)
                    .join(
                        subq,
                        (PlayerFeature.player_id == subq.c.player_id)
                        & (PlayerFeature.format == subq.c.format)
                        & (PlayerFeature.snapshot_date == subq.c.latest_date),
                    )
                    .filter(
                        PlayerFeature.bowling_avg.isnot(None),
                        PlayerFeature.rating.isnot(None),
                    )
                    .order_by(desc(PlayerFeature.rating))
                    .limit(limit)
                    .all()
                )
            else:
                rows = (
                    s.query(Player, PlayerFeature)
                    .join(PlayerFeature, Player.id == PlayerFeature.player_id)
                    .join(
                        subq,
                        (PlayerFeature.player_id == subq.c.player_id)
                        & (PlayerFeature.format == subq.c.format)
                        & (PlayerFeature.snapshot_date == subq.c.latest_date),
                    )
                    .filter(
                        PlayerFeature.format == format,
                        PlayerFeature.bowling_avg.isnot(None),
                        PlayerFeature.rating.isnot(None),
                    )
                    .order_by(desc(PlayerFeature.rating))
                    .limit(limit)
                    .all()
                )

            result = []
            for rank, (player, pf) in enumerate(rows, start=1):
                # Count career wickets from PlayerStat
                wickets_total = _career_wickets(player.id, format, s)
                result.append({
                    "rank": rank,
                    "player_id": player.id,
                    "name": player.name,
                    "country": player.country,
                    "role": player.role,
                    "rating": round(pf.rating or 0.0, 2),
                    "bowling_avg": round(pf.bowling_avg or 0.0, 2),
                    "bowling_econ": round(pf.bowling_econ or 0.0, 2),
                    "wickets": wickets_total,
                })
            return result

        except Exception as exc:
            logger.error("get_bowling_leaderboard: %s", exc)
            return []


def _career_wickets(player_id: int, format: str, session: Session) -> int:
    """Return total career wickets for a player in the given format."""
    if format == "ALL":
        result = (
            session.query(func.sum(PlayerStat.wickets))
            .join(Match, PlayerStat.match_id == Match.id)
            .filter(PlayerStat.player_id == player_id)
            .scalar()
        )
    else:
        result = (
            session.query(func.sum(PlayerStat.wickets))
            .join(Match, PlayerStat.match_id == Match.id)
            .filter(
                PlayerStat.player_id == player_id,
                Match.match_type == format,
            )
            .scalar()
        )
    return int(result or 0)


# ---------------------------------------------------------------------------
# Elo leaderboard
# ---------------------------------------------------------------------------

def get_elo_leaderboard(
    format: str,
    limit: int = 10,
    session: Optional[Session] = None,
) -> List[Dict]:
    """Return the top *limit* teams by latest Elo rating.

    Parameters
    ----------
    format:
        T20 | ODI | Test.  Use ``ALL`` to return the single highest Elo rating
        for each team across all formats.

    Returns
    -------
    List[Dict] with keys: rank, team_name, elo_rating, last_updated.
    """
    with _managed_session(session) as s:
        try:
            if format == "ALL":
                # Latest rating per team, any format
                subq = (
                    s.query(
                        EloRating.team_name,
                        func.max(EloRating.match_date).label("latest_date"),
                    )
                    .group_by(EloRating.team_name)
                    .subquery()
                )
                rows = (
                    s.query(EloRating)
                    .join(
                        subq,
                        (EloRating.team_name == subq.c.team_name)
                        & (EloRating.match_date == subq.c.latest_date),
                    )
                    .order_by(desc(EloRating.rating))
                    .limit(limit)
                    .all()
                )
            else:
                subq = (
                    s.query(
                        EloRating.team_name,
                        func.max(EloRating.match_date).label("latest_date"),
                    )
                    .filter(EloRating.format == format)
                    .group_by(EloRating.team_name)
                    .subquery()
                )
                rows = (
                    s.query(EloRating)
                    .join(
                        subq,
                        (EloRating.team_name == subq.c.team_name)
                        & (EloRating.match_date == subq.c.latest_date),
                    )
                    .filter(EloRating.format == format)
                    .order_by(desc(EloRating.rating))
                    .limit(limit)
                    .all()
                )

            result = []
            for rank, row in enumerate(rows, start=1):
                result.append({
                    "rank": rank,
                    "team_name": row.team_name,
                    "elo_rating": round(row.rating, 1),
                    "last_updated": row.match_date or "",
                })
            return result

        except Exception as exc:
            logger.error("get_elo_leaderboard: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Player profile
# ---------------------------------------------------------------------------

def get_player_profile(
    player_id: int,
    session: Optional[Session] = None,
) -> Optional[Dict]:
    """Return a comprehensive profile dict for a player.

    Returns
    -------
    Dict containing:
    - Base player info (name, country, role, dob, age)
    - Latest PlayerFeature per format (T20, ODI, Test)
    - Last 5 match stats
    - Current PVOR aggregate per format
    - Prime year (year with highest average rating from PlayerFeature history)

    Returns ``None`` if the player is not found.
    """
    with _managed_session(session) as s:
        try:
            player: Optional[Player] = s.get(Player, player_id)
            if player is None:
                return None

            # ---- Age -------------------------------------------------------
            age = None
            if player.dob:
                try:
                    dob = datetime.strptime(player.dob[:10], "%Y-%m-%d").date()
                    today = date.today()
                    age = today.year - dob.year - (
                        (today.month, today.day) < (dob.month, dob.day)
                    )
                except (ValueError, AttributeError):
                    pass

            profile: Dict = {
                "player_id": player.id,
                "name": player.name,
                "country": player.country,
                "role": player.role,
                "batting_style": player.batting_style,
                "bowling_style": player.bowling_style,
                "dob": player.dob,
                "age": age,
            }

            # ---- Latest features per format --------------------------------
            features_by_format: Dict[str, Optional[Dict]] = {}
            for fmt in ["T20", "ODI", "Test"]:
                pf = (
                    s.query(PlayerFeature)
                    .filter(
                        PlayerFeature.player_id == player_id,
                        PlayerFeature.format == fmt,
                    )
                    .order_by(desc(PlayerFeature.snapshot_date))
                    .first()
                )
                if pf:
                    features_by_format[fmt] = {
                        "rating": pf.rating,
                        "batting_avg": pf.batting_avg,
                        "strike_rate": pf.strike_rate,
                        "bowling_avg": pf.bowling_avg,
                        "bowling_econ": pf.bowling_econ,
                        "bowling_sr": pf.bowling_sr,
                        "n_matches": pf.n_matches,
                        "n_innings": pf.n_innings,
                        "snapshot_date": pf.snapshot_date,
                    }
                else:
                    features_by_format[fmt] = None
            profile["features"] = features_by_format

            # ---- Last 5 matches --------------------------------------------
            last5_stats = (
                s.query(PlayerStat, Match)
                .join(Match, PlayerStat.match_id == Match.id)
                .filter(PlayerStat.player_id == player_id)
                .order_by(desc(Match.match_date), desc(Match.id))
                .limit(5)
                .all()
            )
            last5 = []
            for stat, match in last5_stats:
                last5.append({
                    "match_id": match.id,
                    "match_date": match.match_date,
                    "match_type": match.match_type,
                    "team_a": match.team_a,
                    "team_b": match.team_b,
                    "winner": match.winner,
                    "runs": stat.runs,
                    "balls_faced": stat.balls_faced,
                    "wickets": stat.wickets,
                    "overs_bowled": stat.overs_bowled,
                    "economy_rate": stat.economy_rate,
                    "catches": stat.catches,
                    "stumpings": stat.stumpings,
                })
            profile["last_5_matches"] = last5

            # ---- Current PVOR aggregates -----------------------------------
            pvor_data: Dict[str, Dict] = {}
            pvor_rows = (
                s.query(PVORPlayerAgg)
                .filter(PVORPlayerAgg.player_id == player_id)
                .all()
            )
            for row in pvor_rows:
                key = f"{row.format}_{row.period}"
                pvor_data[key] = {
                    "format": row.format,
                    "period": row.period,
                    "batting_pvor_avg": row.batting_pvor_avg,
                    "bowling_pvor_avg": row.bowling_pvor_avg,
                    "total_pvor_avg": row.total_pvor_avg,
                    "n_matches": row.n_matches,
                    "snapshot_date": row.snapshot_date,
                }
            profile["pvor"] = pvor_data

            # ---- Prime year ------------------------------------------------
            prime_year = _get_prime_year(player_id, s)
            profile["prime_year"] = prime_year

            return profile

        except Exception as exc:
            logger.error("get_player_profile: player_id=%s: %s", player_id, exc)
            return None


def _get_prime_year(player_id: int, session: Session) -> Optional[int]:
    """Return the year in which the player had the highest average rating."""
    rows = (
        session.query(PlayerFeature.snapshot_date, PlayerFeature.rating)
        .filter(
            PlayerFeature.player_id == player_id,
            PlayerFeature.rating.isnot(None),
        )
        .all()
    )
    if not rows:
        return None

    year_ratings: Dict[int, List[float]] = {}
    for snapshot_date, rating in rows:
        try:
            year = int(snapshot_date[:4])
        except (TypeError, ValueError):
            continue
        year_ratings.setdefault(year, []).append(rating)

    if not year_ratings:
        return None

    best_year = max(year_ratings, key=lambda y: sum(year_ratings[y]) / len(year_ratings[y]))
    return best_year


# ---------------------------------------------------------------------------
# Head-to-head summary
# ---------------------------------------------------------------------------

def get_h2h_summary(
    team_a: str,
    team_b: str,
    format: Optional[str] = None,
    n: int = 20,
    session: Optional[Session] = None,
) -> Dict:
    """Return a head-to-head match history summary between two teams.

    Parameters
    ----------
    team_a, team_b:
        Team names.
    format:
        If None, considers all formats.
    n:
        Maximum number of recent matches to consider.

    Returns
    -------
    Dict with keys: team_a, team_b, total_matches, team_a_wins, team_b_wins,
                    ties, last_5_results, win_pct_a.
    """
    with _managed_session(session) as s:
        try:
            query = (
                s.query(Match)
                .filter(
                    or_(
                        (Match.team_a == team_a) & (Match.team_b == team_b),
                        (Match.team_a == team_b) & (Match.team_b == team_a),
                    )
                )
                .order_by(desc(Match.match_date), desc(Match.id))
            )
            if format:
                query = query.filter(Match.match_type == format)

            matches: List[Match] = query.limit(n).all()

            team_a_wins = 0
            team_b_wins = 0
            ties = 0
            last_5: List[Dict] = []

            for i, m in enumerate(matches):
                winner = m.winner or ""
                if winner == team_a:
                    team_a_wins += 1
                    result_str = f"{team_a} won"
                elif winner == team_b:
                    team_b_wins += 1
                    result_str = f"{team_b} won"
                elif winner.lower() in ("tie", "draw", "no result", ""):
                    ties += 1
                    result_str = "Tie/No result"
                else:
                    result_str = f"{winner} won"

                if i < 5:
                    last_5.append({
                        "match_date": m.match_date,
                        "match_type": m.match_type,
                        "venue": m.venue,
                        "result": result_str,
                    })

            total = len(matches)
            win_pct_a = (team_a_wins / total) if total > 0 else 0.5

            return {
                "team_a": team_a,
                "team_b": team_b,
                "total_matches": total,
                "team_a_wins": team_a_wins,
                "team_b_wins": team_b_wins,
                "ties": ties,
                "last_5_results": last_5,
                "win_pct_a": round(win_pct_a, 3),
            }

        except Exception as exc:
            logger.error("get_h2h_summary: %s vs %s: %s", team_a, team_b, exc)
            return {
                "team_a": team_a,
                "team_b": team_b,
                "total_matches": 0,
                "team_a_wins": 0,
                "team_b_wins": 0,
                "ties": 0,
                "last_5_results": [],
                "win_pct_a": 0.5,
            }


# ---------------------------------------------------------------------------
# Markdown text formatters
# ---------------------------------------------------------------------------

def format_batting_leaderboard_text(leaders: List[Dict]) -> str:
    """Format a batting leaderboard as Markdown text for Telegram.

    Example output::

        *Top Batters — T20*
        1. Virat Kohli (India) — Rating: 87.4 | Avg: 52.7 | SR: 138.2
        2. ...
    """
    if not leaders:
        return "_No batting data available._"

    lines = ["*Top Batters*\n"]
    for entry in leaders:
        name = entry.get("name", "Unknown")
        country = entry.get("country") or "—"
        rating = entry.get("rating", 0.0)
        avg = entry.get("batting_avg", 0.0)
        sr = entry.get("strike_rate", 0.0)
        rank = entry.get("rank", "?")
        lines.append(
            f"{rank}. *{name}* ({country})\n"
            f"   Rating: `{rating:.1f}` | Avg: `{avg:.1f}` | SR: `{sr:.1f}`"
        )
    return "\n".join(lines)


def format_bowling_leaderboard_text(leaders: List[Dict]) -> str:
    """Format a bowling leaderboard as Markdown text for Telegram.

    Example output::

        *Top Bowlers*
        1. Jasprit Bumrah (India) — Rating: 91.2 | Avg: 22.5 | Econ: 6.8 | Wkts: 120
        2. ...
    """
    if not leaders:
        return "_No bowling data available._"

    lines = ["*Top Bowlers*\n"]
    for entry in leaders:
        name = entry.get("name", "Unknown")
        country = entry.get("country") or "—"
        rating = entry.get("rating", 0.0)
        avg = entry.get("bowling_avg", 0.0)
        econ = entry.get("bowling_econ", 0.0)
        wkts = entry.get("wickets", 0)
        rank = entry.get("rank", "?")
        lines.append(
            f"{rank}. *{name}* ({country})\n"
            f"   Rating: `{rating:.1f}` | Avg: `{avg:.1f}` | Econ: `{econ:.2f}` | Wkts: `{wkts}`"
        )
    return "\n".join(lines)


def format_elo_leaderboard_text(leaders: List[Dict]) -> str:
    """Format an Elo leaderboard as Markdown text for Telegram.

    Example output::

        *Team Elo Rankings*
        1. Australia — 1702.4 (updated: 2024-11-15)
        2. ...
    """
    if not leaders:
        return "_No Elo data available._"

    lines = ["*Team Elo Rankings*\n"]
    for entry in leaders:
        rank = entry.get("rank", "?")
        team = entry.get("team_name", "Unknown")
        elo = entry.get("elo_rating", 1500.0)
        updated = entry.get("last_updated") or "—"
        lines.append(
            f"{rank}. *{team}* — `{elo:.1f}` _(updated: {updated})_"
        )
    return "\n".join(lines)
