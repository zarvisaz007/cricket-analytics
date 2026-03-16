"""
ingestion_agent.py — Fetches and seeds cricket match data into the DB.

Providers
---------
* mock   — generates synthetic historical data (default, no API key needed)
* cricapi — TODO: swap in real CricAPI key (CRICKET_API_KEY env var)
* espn_scrape — TODO: implement ESPNCricinfo scraper

The agent registers itself with the orchestrator via IPC and calls
context_manager.add_message() for significant events.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ingestion_agent")

PROVIDER = os.getenv("CRICKET_API_PROVIDER", "mock")
# TODO: Replace with real CricAPI endpoint when CRICKET_API_KEY is set
CRICKET_API_KEY = os.getenv("CRICKET_API_KEY", "")


# ---------------------------------------------------------------------------
# Mock data generator — realistic synthetic matches (2020–2026)
# ---------------------------------------------------------------------------

TEAMS = [
    "India", "Australia", "England", "Pakistan", "South Africa",
    "New Zealand", "West Indies", "Sri Lanka", "Bangladesh", "Afghanistan",
]
VENUES = [
    "Eden Gardens, Kolkata", "MCG, Melbourne", "Lord's, London",
    "Gaddafi Stadium, Lahore", "Newlands, Cape Town", "Eden Park, Auckland",
    "Kensington Oval, Bridgetown", "SSC, Colombo", "Shere Bangla, Dhaka",
    "Sharjah Cricket Stadium",
]
TOURNAMENTS = [
    "ICC T20 World Cup", "ICC Cricket World Cup", "ICC Champions Trophy",
    "ICC World Test Championship", "Asia Cup", "Bilateral Series",
]
MATCH_TYPES = ["T20", "ODI", "Test"]


def _random_date(start_year: int = 2020, end_year: int = 2026) -> str:
    start = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    end = datetime(end_year, 3, 16, tzinfo=timezone.utc)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).strftime("%Y-%m-%d")


def _generate_mock_matches(n: int = 200) -> List[Dict[str, Any]]:
    """Generate *n* synthetic historical matches."""
    random.seed(0)
    matches = []
    for i in range(n):
        ta = random.choice(TEAMS)
        tb = random.choice([t for t in TEAMS if t != ta])
        winner = random.choice([ta, tb])
        mt = random.choice(MATCH_TYPES)
        margin = (
            f"{random.randint(1,200)} runs" if random.random() > 0.5
            else f"{random.randint(1,10)} wickets"
        )
        toss_w = random.choice([ta, tb])
        matches.append({
            "match_key": f"mock_{i:04d}",
            "team_a": ta,
            "team_b": tb,
            "venue": random.choice(VENUES),
            "match_date": _random_date(),
            "match_type": mt,
            "tournament": random.choice(TOURNAMENTS),
            "winner": winner,
            "result_margin": margin,
            "toss_winner": toss_w,
            "toss_decision": random.choice(["bat", "field"]),
            "source": "mock",
        })
    return matches


# ---------------------------------------------------------------------------
# Real API stubs — implement when keys are available
# ---------------------------------------------------------------------------

def _fetch_cricapi(n: int = 50) -> List[Dict]:
    """
    TODO: Implement real CricAPI fetch.
    Endpoint: https://api.cricapi.com/v1/matches?apikey={CRICKET_API_KEY}
    """
    logger.warning("[ingestion] cricapi not implemented — falling back to mock")
    return _generate_mock_matches(n)


def _fetch_espn_scrape(n: int = 50) -> List[Dict]:
    """Fetch real match data from ESPNcricinfo via web scraping."""
    try:
        from src.scrapers.espn_historical import discover_matches
        from src.scrapers.espn_scorecard import parse_scorecard_to_db
        from src.data.db import get_session, init_db

        init_db()
        session = get_session()

        # Discover and scrape recent matches (last 2 years for regular runs)
        current_year = datetime.now().year
        scraped = []

        # Use discover_matches to get match metadata (lightweight)
        for match_data in discover_matches(
            start_year=current_year - 1,
            end_year=current_year,
            formats=["twenty20-internationals", "one-day-internationals"],
        ):
            # parse_scorecard_to_db handles full upsert
            espn_id = match_data.get("espn_match_id")
            if espn_id:
                parse_scorecard_to_db(int(espn_id), session)
            scraped.append(match_data)
            if len(scraped) >= n:
                break

        session.close()
        logger.info("[ingestion] ESPN scrape complete — %d matches processed", len(scraped))

        # Return basic match dicts for the DB upsert in IngestionAgent.run()
        return scraped if scraped else _generate_mock_matches(n)
    except Exception as exc:
        logger.warning("[ingestion] ESPN scrape failed: %s — falling back to mock", exc)
        return _generate_mock_matches(n)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class IngestionAgent:
    """Fetches match data and upserts it into the database."""

    def run(self) -> None:
        from src.agents.context_manager import context_manager
        from src.agents.orchestrator import ipc_write
        from src.data.db import Match, get_session, init_db

        init_db()
        # Register with orchestrator
        ipc_write(f"reg_ingestion_{int(time.time())}.json", {
            "agent": "ingestion_agent",
            "role": "register",
            "content": "ingestion_agent started",
        })
        context_manager.add_message("ingestion_agent", "info", "ingestion_agent started")

        # Fetch data
        if PROVIDER == "cricapi" and CRICKET_API_KEY:
            matches = _fetch_cricapi(200)
        elif PROVIDER == "espn_scrape":
            matches = _fetch_espn_scrape(200)
        else:
            matches = _generate_mock_matches(200)

        context_manager.add_message(
            "ingestion_agent", "info",
            f"Fetched {len(matches)} matches from provider='{PROVIDER}'"
        )

        # Upsert into DB
        session = get_session()
        new_count = 0
        try:
            for m in matches:
                existing = session.query(Match).filter_by(match_key=m["match_key"]).first()
                if existing:
                    # Update if needed
                    existing.winner = m["winner"]
                    existing.updated_at = datetime.now(timezone.utc).isoformat()
                else:
                    session.add(Match(**m))
                    new_count += 1
            session.commit()
            context_manager.add_message(
                "ingestion_agent", "info",
                f"Upserted {len(matches)} matches — {new_count} new into DB"
            )
            logger.info("[ingestion] %d new matches saved", new_count)
        except Exception as exc:
            session.rollback()
            logger.error("[ingestion] DB error: %s", exc)
            context_manager.add_message("ingestion_agent", "error", f"DB error: {exc}")
        finally:
            session.close()

        # Periodic purge check
        if context_manager.should_purge():
            context_manager.purge_and_archive("ingestion_mid", "token_threshold")

        ipc_write(f"done_ingestion_{int(time.time())}.json", {
            "agent": "ingestion_agent",
            "role": "complete",
            "content": f"Ingestion complete — {new_count} new matches",
        })
        logger.info("[ingestion] Done.")
