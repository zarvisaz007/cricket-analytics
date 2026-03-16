"""
live_agent.py — Continuous live match monitor.

Polls Cricbuzz for live match scores during active matches.
Runs as a daemon process spawned by the orchestrator.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("live_agent")

LIVE_POLL_INTERVAL = int(os.getenv("LIVE_POLL_INTERVAL_SECONDS", "30"))
LIVE_CHECK_INTERVAL = int(os.getenv("LIVE_CHECK_INTERVAL_SECONDS", "300"))


class LiveAgent:
    """Monitors live cricket matches and pushes IPC updates."""

    def run(self) -> None:
        from src.agents.context_manager import context_manager
        from src.agents.orchestrator import ipc_write

        # Register with orchestrator
        ipc_write(f"reg_live_{int(time.time())}.json", {
            "agent": "live_agent",
            "role": "register",
            "content": "live_agent started",
        })
        context_manager.add_message("live_agent", "info", "live_agent started")
        logger.info("[live_agent] Started — poll_interval=%ds check_interval=%ds",
                    LIVE_POLL_INTERVAL, LIVE_CHECK_INTERVAL)

        while True:
            try:
                live_matches = self._get_live_matches()

                if not live_matches:
                    logger.debug("[live_agent] No live matches — sleeping %ds", LIVE_CHECK_INTERVAL)
                    time.sleep(LIVE_CHECK_INTERVAL)
                    continue

                for match in live_matches:
                    try:
                        new_data = self._poll_live_match(match)
                        if new_data:
                            ipc_write(f"live_update_{match.id}_{int(time.time())}.json", {
                                "agent": "live_agent",
                                "role": "live_update",
                                "content": f"Live update for match {match.id}: {match.team_a} vs {match.team_b}",
                                "match_id": match.id,
                                "data": new_data,
                            })
                            context_manager.add_message(
                                "live_agent", "info",
                                f"Live update for match {match.id}: {match.team_a} vs {match.team_b}"
                            )
                    except Exception as exc:
                        logger.error("[live_agent] Failed to poll match %s: %s", match.id, exc)

                time.sleep(LIVE_POLL_INTERVAL)

            except Exception as exc:
                logger.error("[live_agent] Poll loop error: %s", exc)
                time.sleep(LIVE_POLL_INTERVAL)

    def _get_live_matches(self):
        """Query DB for matches where match_date = today AND winner IS NULL."""
        try:
            from src.data.db import Match, get_session
            session = get_session()
            today = date.today().isoformat()
            matches = (
                session.query(Match)
                .filter(Match.match_date == today, Match.winner.is_(None))
                .all()
            )
            session.close()
            return matches
        except Exception as exc:
            logger.error("[live_agent] DB query failed: %s", exc)
            return []

    def _poll_live_match(self, match) -> dict | None:
        """Poll Cricbuzz for live score data for a match with cricbuzz_match_id."""
        cricbuzz_id = getattr(match, "cricbuzz_match_id", None)
        if not cricbuzz_id:
            return None

        try:
            from src.scrapers.cricbuzz_live import fetch_live_score
            data = fetch_live_score(cricbuzz_id)
            return data
        except ImportError:
            logger.debug("[live_agent] cricbuzz_live scraper not available")
            return None
        except Exception as exc:
            logger.warning("[live_agent] Cricbuzz poll failed for match %s: %s", match.id, exc)
            return None
