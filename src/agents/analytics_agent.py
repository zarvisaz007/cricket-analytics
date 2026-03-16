"""
analytics_agent.py — Runs the analytics pipeline.

Computes Elo ratings, player ratings, team strengths, and PVOR.
Registers with the orchestrator via IPC and signals completion.
"""
from __future__ import annotations

import logging
import time
from datetime import date

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("analytics_agent")


class AnalyticsAgent:
    """Runs Elo, player ratings, team strengths, and PVOR pipelines."""

    def run(self) -> None:
        from src.agents.context_manager import context_manager
        from src.agents.orchestrator import ipc_write
        from src.data.db import init_db

        # Register with orchestrator
        ipc_write(f"reg_analytics_{int(time.time())}.json", {
            "agent": "analytics_agent",
            "role": "register",
            "content": "analytics_agent started",
        })
        context_manager.add_message("analytics_agent", "info", "analytics_agent started")

        # Step 1: Init DB and run migration
        try:
            init_db()
            from src.data.migrations.add_phase2_tables import run_migration
            run_migration()
            context_manager.add_message("analytics_agent", "info", "DB migration complete")
        except Exception as exc:
            logger.error("[analytics] DB init/migration failed: %s", exc)
            context_manager.add_message("analytics_agent", "error", f"DB init/migration failed: {exc}")

        # Step 2: Elo replay
        try:
            logger.info("[analytics] Running Elo replay...")
            context_manager.add_message("analytics_agent", "info", "Running Elo replay...")
            from src.data.db import get_session
            from src.analytics.elo import replay_all_elo
            session = get_session()
            for fmt in ["T20", "ODI", "Test"]:
                replay_all_elo(session, fmt)
            session.close()
            logger.info("[analytics] Elo replay complete")
            context_manager.add_message("analytics_agent", "info", "Elo replay complete")
        except Exception as exc:
            logger.error("[analytics] Elo replay failed: %s", exc)
            context_manager.add_message("analytics_agent", "error", f"Elo replay failed: {exc}")

        # Step 3: Player ratings
        try:
            logger.info("[analytics] Computing player ratings...")
            context_manager.add_message("analytics_agent", "info", "Computing player ratings...")
            from src.analytics.player_rating import run_all_ratings
            run_all_ratings(date.today())
            logger.info("[analytics] Player ratings complete")
            context_manager.add_message("analytics_agent", "info", "Player ratings complete")
        except Exception as exc:
            logger.error("[analytics] Player ratings failed: %s", exc)
            context_manager.add_message("analytics_agent", "error", f"Player ratings failed: {exc}")

        # Step 4: Team strengths
        try:
            logger.info("[analytics] Computing team strengths...")
            context_manager.add_message("analytics_agent", "info", "Computing team strengths...")
            from src.analytics.team_strength import run_all_team_strengths
            run_all_team_strengths()
            logger.info("[analytics] Team strengths complete")
            context_manager.add_message("analytics_agent", "info", "Team strengths complete")
        except Exception as exc:
            logger.error("[analytics] Team strengths failed: %s", exc)
            context_manager.add_message("analytics_agent", "error", f"Team strengths failed: {exc}")

        # Step 5: PVOR
        try:
            logger.info("[analytics] Computing PVOR...")
            context_manager.add_message("analytics_agent", "info", "Computing PVOR...")
            from src.analytics.pvor import run_all_pvor
            run_all_pvor()
            logger.info("[analytics] PVOR complete")
            context_manager.add_message("analytics_agent", "info", "PVOR complete")
        except Exception as exc:
            logger.error("[analytics] PVOR failed: %s", exc)
            context_manager.add_message("analytics_agent", "error", f"PVOR failed: {exc}")

        # Signal done
        ipc_write(f"done_analytics_{int(time.time())}.json", {
            "agent": "analytics_agent",
            "role": "complete",
            "content": "Analytics pipeline complete",
        })
        logger.info("[analytics] Done.")
        context_manager.add_message("analytics_agent", "info", "analytics_agent done")

        # Purge check
        if context_manager.should_purge():
            context_manager.purge_and_archive("phase_analytics_compute", "token_threshold")
