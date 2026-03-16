#!/usr/bin/env python3
"""
start.py — Local deployment entry point for the Cricket Analytics Bot.

Usage:
    # Full pipeline (first run or after long gap):
    python start.py --full

    # Bot only (after first run):
    python start.py --bot

    # Analytics + retrain only (run this nightly):
    python start.py --retrain

    # Historical backfill (run once, takes ~hours):
    python start.py --backfill --start-year 2022 --end-year 2024
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/agents.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("start")


def _ensure_dirs():
    for d in ("data/models", "data/context_archives", "data/llm_cache", "logs", "run/queue"):
        Path(d).mkdir(parents=True, exist_ok=True)


def _init():
    """Initialise DB tables and run Phase 2 migration."""
    from src.data.db import init_db
    from src.data.migrations.add_phase2_tables import run_migration
    init_db()
    run_migration()
    log.info("DB ready.")


def _seed_mock():
    """Seed mock data so the bot has something to work with immediately."""
    from src.data.db import Match, get_session
    from src.agents.ingestion_agent import _generate_mock_matches
    session = get_session()
    try:
        count = session.query(Match).count()
        if count < 50:
            log.info("Seeding %d mock matches for initial bot use...", 200 - count)
            for m in _generate_mock_matches(200):
                if not session.query(Match).filter_by(match_key=m["match_key"]).first():
                    session.add(Match(**{k: v for k, v in m.items() if k != "espn_match_id"}))
            session.commit()
            log.info("Mock seed complete.")
    finally:
        session.close()


def _train():
    """Train / retrain the prediction model."""
    from src.ml.train import train
    meta = train(force_synthetic=False)
    log.info(
        "Model v%s trained — accuracy=%.3f log_loss=%.4f n=%d",
        meta["model_version"], meta["accuracy"], meta.get("log_loss", 0), meta["n_samples"],
    )


def _analytics():
    """Run Elo, player ratings, team strengths, PVOR."""
    from src.data.db import get_session
    session = get_session()
    try:
        from src.analytics.elo import replay_all_elo
        from src.analytics.player_rating import run_all_ratings
        from src.analytics.team_strength import run_all_team_strengths
        from src.analytics.pvor import run_all_pvor

        for fmt in ("T20", "ODI", "Test"):
            replay_all_elo(session, fmt)
        log.info("Elo ratings updated.")

        n = run_all_ratings(session)
        log.info("Player ratings: %d players processed.", n)

        n = run_all_team_strengths(session)
        log.info("Team strengths: %d teams processed.", n)

        n = run_all_pvor(session)
        log.info("PVOR: %d matches processed.", n)
    except Exception as exc:
        log.error("Analytics failed: %s", exc)
    finally:
        session.close()


def _backfill(start_year: int, end_year: int):
    """Scrape ESPNcricinfo historical data."""
    import subprocess
    cmd = [
        sys.executable, "scripts/backfill_espn.py",
        "--start-year", str(start_year),
        "--end-year", str(end_year),
    ]
    log.info("Starting backfill %d–%d (this may take hours)...", start_year, end_year)
    subprocess.run(cmd, check=False)


def _run_bot():
    """Start the Telegram bot (blocking)."""
    from src.bot.main import run_bot
    log.info("Starting Telegram bot...")
    run_bot()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cricket Analytics Bot — local launcher")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--full",     action="store_true", help="Init DB → seed → analytics → train → start bot")
    group.add_argument("--bot",      action="store_true", help="Start bot only (DB must already be seeded)")
    group.add_argument("--retrain",  action="store_true", help="Analytics → retrain model (no bot)")
    group.add_argument("--backfill", action="store_true", help="Scrape ESPN historical data")
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year",   type=int, default=2024)
    args = parser.parse_args()

    _ensure_dirs()

    if args.full:
        _init()
        _seed_mock()       # gives bot data immediately while real scrape runs
        _analytics()
        _train()
        _run_bot()         # blocking — Ctrl+C to stop

    elif args.bot:
        _init()
        _run_bot()         # blocking

    elif args.retrain:
        _init()
        _analytics()
        _train()

    elif args.backfill:
        _init()
        _backfill(args.start_year, args.end_year)
        _analytics()
        _train()


if __name__ == "__main__":
    main()
