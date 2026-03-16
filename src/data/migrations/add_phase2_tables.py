"""
add_phase2_tables.py — Idempotent migration for Phase 2 schema additions.

Safe to re-run: uses CREATE TABLE IF NOT EXISTS for new tables and
ALTER TABLE ADD COLUMN for new columns on existing tables.

Usage:
    python -m src.data.migrations.add_phase2_tables
"""
from __future__ import annotations

import logging

from sqlalchemy import text

logger = logging.getLogger(__name__)


def run_migration() -> None:
    from src.data.db import Base, engine, init_db

    # Create all new tables (IF NOT EXISTS is implicit via create_all)
    init_db()
    logger.info("[migration] All new tables created/verified.")

    # Add new columns to existing tables using ALTER TABLE
    # SQLite 3.37+ supports ADD COLUMN IF NOT EXISTS
    new_columns = [
        # players
        ("players", "espn_id",      "INTEGER"),
        ("players", "cricbuzz_id",  "INTEGER"),
        # matches
        ("matches", "venue_id",          "INTEGER"),
        ("matches", "tournament_id",     "INTEGER"),
        ("matches", "cricbuzz_match_id", "INTEGER"),
        ("matches", "innings_complete",  "INTEGER DEFAULT 0"),
        # player_stats
        ("player_stats", "innings_id",        "INTEGER"),
        ("player_stats", "stumpings",         "INTEGER DEFAULT 0"),
        ("player_stats", "not_out",           "INTEGER DEFAULT 0"),
        ("player_stats", "batting_position",  "INTEGER"),
        ("player_stats", "bowling_slot",      "INTEGER"),
        # predictions
        ("predictions", "sim_win_prob_a",  "REAL"),
        ("predictions", "sim_score_p50_a", "REAL"),
        ("predictions", "sim_score_p50_b", "REAL"),
    ]

    with engine.connect() as conn:
        # Build a map of existing columns per table to avoid duplicate-column errors
        # on SQLite versions that don't support ADD COLUMN IF NOT EXISTS
        existing: dict = {}
        for table, _, _ in new_columns:
            if table not in existing:
                try:
                    result = conn.execute(text(f"PRAGMA table_info({table})"))
                    existing[table] = {row[1] for row in result}
                except Exception:
                    existing[table] = set()

        for table, column, col_type in new_columns:
            if column in existing.get(table, set()):
                logger.debug("[migration] Column %s.%s already exists — skipped", table, column)
                continue
            try:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
                conn.commit()
                logger.info("[migration] Added column %s.%s", table, column)
            except Exception as exc:
                err = str(exc).lower()
                if "duplicate column" in err or "already exists" in err:
                    logger.debug("[migration] Column %s.%s already exists — skipped", table, column)
                else:
                    logger.warning("[migration] %s.%s: %s", table, column, exc)

    logger.info("[migration] Phase 2 migration complete.")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run_migration()
    print("Migration complete.")
