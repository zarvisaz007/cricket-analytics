#!/usr/bin/env python3
"""
run.py — Single-file launcher for the Cricket Analytics Bot.

Just run:
    .venv/bin/python3 run.py

Everything runs in one terminal:
  • DB initialisation + migration
  • Seeds data (mock immediately, real ESPN scrape in background)
  • Analytics pipeline (Elo, player ratings, team strengths, PVOR)
  • XGBoost model training
  • Telegram bot (main thread — keeps terminal alive)
  • Live match poller (background thread)
  • Nightly auto-retrain at 04:00 every day (background thread)

Press Ctrl+C to stop everything cleanly.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# ── project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

# ── logging ───────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/agents.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("run")

# ── shared stop event (Ctrl+C sets this) ─────────────────────────────────────
_stop = threading.Event()


# ─────────────────────────────────────────────────────────────────────────────
# 1. INITIALISE
# ─────────────────────────────────────────────────────────────────────────────

def step_init():
    log.info("═══ STEP 1/5  Init DB & migration ═══")
    for d in ("data/models", "data/context_archives", "data/llm_cache",
              "logs", "run/queue"):
        Path(d).mkdir(parents=True, exist_ok=True)
    from src.data.db import init_db
    from src.data.migrations.add_phase2_tables import run_migration
    init_db()
    run_migration()
    log.info("DB ready.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEED MOCK DATA  (instant — so bot works from the first second)
# ─────────────────────────────────────────────────────────────────────────────

def step_seed_mock():
    log.info("═══ STEP 2/5  Seeding mock data (instant fallback) ═══")
    from src.data.db import Match, get_session
    from src.agents.ingestion_agent import _generate_mock_matches
    session = get_session()
    try:
        existing = session.query(Match).count()
        if existing >= 50:
            log.info("DB already has %d matches — skipping mock seed.", existing)
            return
        for m in _generate_mock_matches(200):
            row = {k: v for k, v in m.items() if k != "espn_match_id"}
            if not session.query(Match).filter_by(match_key=row["match_key"]).first():
                session.add(Match(**row))
        session.commit()
        log.info("200 mock matches seeded.")
    except Exception as exc:
        session.rollback()
        log.warning("Mock seed failed: %s", exc)
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. ANALYTICS  (Elo → ratings → team strengths → PVOR)
# ─────────────────────────────────────────────────────────────────────────────

def step_analytics():
    log.info("═══ STEP 3/5  Analytics pipeline ═══")
    from src.data.db import get_session
    session = get_session()
    try:
        from src.analytics.elo import replay_all_elo
        from src.analytics.player_rating import run_all_ratings
        from src.analytics.team_strength import run_all_team_strengths
        from src.analytics.pvor import run_all_pvor

        for fmt in ("T20", "ODI", "Test"):
            replay_all_elo(session, fmt)
        log.info("Elo ratings computed.")

        n = run_all_ratings(session)
        log.info("Player ratings: %d players.", n)

        n = run_all_team_strengths(session)
        log.info("Team strengths: %d entries.", n)

        run_all_pvor(session)
        log.info("PVOR computed.")
    except Exception as exc:
        log.error("Analytics error: %s", exc)
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────

def step_train():
    log.info("═══ STEP 4/5  Training XGBoost model ═══")
    try:
        from src.ml.train import train
        meta = train(force_synthetic=False)
        log.info(
            "Model v%s ready — accuracy=%.3f  log_loss=%.4f  n=%d",
            meta["model_version"], meta["accuracy"],
            meta.get("log_loss", 0), meta["n_samples"],
        )
    except Exception as exc:
        log.error("Training error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND THREAD A — ESPN historical backfill
# Runs silently in the background while the bot is already live.
# ─────────────────────────────────────────────────────────────────────────────

def _bg_backfill():
    """Download real ball-by-ball data from cricsheet.org (free, no scraping needed)."""
    log.info("[backfill] Downloading real match data from cricsheet.org…")
    try:
        from src.scrapers.cricsheet import ingest_to_db
        # T20I + ODI first (faster), then Test + IPL
        count = ingest_to_db(formats=["t20i", "odi", "test", "ipl"])
        log.info("[backfill] Done — %d new matches ingested.", count)
        if count > 0 and not _stop.is_set():
            log.info("[backfill] Re-running analytics on real data…")
            step_analytics()
            step_train()
            log.info("[backfill] Model retrained on real data.")
    except Exception as exc:
        log.error("[backfill] Thread error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND THREAD B — Live match poller (Cricbuzz, every 30 s)
# ─────────────────────────────────────────────────────────────────────────────

LIVE_POLL_INTERVAL  = int(os.getenv("LIVE_POLL_INTERVAL_SECONDS",  "30"))
LIVE_CHECK_INTERVAL = int(os.getenv("LIVE_CHECK_INTERVAL_SECONDS", "300"))


def _bg_live_poll():
    log.info("[live] Live match poller started.")
    from src.data.db import Match, get_session
    while not _stop.is_set():
        try:
            session = get_session()
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            live = (
                session.query(Match)
                .filter(Match.match_date == today, Match.winner.is_(None))
                .all()
            )
            session.close()

            if live:
                log.info("[live] %d live match(es) detected.", len(live))
                from src.scrapers.cricbuzz_live import poll_live_match
                for m in live:
                    if _stop.is_set():
                        break
                    if m.cricbuzz_match_id:
                        try:
                            session2 = get_session()
                            new_balls = poll_live_match(m.cricbuzz_match_id, m.id, session2)
                            session2.close()
                            if new_balls:
                                log.info("[live] match %d — %d new deliveries.", m.id, new_balls)
                        except Exception as exc:
                            log.debug("[live] poll error for match %d: %s", m.id, exc)
                _stop.wait(LIVE_POLL_INTERVAL)
            else:
                _stop.wait(LIVE_CHECK_INTERVAL)
        except Exception as exc:
            log.error("[live] Thread error: %s", exc)
            _stop.wait(60)


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND THREAD C — Nightly retrain at 04:00 every day
# ─────────────────────────────────────────────────────────────────────────────

def _bg_nightly_retrain():
    log.info("[retrain] Nightly retrain scheduler started (fires at 04:00 daily).")
    while not _stop.is_set():
        now = datetime.now()
        # seconds until next 04:00
        target = now.replace(hour=4, minute=0, second=0, microsecond=0)
        if now >= target:
            target = target.replace(day=target.day + 1)
        wait_sec = (target - now).total_seconds()
        log.info("[retrain] Next retrain in %.1f hours.", wait_sec / 3600)

        # sleep in 60-second chunks so _stop can interrupt
        slept = 0
        while slept < wait_sec and not _stop.is_set():
            time.sleep(min(60, wait_sec - slept))
            slept += 60

        if _stop.is_set():
            break

        log.info("[retrain] Starting nightly analytics + retrain…")
        try:
            step_analytics()
            step_train()
            log.info("[retrain] Nightly retrain complete.")
        except Exception as exc:
            log.error("[retrain] Nightly retrain error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# 5. BOT  (main thread — blocks until Ctrl+C)
# ─────────────────────────────────────────────────────────────────────────────

def step_run_bot():
    log.info("═══ STEP 5/5  Starting Telegram bot ═══")
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        log.error("TELEGRAM_BOT_TOKEN not set in .env — cannot start bot.")
        sys.exit(1)
    from src.bot.main import run_bot
    run_bot()   # blocks here until Ctrl+C


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _kill_old_instances():
    """Kill any previously running cricket bot processes before starting."""
    import os
    import signal
    import subprocess
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ["pgrep", "-f", "src.bot.main"],
            capture_output=True, text=True
        )
        pids = [int(p) for p in result.stdout.split() if p.strip() and int(p) != my_pid]
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                log.info("Killed old bot instance (PID %d).", pid)
            except ProcessLookupError:
                pass
        if pids:
            time.sleep(2)  # give it time to die cleanly
    except Exception:
        pass  # pgrep not available or other error — continue anyway


def main():
    print("""
╔══════════════════════════════════════════════╗
║   🏏  Cricket Analytics Bot — starting up   ║
╚══════════════════════════════════════════════╝
""")
    _kill_old_instances()

    # ── Sequential startup ────────────────────────────────────────────────────
    step_init()
    step_seed_mock()
    step_analytics()
    step_train()

    # ── Launch background threads ─────────────────────────────────────────────
    threads = [
        threading.Thread(target=_bg_backfill,       name="backfill",  daemon=True),
        threading.Thread(target=_bg_live_poll,       name="live_poll", daemon=True),
        threading.Thread(target=_bg_nightly_retrain, name="retrain",   daemon=True),
    ]
    for t in threads:
        t.start()

    log.info("Background threads started: backfill, live_poll, nightly_retrain")

    # ── Start bot (blocks) ────────────────────────────────────────────────────
    try:
        step_run_bot()
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Shutting down — signalling threads to stop…")
        _stop.set()
        for t in threads:
            t.join(timeout=5)
        log.info("Goodbye.")


if __name__ == "__main__":
    main()
