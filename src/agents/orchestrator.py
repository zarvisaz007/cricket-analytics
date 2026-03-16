"""
orchestrator.py — Top-level pipeline orchestrator.

Phases
------
1. phase_ingestion_seed   — fetch / seed match data
2. phase_model_train      — train XGBoost model
3. phase_bot_ready        — launch Telegram bot

Usage
-----
    python -m src.agents.orchestrator              # run full pipeline
    python -m src.agents.orchestrator --phase-only phase_model_train
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup (must happen before any src imports so handlers attach)
# ---------------------------------------------------------------------------
LOG_FILE = Path(os.getenv("LOG_FILE", "./logs/agents.log"))
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("orchestrator")

# ---------------------------------------------------------------------------
# IPC helpers — file-based queue in run/queue/
# ---------------------------------------------------------------------------
QUEUE_DIR = Path("./run/queue")
QUEUE_DIR.mkdir(parents=True, exist_ok=True)
PHASE_STATUS_FILE = Path("./run/phase_status.json")


def ipc_write(filename: str, data: Dict[str, Any]) -> None:
    """
    Write *data* as JSON to ``run/queue/<filename>``.
    Uses a .lock file to prevent concurrent corruption.

    Parameters
    ----------
    filename : str
        Destination filename inside ``run/queue/``.
    data : dict
        JSON-serialisable payload.
    """
    from filelock import FileLock
    target = QUEUE_DIR / filename
    lock_path = target.with_suffix(".lock")
    with FileLock(str(lock_path), timeout=5):
        with target.open("w", encoding="utf-8") as f:
            json.dump(data, f)


def ipc_read_next(pattern: str = "*.json") -> Optional[Dict[str, Any]]:
    """
    Read and *delete* the oldest matching file from ``run/queue/``.

    Parameters
    ----------
    pattern : str
        Glob pattern for queue files.

    Returns
    -------
    dict or None
        Parsed JSON content, or None if queue is empty.
    """
    from filelock import FileLock
    files = sorted(QUEUE_DIR.glob(pattern))
    for target in files:
        lock_path = target.with_suffix(".lock")
        try:
            with FileLock(str(lock_path), timeout=2):
                data = json.loads(target.read_text(encoding="utf-8"))
                target.unlink(missing_ok=True)
                return data
        except Exception:
            continue
    return None


def _write_phase_status(phase: str, summary: Dict[str, Any]) -> None:
    """Persist phase completion status and pointers to run/phase_status.json."""
    try:
        existing: Dict = {}
        if PHASE_STATUS_FILE.exists():
            existing = json.loads(PHASE_STATUS_FILE.read_text())
        existing[phase] = {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "archive_path": summary.get("archive_path"),
            "summary_excerpt": summary.get("summary", "")[:200],
            "archival_failure": summary.get("archival_failure", False),
        }
        PHASE_STATUS_FILE.write_text(json.dumps(existing, indent=2))
    except Exception as exc:
        logger.error("[orchestrator] phase_status write failed: %s", exc)


# ---------------------------------------------------------------------------
# Agent process targets
# ---------------------------------------------------------------------------

def _run_ingestion_agent() -> None:
    from src.agents.ingestion_agent import IngestionAgent
    IngestionAgent().run()


def _run_ml_agent() -> None:
    from src.agents.ml_agent import MLAgent
    MLAgent().run()


def _run_nlp_agent() -> None:
    from src.agents.nlp_agent import NLPAgent
    NLPAgent().run()


def _run_bot_agent() -> None:
    from src.agents.bot_agent import BotAgent
    BotAgent().run()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Sequentially runs pipeline phases and manages sub-agent processes."""

    PHASES = [
        "phase_ingestion_seed",
        "phase_model_train",
        "phase_bot_ready",
    ]

    def __init__(self) -> None:
        from src.agents.context_manager import context_manager
        self.ctx = context_manager

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(self, phase_only: Optional[str] = None) -> None:
        """
        Run the full pipeline (or a single phase if *phase_only* is set).

        Parameters
        ----------
        phase_only : str, optional
            Run only this phase (for development).
        """
        phases = [phase_only] if phase_only else self.PHASES
        logger.info("[orchestrator] Starting — phases: %s", phases)
        self.ctx.add_message("orchestrator", "info", f"Starting phases: {phases}")

        for phase in phases:
            logger.info("[orchestrator] ═══ BEGIN %s ═══", phase)
            self.ctx.add_message("orchestrator", "info", f"Begin {phase}")
            try:
                self._run_phase(phase)
            except Exception as exc:
                logger.error("[orchestrator] Phase %s FAILED: %s", phase, exc)
                self.ctx.add_message("orchestrator", "error", f"Phase {phase} failed: {exc}")

            # Purge after every phase regardless of token count
            summary = self.ctx.purge_and_archive(phase, "phase_complete")
            _write_phase_status(phase, summary)
            logger.info("[orchestrator] ═══ END %s — summary written ═══", phase)

        logger.info("[orchestrator] All phases complete.")
        print("\nMVP ready — how would you like the first demo flow to behave?")

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _run_phase(self, phase: str) -> None:
        dispatch = {
            "phase_ingestion_seed": self._phase_ingestion,
            "phase_model_train": self._phase_model_train,
            "phase_bot_ready": self._phase_bot_ready,
        }
        handler = dispatch.get(phase)
        if handler is None:
            raise ValueError(f"Unknown phase: {phase}")
        handler()

    def _phase_ingestion(self) -> None:
        """Spawn ingestion agent, wait for completion signal."""
        p = multiprocessing.Process(target=_run_ingestion_agent, name="ingestion_agent")
        p.start()
        self.ctx.add_message("orchestrator", "info", "Spawned ingestion_agent")
        p.join(timeout=120)
        if p.is_alive():
            logger.warning("[orchestrator] ingestion_agent timed out — terminating")
            p.terminate()
        self._drain_queue()

    def _phase_model_train(self) -> None:
        """Spawn ML agent, wait for completion."""
        p = multiprocessing.Process(target=_run_ml_agent, name="ml_agent")
        p.start()
        self.ctx.add_message("orchestrator", "info", "Spawned ml_agent")
        p.join(timeout=180)
        if p.is_alive():
            logger.warning("[orchestrator] ml_agent timed out — terminating")
            p.terminate()
        self._drain_queue()

    def _phase_bot_ready(self) -> None:
        """Spawn bot agent (non-blocking — runs until KeyboardInterrupt)."""
        logger.info("[orchestrator] Launching bot_agent (detached) …")
        p = multiprocessing.Process(target=_run_bot_agent, name="bot_agent", daemon=True)
        p.start()
        self.ctx.add_message("orchestrator", "info", "Bot agent launched (PID %d)" % p.pid)
        # Keep orchestrator alive while bot runs
        logger.info("[orchestrator] Bot running. Press Ctrl+C to stop.")
        try:
            p.join()
        except KeyboardInterrupt:
            logger.info("[orchestrator] Received Ctrl+C — stopping bot.")
            p.terminate()

    def _drain_queue(self) -> None:
        """Read and log all pending IPC messages."""
        while True:
            msg = ipc_read_next()
            if msg is None:
                break
            logger.info("[orchestrator] IPC: %s", msg)
            self.ctx.add_message(
                msg.get("agent", "unknown"),
                msg.get("role", "info"),
                msg.get("content", str(msg)),
            )
            # Check purge after each IPC message
            if self.ctx.should_purge():
                logger.info("[orchestrator] Token threshold hit — mid-phase purge")
                summary = self.ctx.purge_and_archive("mid_phase", "token_threshold")
                _write_phase_status("mid_phase_purge", summary)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Cricket Analytics Orchestrator")
    parser.add_argument(
        "--phase-only",
        metavar="PHASE",
        help="Run a single phase (phase_ingestion_seed | phase_model_train | phase_bot_ready)",
    )
    args = parser.parse_args()

    # Initialise DB tables
    from src.data.db import init_db
    init_db()

    orch = Orchestrator()
    orch.run(phase_only=args.phase_only)


if __name__ == "__main__":
    main()
