"""
ml_agent.py — Trains the XGBoost match-prediction model.

Registers with the orchestrator via IPC, trains the model (or re-trains
if a newer DB snapshot is available), and signals completion.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("ml_agent")


class MLAgent:
    """Wraps model training and wires into the context/IPC system."""

    def run(self) -> None:
        from src.agents.context_manager import context_manager
        from src.agents.orchestrator import ipc_write
        from src.ml.train import train

        # Register
        ipc_write(f"reg_ml_{int(time.time())}.json", {
            "agent": "ml_agent",
            "role": "register",
            "content": "ml_agent started",
        })
        context_manager.add_message("ml_agent", "info", "ml_agent started — beginning training")

        try:
            meta = train(force_synthetic=False)
            msg = (
                f"trained model v{meta['model_version']} "
                f"accuracy={meta['accuracy']:.3f} "
                f"samples={meta['n_samples']}"
            )
            context_manager.add_message("ml_agent", "info", msg)
            logger.info("[ml_agent] %s", msg)

            ipc_write(f"done_ml_{int(time.time())}.json", {
                "agent": "ml_agent",
                "role": "complete",
                "content": msg,
            })
        except Exception as exc:
            logger.error("[ml_agent] Training failed: %s", exc)
            context_manager.add_message("ml_agent", "error", f"Training failed: {exc}")

        # Purge check
        if context_manager.should_purge():
            context_manager.purge_and_archive("phase_model_train", "token_threshold")
