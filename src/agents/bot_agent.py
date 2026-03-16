"""
bot_agent.py — Thin wrapper that starts the Telegram bot within the agent framework.

Registers with orchestrator, then hands off to src/bot/main.py.
"""
from __future__ import annotations

import logging
import time

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("bot_agent")


class BotAgent:
    """Launches the Telegram bot and integrates with context/IPC."""

    def run(self) -> None:
        from src.agents.context_manager import context_manager
        from src.agents.orchestrator import ipc_write

        ipc_write(f"reg_bot_{int(time.time())}.json", {
            "agent": "bot_agent", "role": "register", "content": "bot_agent starting",
        })
        context_manager.add_message("bot_agent", "info", "bot_agent starting Telegram polling")
        logger.info("[bot_agent] Launching Telegram bot …")

        try:
            from src.bot.main import run_bot
            run_bot()
        except Exception as exc:
            logger.error("[bot_agent] Bot crashed: %s", exc)
            context_manager.add_message("bot_agent", "error", f"Bot crashed: {exc}")
