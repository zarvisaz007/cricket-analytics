"""
main.py — Telegram bot entry point.

Menus: Upcoming Matches | Predict | Player Report | Settings
Inline keyboards with concise messages.
"""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

load_dotenv()

logger = logging.getLogger("bot.main")

TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")


def run_bot() -> None:
    """Build and start the Telegram bot (blocking)."""
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in environment")

    from src.bot.handlers import (
        cmd_start,
        cmd_help,
        handle_menu_button,
        handle_predict_callback,
        handle_why_callback,
        handle_matches_callback,
        handle_player_callback,
        handle_settings_callback,
        handle_leaderboard_callback,
        handle_simulate_callback,
        handle_h2h_callback,
        handle_text,
    )

    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))

    # Inline keyboard callbacks (order matters — more specific patterns first)
    app.add_handler(CallbackQueryHandler(handle_matches_callback, pattern="^matches"))
    app.add_handler(CallbackQueryHandler(handle_simulate_callback, pattern="^simulate"))
    app.add_handler(CallbackQueryHandler(handle_predict_callback, pattern="^predict"))
    app.add_handler(CallbackQueryHandler(handle_why_callback, pattern="^why_"))
    app.add_handler(CallbackQueryHandler(handle_h2h_callback, pattern="^h2h_"))
    app.add_handler(CallbackQueryHandler(handle_leaderboard_callback, pattern="^leaderboard"))
    app.add_handler(CallbackQueryHandler(handle_player_callback, pattern="^player"))
    app.add_handler(CallbackQueryHandler(handle_settings_callback, pattern="^settings"))
    app.add_handler(CallbackQueryHandler(handle_menu_button, pattern="^menu_"))

    # Free text
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("[bot] Starting polling …")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_bot()
