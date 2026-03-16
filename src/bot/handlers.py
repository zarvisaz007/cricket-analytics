"""
handlers.py — Telegram bot command and callback handlers.

Menu structure
--------------
/start → main menu (inline keyboard):
  [Upcoming Matches]  [Predict]
  [Player Report]     [Settings]

Predict → select match → show probabilities + [Why?] button
Why?    → calls nlp_agent.explain_prediction (LLM or fallback)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("bot.handlers")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📅 Upcoming Matches", callback_data="matches_upcoming"),
            InlineKeyboardButton("🔮 Predict", callback_data="predict_list"),
        ],
        [
            InlineKeyboardButton("👤 Player Report", callback_data="player_list"),
            InlineKeyboardButton("⚙️ Settings", callback_data="settings_main"),
        ],
    ])


def _back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("⬅️ Back", callback_data="matches_upcoming"),
    ]])


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_upcoming_matches(limit: int = 8):
    from src.data.db import Match, get_session
    session = get_session()
    try:
        # Matches with no winner yet (upcoming) or most recent
        from sqlalchemy import or_
        rows = (
            session.query(Match)
            .filter(or_(Match.winner.is_(None), Match.winner == ""))
            .order_by(Match.match_date.desc())
            .limit(limit)
            .all()
        )
        if not rows:
            # Fall back to most recent historical matches
            rows = (
                session.query(Match)
                .order_by(Match.match_date.desc())
                .limit(limit)
                .all()
            )
        return rows
    finally:
        session.close()


def _get_match_by_id(match_id: int) -> Optional[object]:
    from src.data.db import Match, get_session
    session = get_session()
    try:
        return session.query(Match).get(match_id)
    finally:
        session.close()


def _ensure_user(telegram_id: int) -> None:
    """Upsert a TelegramUser record (stores only telegram_id — no PII)."""
    from src.data.db import TelegramUser, get_session
    session = get_session()
    try:
        user = session.query(TelegramUser).filter_by(telegram_id=telegram_id).first()
        if not user:
            session.add(TelegramUser(telegram_id=telegram_id))
            session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — show main menu."""
    user_id = update.effective_user.id
    _ensure_user(user_id)
    from src.agents.context_manager import context_manager
    context_manager.add_message("bot_agent", "event", f"User {user_id} started bot")

    await update.message.reply_text(
        "🏏 *Cricket Analytics Bot*\n\nChoose an option:",
        reply_markup=_main_keyboard(),
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help."""
    text = (
        "🏏 *Cricket Analytics Bot Help*\n\n"
        "• *Upcoming Matches* — browse scheduled games\n"
        "• *Predict* — get win probability for a match\n"
        "• *Player Report* — AI-generated player summary\n"
        "• *Settings* — notification preferences\n\n"
        "Type /start to open the main menu."
    )
    await update.message.reply_text(text, parse_mode="Markdown")


# ---------------------------------------------------------------------------
# Callback handlers
# ---------------------------------------------------------------------------

async def handle_menu_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generic menu re-display."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("🏏 Main Menu:", reply_markup=_main_keyboard())


async def handle_matches_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show upcoming/recent matches list."""
    query = update.callback_query
    await query.answer()

    matches = _get_upcoming_matches(8)
    if not matches:
        await query.edit_message_text("No matches found. Run setup to seed data.")
        return

    buttons = []
    for m in matches:
        label = f"{m.team_a} vs {m.team_b} ({m.match_date or '?'})"
        buttons.append([InlineKeyboardButton(label, callback_data=f"predict_match_{m.id}")])
    buttons.append([InlineKeyboardButton("⬅️ Back", callback_data="menu_main")])

    await query.edit_message_text(
        "📅 *Recent / Upcoming Matches*\nTap a match to predict:",
        reply_markup=InlineKeyboardMarkup(buttons),
        parse_mode="Markdown",
    )


async def handle_predict_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle predict_list or predict_match_<id>."""
    query = update.callback_query
    await query.answer()
    data = query.data  # e.g. "predict_list" or "predict_match_42"

    if data == "predict_list":
        # Reuse matches list view
        await handle_matches_callback(update, context)
        return

    # predict_match_<id>
    parts = data.split("_")
    if len(parts) < 3:
        await query.edit_message_text("Invalid selection.")
        return

    match_id = int(parts[-1])
    match = _get_match_by_id(match_id)
    if not match:
        await query.edit_message_text("Match not found.")
        return

    # Run prediction
    from src.ml.train import predict_match
    from src.agents.context_manager import context_manager

    pred = predict_match(
        team_a=match.team_a,
        team_b=match.team_b,
        match_type=match.match_type or "T20",
        toss_winner=match.toss_winner or "",
        toss_decision=match.toss_decision or "bat",
    )
    context_manager.add_message(
        "bot_agent", "event",
        f"user requested prediction: match_id {match_id} "
        f"{match.team_a} vs {match.team_b}"
    )

    # Save prediction to DB
    _save_prediction(match_id, pred)

    ta_pct = f"{pred['team_a_win_prob'] * 100:.1f}%"
    tb_pct = f"{pred['team_b_win_prob'] * 100:.1f}%"
    winner = pred["predicted_winner"]

    text = (
        f"🔮 *Prediction*\n\n"
        f"*{match.team_a}* {ta_pct}\n"
        f"*{match.team_b}* {tb_pct}\n\n"
        f"🏆 Predicted winner: *{winner}* ({pred['confidence'] * 100:.0f}% confidence)\n"
        f"Match type: {match.match_type} | {match.venue or 'Venue TBD'}"
    )
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("❓ Why?", callback_data=f"why_{match_id}")],
        [InlineKeyboardButton("⬅️ Back", callback_data="matches_upcoming")],
    ])
    await query.edit_message_text(text, reply_markup=keyboard, parse_mode="Markdown")


async def handle_why_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Explain a prediction using NLP agent."""
    query = update.callback_query
    await query.answer("Generating explanation…")

    match_id = int(query.data.split("_")[1])
    match = _get_match_by_id(match_id)
    if not match:
        await query.edit_message_text("Match not found.")
        return

    from src.ml.train import predict_match
    from src.agents.nlp_agent import NLPAgent

    pred = predict_match(match.team_a, match.team_b, match.match_type or "T20")
    match_info = {
        "team_a": match.team_a, "team_b": match.team_b,
        "venue": match.venue, "match_type": match.match_type,
    }
    explanation = NLPAgent().explain_prediction(match_info, pred)

    await query.edit_message_text(
        f"❓ *Why {pred['predicted_winner']}?*\n\n{explanation}",
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton("⬅️ Back", callback_data=f"predict_match_{match_id}"),
        ]]),
        parse_mode="Markdown",
    )


async def handle_player_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show player list and generate report."""
    query = update.callback_query
    await query.answer()
    data = query.data  # player_list | player_<id>

    if data == "player_list":
        from src.data.db import Player, get_session
        session = get_session()
        players = session.query(Player).limit(10).all()
        session.close()

        if not players:
            await query.edit_message_text(
                "No players in DB yet. Run the bot after seeding data.",
                reply_markup=_back_keyboard(),
            )
            return

        buttons = [
            [InlineKeyboardButton(p.name, callback_data=f"player_{p.id}")]
            for p in players
        ]
        buttons.append([InlineKeyboardButton("⬅️ Back", callback_data="menu_main")])
        await query.edit_message_text(
            "👤 Select a player:",
            reply_markup=InlineKeyboardMarkup(buttons),
        )
        return

    # player_<id>
    player_id = int(data.split("_")[1])
    from src.data.db import Player, PlayerStat, get_session
    session = get_session()
    player = session.query(Player).get(player_id)
    stats_rows = session.query(PlayerStat).filter_by(player_id=player_id).limit(20).all()
    session.close()

    if not player:
        await query.edit_message_text("Player not found.")
        return

    # Aggregate
    total_runs = sum(s.runs for s in stats_rows)
    total_balls = sum(s.balls_faced for s in stats_rows) or 1
    total_wkts = sum(s.wickets for s in stats_rows)
    agg_stats = {
        "matches": len(stats_rows),
        "total_runs": total_runs,
        "batting_avg": round(total_runs / max(len(stats_rows), 1), 2),
        "strike_rate": round(total_runs / total_balls * 100, 2),
        "wickets": total_wkts,
    }

    from src.agents.nlp_agent import NLPAgent
    report = NLPAgent().generate_report(player.name, agg_stats)

    await query.edit_message_text(
        f"👤 *{player.name}* ({player.role or 'N/A'})\n\n{report}",
        reply_markup=_back_keyboard(),
        parse_mode="Markdown",
    )


async def handle_settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show settings menu."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        "⚙️ *Settings*\n\nNotifications: ON (tap to toggle — TODO)\n"
        "Language: English",
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton("⬅️ Back", callback_data="menu_main"),
        ]]),
        parse_mode="Markdown",
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle free text — redirect to main menu."""
    await update.message.reply_text(
        "Use /start to open the menu.",
        reply_markup=_main_keyboard(),
    )


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _save_prediction(match_id: int, pred: dict) -> None:
    import json
    from src.data.db import Prediction, get_session
    session = get_session()
    try:
        p = Prediction(
            match_id=match_id,
            model_version=pred.get("model_version", "1.0.0"),
            team_a_win_prob=pred.get("team_a_win_prob"),
            team_b_win_prob=pred.get("team_b_win_prob"),
            predicted_winner=pred.get("predicted_winner"),
            confidence=pred.get("confidence"),
            key_features_json=json.dumps(pred.get("key_features", {})),
        )
        session.add(p)
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.warning("[handlers] prediction save failed: %s", exc)
    finally:
        session.close()
