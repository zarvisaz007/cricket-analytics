"""
llm_client.py — Thin wrapper around OpenRouter for LLM calls.

Capabilities
------------
* summarize(messages, phase) → str
* generate_report(player_name, stats) → str
* explain_prediction(match_info, prediction) → str

All results are cached in ./data/llm_cache/ keyed by SHA-256 of inputs.
If OPENROUTER_API_KEY is missing, deterministic fallback strings are returned
and stored in cache so behaviour stays consistent.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CACHE_DIR = Path("./data/llm_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

MAX_RETRIES = 3
RETRY_BASE = 2.0  # seconds


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> Optional[str]:
    path = CACHE_DIR / f"{key}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _cache_set(key: str, value: str) -> None:
    (CACHE_DIR / f"{key}.txt").write_text(value, encoding="utf-8")


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """Wrapper for OpenRouter chat-completion calls with caching and retries."""

    def __init__(self) -> None:
        self._api_key = API_KEY
        self._model = LLM_MODEL

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def summarize(self, messages: List[Dict], phase_name: str) -> str:
        """
        Produce a concise summary of *messages* for *phase_name*.

        Parameters
        ----------
        messages : list
            Ephemeral context message dicts.
        phase_name : str
            Pipeline phase label for context.

        Returns
        -------
        str
            Summary text (LLM or fallback).
        """
        key = _cache_key({"op": "summarize", "phase": phase_name, "msgs": messages[:50]})
        cached = _cache_get(key)
        if cached is not None:
            return cached

        system_prompt = (
            "You are a concise technical summariser for a cricket analytics pipeline. "
            "Given a list of agent log messages from one pipeline phase, produce a "
            "short (≤120 word) structured summary covering: what happened, key numbers, "
            "and any errors."
        )
        user_content = (
            f"Phase: {phase_name}\n\nMessages:\n"
            + "\n".join(
                f"[{m.get('ts','')}] {m.get('agent','?')} — {m.get('content','')}"
                for m in messages[-40:]  # pass only tail to limit tokens
            )
        )
        result = self._call(system_prompt, user_content)
        if result is None:
            result = (
                f"[Fallback summary] Phase '{phase_name}' produced {len(messages)} "
                "context messages. LLM summarisation unavailable."
            )
        _cache_set(key, result)
        return result

    def generate_report(self, player_name: str, stats: Dict[str, Any]) -> str:
        """
        Generate a short player report.

        Parameters
        ----------
        player_name : str
        stats : dict
            Key statistics dict (avg, strike_rate, wickets, etc.).

        Returns
        -------
        str
            Narrative report (LLM or fallback).
        """
        key = _cache_key({"op": "report", "player": player_name, "stats": stats})
        cached = _cache_get(key)
        if cached is not None:
            return cached

        system_prompt = (
            "You are a cricket analyst. Write a concise 3-sentence player report "
            "using only the provided statistics. Be factual, no speculation."
        )
        user_content = f"Player: {player_name}\nStats: {json.dumps(stats, indent=2)}"
        result = self._call(system_prompt, user_content)
        if result is None:
            avg = stats.get("batting_avg", "N/A")
            sr = stats.get("strike_rate", "N/A")
            result = (
                f"{player_name} has a batting average of {avg} and a strike rate of {sr}. "
                "Detailed LLM report unavailable — set OPENROUTER_API_KEY for AI-generated reports."
            )
        _cache_set(key, result)
        return result

    def explain_prediction(
        self, match_info: Dict[str, Any], prediction: Dict[str, Any]
    ) -> str:
        """
        Explain why a match prediction was made.

        Parameters
        ----------
        match_info : dict
            Match metadata (teams, venue, date, etc.).
        prediction : dict
            Model output (team_a_win_prob, team_b_win_prob, key_features, etc.).

        Returns
        -------
        str
            Short explanation (≤80 words).
        """
        key = _cache_key({"op": "explain", "match": match_info, "pred": prediction})
        cached = _cache_get(key)
        if cached is not None:
            return cached

        system_prompt = (
            "You are a cricket prediction explainer. In ≤80 words, explain why "
            "the model predicts this outcome using the feature importances provided."
        )
        user_content = (
            f"Match: {match_info}\nPrediction: {prediction}"
        )
        result = self._call(system_prompt, user_content)
        if result is None:
            winner = prediction.get("predicted_winner", "Team A")
            prob = prediction.get("win_probability", 0.5)
            result = (
                f"The model favours {winner} with {prob:.0%} confidence based on "
                "historical head-to-head record and recent form. "
                "(LLM explanation unavailable — set OPENROUTER_API_KEY.)"
            )
        _cache_set(key, result)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call(self, system: str, user: str) -> Optional[str]:
        """
        Call OpenRouter with retry + exponential back-off.
        Returns None if API key is missing or all retries fail.
        """
        if not self._api_key:
            logger.debug("[llm] no API key — returning None for fallback")
            return None

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 512,
            "temperature": 0.3,
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with httpx.Client(timeout=30) as client:
                    resp = client.post(OPENROUTER_URL, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                wait = RETRY_BASE ** attempt
                logger.warning("[llm] attempt %d/%d failed: %s — retrying in %.1fs",
                               attempt, MAX_RETRIES, exc, wait)
                if attempt < MAX_RETRIES:
                    time.sleep(wait)

        logger.error("[llm] all %d retries exhausted — falling back", MAX_RETRIES)
        return None
