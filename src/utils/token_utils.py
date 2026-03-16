"""
token_utils.py — Helpers for token estimation and message trimming.

Uses tiktoken when available; falls back to a character-count heuristic (len // 4).
"""
from __future__ import annotations

import os
import json
from typing import Union

# ---------------------------------------------------------------------------
# tiktoken — optional dependency
# ---------------------------------------------------------------------------
try:
    import tiktoken as _tiktoken
    _TIKTOKEN_OK = True
except ImportError:
    _TIKTOKEN_OK = False


def _get_encoding():
    """Return a tiktoken encoding for the configured model, or None on failure."""
    if not _TIKTOKEN_OK:
        return None
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    try:
        return _tiktoken.encoding_for_model(model)
    except Exception:
        try:
            # fallback to a well-known encoding
            return _tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def estimate_tokens(text_or_messages: Union[str, list]) -> int:
    """
    Estimate the number of tokens in *text_or_messages*.

    Parameters
    ----------
    text_or_messages:
        Either a raw string or a list of message dicts
        (each with at least a ``"content"`` key).

    Returns
    -------
    int
        Estimated token count.
    """
    if isinstance(text_or_messages, list):
        # Flatten all content fields into one string for estimation.
        text = " ".join(
            m.get("content", "") if isinstance(m, dict) else str(m)
            for m in text_or_messages
        )
    else:
        text = str(text_or_messages)

    enc = _get_encoding()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # Heuristic fallback: ~4 chars per token
    return max(1, len(text) // 4)


def trim_messages_to_token_limit(messages: list, limit: int) -> list:
    """
    Trim *messages* (oldest-first) so that the total token count stays
    below *limit*.  Always keeps the most-recent messages.

    Parameters
    ----------
    messages : list
        List of message dicts (newest last).
    limit : int
        Maximum allowed token count.

    Returns
    -------
    list
        A (possibly shorter) list of messages that fits within *limit*.
    """
    if not messages:
        return messages

    total = estimate_tokens(messages)
    if total <= limit:
        return messages

    # Drop oldest messages until we fit
    trimmed = list(messages)
    while trimmed and estimate_tokens(trimmed) > limit:
        trimmed.pop(0)
    return trimmed
