"""
context_manager.py — Shared in-memory context for all sub-agents.

Responsibilities
----------------
* Stores ephemeral agent messages in a list.
* Estimates token usage via token_utils.
* Triggers purge when usage crosses the OR-threshold:
    - tokens > MODEL_MAX_TOKENS * CONTEXT_PURGE_PERCENT / 100
    - tokens > CONTEXT_HARD_LIMIT (absolute ceiling)
* On purge: archives to JSONL, writes summary via NLP agent or local
  extractive summariser, and logs to run/metrics.json.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from src.utils.token_utils import estimate_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARCHIVES_DIR = Path("./data/context_archives")
SUMMARIES_FILE = Path("./data/context_summaries.jsonl")
METRICS_FILE = Path("./run/metrics.json")

# ---------------------------------------------------------------------------
# Config (from env)
# ---------------------------------------------------------------------------
MODEL_MAX_TOKENS: int = int(os.getenv("MODEL_MAX_TOKENS", "200000"))
PURGE_PERCENT: float = float(os.getenv("CONTEXT_PURGE_PERCENT", "50"))
HARD_LIMIT: int = int(os.getenv("CONTEXT_HARD_LIMIT", "100000"))
FORCE_KEEP: bool = os.getenv("FORCE_KEEP_EPHEMERAL", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Simple extractive summariser (TF-IDF sentence ranking)
# ---------------------------------------------------------------------------

def _extractive_summary(messages: List[Dict], max_sentences: int = 8) -> str:
    """
    Produce a short extractive summary by TF-IDF sentence ranking.
    Returns a plain-text summary string.
    """
    # Build corpus of sentences from content fields
    sentences: List[str] = []
    for m in messages:
        text = m.get("content", "")
        for sent in re.split(r"(?<=[.!?])\s+", text):
            sent = sent.strip()
            if len(sent) > 10:
                sentences.append(sent)

    if not sentences:
        return "No content to summarise."

    # TF
    def tokenize(s: str) -> List[str]:
        return re.findall(r"[a-z]+", s.lower())

    term_freq: List[Dict[str, float]] = []
    for s in sentences:
        tokens = tokenize(s)
        freq: Dict[str, float] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        total = max(len(tokens), 1)
        term_freq.append({t: c / total for t, c in freq.items()})

    # IDF
    N = len(sentences)
    df: Dict[str, int] = {}
    for freq in term_freq:
        for t in freq:
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log((N + 1) / (c + 1)) for t, c in df.items()}

    # Score sentences
    scores = []
    for freq in term_freq:
        score = sum(f * idf.get(t, 0) for t, f in freq.items())
        scores.append(score)

    ranked = sorted(range(N), key=lambda i: scores[i], reverse=True)
    top_indices = sorted(ranked[:max_sentences])
    return " ".join(sentences[i] for i in top_indices)


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class ContextManager:
    """Thread-safe (single-process) in-memory context store for agent messages."""

    def __init__(self) -> None:
        self._messages: List[Dict[str, Any]] = []
        self._latest_summary: Optional[Dict[str, Any]] = None
        self._purge_count: int = 0
        ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
        SUMMARIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_message(
        self,
        agent: str,
        role: str,
        content: str,
    ) -> None:
        """
        Append an ephemeral message to the context.

        Parameters
        ----------
        agent : str
            Name of the agent emitting the message.
        role : str
            Semantic role: ``"info"``, ``"error"``, ``"event"``, etc.
        content : str
            Human-readable message body.
        """
        entry: Dict[str, Any] = {
            "agent": agent,
            "role": role,
            "ts": datetime.now(timezone.utc).isoformat(),
            "content": content,
        }
        self._messages.append(entry)
        logger.debug("[context] +msg agent=%s role=%s tokens_est=%d",
                     agent, role, self.current_token_count())

    def current_token_count(self) -> int:
        """Return estimated token count of the current ephemeral message list."""
        return estimate_tokens(self._messages)

    def should_purge(self) -> bool:
        """
        Return True when the OR-threshold is triggered:
          - tokens > MODEL_MAX_TOKENS * PURGE_PERCENT / 100
          - tokens > HARD_LIMIT
        """
        n = self.current_token_count()
        pct_threshold = int(MODEL_MAX_TOKENS * PURGE_PERCENT / 100)
        return n > pct_threshold or n > HARD_LIMIT

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return a copy of the current ephemeral messages."""
        return list(self._messages)

    def get_latest_summary(self) -> Optional[Dict[str, Any]]:
        """
        Return the most recent purge summary so agents can rehydrate context.
        """
        return self._latest_summary

    def purge_and_archive(
        self,
        phase_name: str,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Archive the current ephemeral context, produce a summary, and reset.

        Parameters
        ----------
        phase_name : str
            Label for the current pipeline phase (e.g. ``"phase_ingestion"``).
        reason : str
            Why the purge was triggered (e.g. ``"phase_complete"`` or
            ``"token_threshold"``).

        Returns
        -------
        dict
            The summary object that was persisted to disk.
        """
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        messages_snapshot = list(self._messages)
        token_count = self.current_token_count()

        # ── 1. Build summary text ──────────────────────────────────────────
        summary_text = self._build_summary(messages_snapshot, phase_name)

        summary_obj: Dict[str, Any] = {
            "phase": phase_name,
            "reason": reason,
            "ts": ts_str,
            "token_count_before_purge": token_count,
            "message_count": len(messages_snapshot),
            "summary": summary_text,
            "archive_path": None,
            "archival_failure": False,
        }

        # ── 2. Write archive JSONL ─────────────────────────────────────────
        archive_path = ARCHIVES_DIR / f"{ts_str}_{phase_name}.jsonl"
        try:
            with archive_path.open("w", encoding="utf-8") as f:
                for msg in messages_snapshot:
                    f.write(json.dumps(msg) + "\n")
            summary_obj["archive_path"] = str(archive_path)
            logger.info("[context] archived %d messages -> %s", len(messages_snapshot), archive_path)
        except OSError as exc:
            logger.error("[context] archive write failed: %s", exc)
            summary_obj["archival_failure"] = True

        # ── 3. Append summary to summaries JSONL ──────────────────────────
        try:
            with SUMMARIES_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(summary_obj) + "\n")
        except OSError as exc:
            logger.error("[context] summary write failed: %s", exc)
            summary_obj["archival_failure"] = True

        # ── 4. Reset ephemeral list ────────────────────────────────────────
        if not FORCE_KEEP:
            self._messages = []

        self._latest_summary = summary_obj
        self._purge_count += 1

        # ── 5. Log purge event to metrics ─────────────────────────────────
        self._log_metric("purge", {
            "phase": phase_name,
            "reason": reason,
            "ts": ts_str,
            "token_count": token_count,
            "purge_index": self._purge_count,
        })

        logger.info("[context] purge #%d complete — phase=%s reason=%s tokens=%d",
                    self._purge_count, phase_name, reason, token_count)
        return summary_obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_summary(self, messages: List[Dict], phase_name: str) -> str:
        """
        Attempt LLM summarisation via nlp_agent; fall back to extractive summary.
        """
        if os.getenv("OPENROUTER_API_KEY", "").strip():
            try:
                from src.nlp.llm_client import LLMClient
                client = LLMClient()
                return client.summarize(messages, phase_name)
            except Exception as exc:
                logger.warning("[context] LLM summarise failed (%s); using extractive", exc)
        return _extractive_summary(messages)

    @staticmethod
    def _log_metric(event: str, data: Dict[str, Any]) -> None:
        """Append a structured event to run/metrics.json."""
        try:
            if METRICS_FILE.exists():
                with METRICS_FILE.open("r", encoding="utf-8") as f:
                    metrics = json.load(f)
            else:
                metrics = {}
            metrics.setdefault("events", []).append({"event": event, **data})
            with METRICS_FILE.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        except Exception as exc:
            logger.error("[context] metrics write failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton — imported by all agents
# ---------------------------------------------------------------------------
context_manager = ContextManager()
