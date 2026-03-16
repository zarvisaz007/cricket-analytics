"""
nlp_agent.py — NLP/LLM wrapper agent.

Provides summarise, report, and explain capabilities.
Delegates to llm_client.py for actual LLM calls.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("nlp_agent")


class NLPAgent:
    """Thin agent wrapper around LLMClient for pipeline use."""

    def run(self) -> None:
        """Idle loop — NLPAgent is invoked on-demand by other agents."""
        from src.agents.context_manager import context_manager
        from src.agents.orchestrator import ipc_write
        ipc_write(f"reg_nlp_{int(time.time())}.json", {
            "agent": "nlp_agent", "role": "register", "content": "nlp_agent ready",
        })
        context_manager.add_message("nlp_agent", "info", "nlp_agent ready (on-demand)")

    # ------------------------------------------------------------------
    # Public API called by other agents
    # ------------------------------------------------------------------

    def summarize_json(self, messages: List[Dict], phase: str) -> str:
        """
        Produce a summary string for *messages* from *phase*.

        Parameters
        ----------
        messages : list
            Ephemeral context message dicts.
        phase : str
            Pipeline phase name.

        Returns
        -------
        str
            Summary text.
        """
        from src.nlp.llm_client import LLMClient
        client = LLMClient()
        result = client.summarize(messages, phase)
        logger.info("[nlp_agent] summarize phase=%s len=%d", phase, len(messages))
        return result

    def explain_prediction(
        self, match_info: Dict[str, Any], prediction: Dict[str, Any]
    ) -> str:
        """
        Return a short explanation for a match prediction.

        Parameters
        ----------
        match_info : dict
        prediction : dict

        Returns
        -------
        str
        """
        from src.agents.context_manager import context_manager
        from src.nlp.llm_client import LLMClient
        client = LLMClient()
        result = client.explain_prediction(match_info, prediction)
        context_manager.add_message(
            "nlp_agent", "info",
            f"explain_prediction for {match_info.get('team_a')} vs {match_info.get('team_b')}"
        )
        return result

    def generate_report(self, player_name: str, stats: Dict[str, Any]) -> str:
        """
        Generate a narrative player report.

        Parameters
        ----------
        player_name : str
        stats : dict

        Returns
        -------
        str
        """
        from src.agents.context_manager import context_manager
        from src.nlp.llm_client import LLMClient
        client = LLMClient()
        result = client.generate_report(player_name, stats)
        context_manager.add_message("nlp_agent", "info", f"generate_report for {player_name}")
        return result
