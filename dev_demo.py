"""
dev_demo.py — End-to-end synthetic demo.

Demonstrates:
1. DB initialisation
2. Synthetic match seeding
3. Toy XGBoost model training
4. Bot-agent style prediction
5. Context purge triggered by synthetic messages
6. Shows archive file path

Run: python dev_demo.py
"""
from __future__ import annotations

import json
import logging
import os
import sys

# Make sure we can import from src/
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()


def main() -> None:
    console.print(Rule("[bold cyan]Cricket Analytics — Dev Demo[/bold cyan]"))

    # ── 1. Init DB ─────────────────────────────────────────────────────────
    console.print("\n[bold]1. Initialising database …[/bold]")
    from src.data.db import init_db, get_session, Match
    init_db()
    console.print("   ✓ DB tables created")

    # ── 2. Seed matches ─────────────────────────────────────────────────────
    console.print("\n[bold]2. Seeding synthetic matches (200) …[/bold]")
    from src.agents.ingestion_agent import _generate_mock_matches
    matches_data = _generate_mock_matches(200)
    session = get_session()
    inserted = 0
    for m in matches_data:
        if not session.query(Match).filter_by(match_key=m["match_key"]).first():
            session.add(Match(**m))
            inserted += 1
    session.commit()
    session.close()
    console.print(f"   ✓ {inserted} matches inserted into DB")

    # ── 3. Train model ─────────────────────────────────────────────────────
    console.print("\n[bold]3. Training XGBoost model …[/bold]")
    from src.ml.train import train
    meta = train(force_synthetic=True)
    console.print(
        f"   ✓ Model v{meta['model_version']} trained "
        f"— accuracy={meta['accuracy']:.3f} samples={meta['n_samples']}"
    )
    console.print(f"   ✓ Saved → {meta['model_path']}")

    # ── 4. Single prediction ───────────────────────────────────────────────
    console.print("\n[bold]4. Sample prediction: India vs Australia (T20) …[/bold]")
    from src.ml.train import predict_match
    pred = predict_match("India", "Australia", match_type="T20")
    console.print(Panel(
        f"India win prob:     {pred['team_a_win_prob'] * 100:.1f}%\n"
        f"Australia win prob: {pred['team_b_win_prob'] * 100:.1f}%\n"
        f"Predicted winner:   {pred['predicted_winner']} "
        f"({pred['confidence'] * 100:.0f}% confidence)",
        title="Prediction",
        border_style="green",
    ))

    # Explanation via NLP agent
    from src.agents.nlp_agent import NLPAgent
    explanation = NLPAgent().explain_prediction(
        {"team_a": "India", "team_b": "Australia", "match_type": "T20"}, pred
    )
    console.print(f"   💬 Explanation: {explanation}")

    # ── 5. Context purge demo ──────────────────────────────────────────────
    console.print("\n[bold]5. Demonstrating context purge …[/bold]")
    from src.agents.context_manager import context_manager, HARD_LIMIT

    # Inject enough synthetic messages to trigger the hard limit
    fake_message = "x" * 400  # ~100 tokens per message
    threshold_msgs = (HARD_LIMIT // 100) + 5

    console.print(f"   Adding {threshold_msgs} synthetic messages "
                  f"(target: >{HARD_LIMIT:,} tokens) …")

    for i in range(threshold_msgs):
        context_manager.add_message(
            "demo_agent", "synthetic",
            f"synthetic event #{i}: {fake_message}"
        )
        if context_manager.should_purge():
            console.print(
                f"   ⚡ Purge threshold hit at message #{i} "
                f"({context_manager.current_token_count():,} tokens)"
            )
            break

    summary = context_manager.purge_and_archive("demo_purge", "token_threshold")

    console.print(Panel(
        f"Phase:    {summary['phase']}\n"
        f"Reason:   {summary['reason']}\n"
        f"Messages: {summary['message_count']}\n"
        f"Tokens:   {summary['token_count_before_purge']:,}\n"
        f"Archive:  {summary.get('archive_path', 'N/A')}\n"
        f"Failure:  {summary.get('archival_failure', False)}",
        title="Purge Summary",
        border_style="yellow",
    ))
    console.print(f"   ✓ Archive file: [cyan]{summary.get('archive_path')}[/cyan]")
    console.print(f"   ✓ Summaries JSONL: [cyan]data/context_summaries.jsonl[/cyan]")

    console.print(Rule())
    console.print("\n[bold green]MVP ready — how would you like the first demo flow to behave?[/bold green]\n")


if __name__ == "__main__":
    main()
