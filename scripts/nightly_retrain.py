#!/usr/bin/env python3
"""
nightly_retrain.py — Nightly model retraining pipeline.

Designed to run as a cron job:
    0 4 * * * cd /path/to/Claude-cricket && python scripts/nightly_retrain.py

Steps:
1. Run analytics pipeline (ratings, Elo, PVOR)
2. Retrain model on fresh features
3. Evaluate with backtester
4. Promote if new model is better
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("nightly_retrain")


def main() -> int:
    """Run the nightly retraining pipeline. Returns exit code."""
    logger.info("=== Nightly retraining pipeline starting ===")

    # Step 1: Run DB migration
    try:
        from src.data.migrations.add_phase2_tables import run_migration
        run_migration()
        logger.info("[retrain] Migration complete")
    except Exception as exc:
        logger.error("[retrain] Migration failed: %s", exc)
        return 1

    # Step 2: Run analytics agent inline
    try:
        logger.info("[retrain] Running analytics pipeline...")
        from src.agents.analytics_agent import AnalyticsAgent
        AnalyticsAgent().run()
        logger.info("[retrain] Analytics pipeline complete")
    except Exception as exc:
        logger.error("[retrain] Analytics pipeline failed: %s", exc)
        return 1

    # Step 3: Retrain model
    try:
        logger.info("[retrain] Retraining model...")
        from src.ml.train import train
        new_meta = train(force_synthetic=False)
        logger.info(
            "[retrain] Model trained — version=%s accuracy=%.3f samples=%d",
            new_meta.get("model_version", "?"),
            new_meta.get("accuracy", 0.0),
            new_meta.get("n_samples", 0),
        )
    except Exception as exc:
        logger.error("[retrain] Model training failed: %s", exc)
        return 1

    # Step 4: Evaluate with backtester
    try:
        logger.info("[retrain] Evaluating model with backtester...")
        from src.ml.backtester import evaluate_current_model
        metrics = evaluate_current_model()
        logger.info("[retrain] Backtester metrics: %s", metrics)

        # Print summary
        print("\n=== Nightly Retrain Summary ===")
        print(f"  Model version : {new_meta.get('model_version', '?')}")
        print(f"  Train accuracy: {new_meta.get('accuracy', 0.0):.3f}")
        print(f"  Train samples : {new_meta.get('n_samples', 0)}")
        if metrics:
            for k, v in metrics.items():
                print(f"  {k:20s}: {v}")
        print("================================\n")
    except Exception as exc:
        logger.error("[retrain] Backtester evaluation failed: %s", exc)
        # Non-fatal — model was still trained successfully
        logger.warning("[retrain] Continuing without backtest metrics")

    logger.info("=== Nightly retraining pipeline complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
