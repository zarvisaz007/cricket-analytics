"""
train.py — XGBoost model training for match outcome prediction.

If the database is empty a synthetic seed dataset is generated so the
model can be trained immediately for demonstration.

Model is saved to MODEL_PATH with metadata including version and timestamp.
"""
from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "./data/models/xgb_v1.joblib"))
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "team_a_avg_score",
    "team_b_avg_score",
    "team_a_win_rate",
    "team_b_win_rate",
    "h2h_team_a_wins",
    "h2h_total",
    "venue_home_advantage",
    "toss_winner_batting",   # 1 if toss winner chose to bat
    "match_type_encoded",    # T20=0, ODI=1, Test=2
    "team_a_recent_form",    # wins in last 5
    "team_b_recent_form",
]


def _encode_match_type(mt: str) -> int:
    return {"T20": 0, "ODI": 1, "Test": 2}.get(mt.upper(), 0)


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

TEAMS = [
    "India", "Australia", "England", "Pakistan", "South Africa",
    "New Zealand", "West Indies", "Sri Lanka", "Bangladesh", "Afghanistan",
]


def _generate_synthetic_dataset(n: int = 500) -> pd.DataFrame:
    """
    Create a synthetic cricket match dataset for toy model training.
    Stats are randomly generated within realistic bounds.
    """
    random.seed(42)
    np.random.seed(42)
    rows = []
    for _ in range(n):
        ta = random.choice(TEAMS)
        tb = random.choice([t for t in TEAMS if t != ta])
        mt = random.choice(["T20", "ODI", "Test"])
        ta_avg = random.uniform(130, 320)
        tb_avg = random.uniform(130, 320)
        ta_wr = random.uniform(0.3, 0.8)
        tb_wr = random.uniform(0.3, 0.8)
        h2h_a = random.randint(0, 15)
        h2h_tot = h2h_a + random.randint(0, 15)
        home_adv = random.choice([0, 1])
        toss_bat = random.choice([0, 1])
        ta_form = random.randint(0, 5)
        tb_form = random.randint(0, 5)

        # Deterministic label with noise
        score_a = ta_avg * ta_wr + h2h_a / max(h2h_tot, 1) * 30 + home_adv * 10 + ta_form * 5
        score_b = tb_avg * tb_wr + (h2h_tot - h2h_a) / max(h2h_tot, 1) * 30 + tb_form * 5
        label = 1 if score_a > score_b else 0
        # Add 10% label noise for realism
        if random.random() < 0.10:
            label = 1 - label

        rows.append({
            "team_a_avg_score": ta_avg,
            "team_b_avg_score": tb_avg,
            "team_a_win_rate": ta_wr,
            "team_b_win_rate": tb_wr,
            "h2h_team_a_wins": h2h_a,
            "h2h_total": h2h_tot,
            "venue_home_advantage": home_adv,
            "toss_winner_batting": toss_bat,
            "match_type_encoded": _encode_match_type(mt),
            "team_a_recent_form": ta_form,
            "team_b_recent_form": tb_form,
            "label": label,         # 1 = team_a wins
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DB-to-features conversion
# ---------------------------------------------------------------------------

def _load_features_from_db() -> Optional[pd.DataFrame]:
    """
    Extract features from the live DB.  Returns None if insufficient data.
    """
    try:
        from src.data.db import get_session, Match, PlayerStat
        session = get_session()
        matches = session.query(Match).filter(Match.winner.isnot(None)).all()
        if len(matches) < 20:
            logger.info("[train] Only %d labelled matches — using synthetic data", len(matches))
            session.close()
            return None

        rows = []
        for m in matches:
            # Aggregate per-team stats for these two teams (simplified)
            label = 1 if m.winner == m.team_a else 0
            rows.append({
                "team_a_avg_score": 180.0,   # TODO: compute from player_stats
                "team_b_avg_score": 175.0,
                "team_a_win_rate": 0.5,
                "team_b_win_rate": 0.5,
                "h2h_team_a_wins": 5,
                "h2h_total": 10,
                "venue_home_advantage": 0,
                "toss_winner_batting": 1 if m.toss_decision == "bat" else 0,
                "match_type_encoded": _encode_match_type(m.match_type or "T20"),
                "team_a_recent_form": 3,
                "team_b_recent_form": 2,
                "label": label,
            })
        session.close()
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.warning("[train] DB feature load failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(force_synthetic: bool = False) -> Dict[str, Any]:
    """
    Train the XGBoost prediction model.

    Parameters
    ----------
    force_synthetic : bool
        If True, skip DB and use synthetic data (useful for demos).

    Returns
    -------
    dict
        Metadata dict including model_version, accuracy, and saved path.
    """
    logger.info("[train] Starting model training …")

    df = None if force_synthetic else _load_features_from_db()
    if df is None:
        logger.info("[train] Generating synthetic training dataset …")
        df = _generate_synthetic_dataset(500)

    X = df[FEATURE_COLS].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    accuracy = float((clf.predict(X_test) == y_test).mean())
    logger.info("[train] Accuracy on hold-out: %.3f", accuracy)

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "accuracy": accuracy,
        "feature_cols": FEATURE_COLS,
        "n_samples": len(df),
        "model_path": str(MODEL_PATH),
    }

    payload = {"model": clf, "metadata": metadata}
    joblib.dump(payload, MODEL_PATH)
    logger.info("[train] Model saved → %s", MODEL_PATH)

    # Save metadata sidecar
    meta_path = MODEL_PATH.with_suffix(".json")
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model() -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Load the trained model from disk."""
    if not MODEL_PATH.exists():
        logger.warning("[train] Model not found — training now …")
        train(force_synthetic=True)
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["metadata"]


def predict_match(
    team_a: str,
    team_b: str,
    match_type: str = "T20",
    venue: str = "",
    toss_winner: str = "",
    toss_decision: str = "bat",
) -> Dict[str, Any]:
    """
    Predict the outcome of a match.

    Returns
    -------
    dict
        {team_a_win_prob, team_b_win_prob, predicted_winner,
         confidence, key_features, model_version}
    """
    clf, meta = load_model()

    # Build feature vector (simplified — TODO: look up real stats from DB)
    features = np.array([[
        180.0,   # team_a_avg_score
        175.0,   # team_b_avg_score
        0.55,    # team_a_win_rate
        0.50,    # team_b_win_rate
        6,       # h2h_team_a_wins
        12,      # h2h_total
        1 if toss_winner == team_a else 0,
        1 if toss_decision == "bat" else 0,
        _encode_match_type(match_type),
        3,       # team_a_recent_form
        2,       # team_b_recent_form
    ]])

    proba = clf.predict_proba(features)[0]
    # proba[1] = P(team_a wins)
    team_a_prob = float(proba[1])
    team_b_prob = float(proba[0])
    winner = team_a if team_a_prob >= team_b_prob else team_b

    importances = dict(zip(FEATURE_COLS, clf.feature_importances_.tolist()))

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_win_prob": round(team_a_prob, 4),
        "team_b_win_prob": round(team_b_prob, 4),
        "predicted_winner": winner,
        "confidence": round(max(team_a_prob, team_b_prob), 4),
        "key_features": importances,
        "model_version": meta.get("model_version", MODEL_VERSION),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    meta = train(force_synthetic=True)
    print(json.dumps(meta, indent=2))
