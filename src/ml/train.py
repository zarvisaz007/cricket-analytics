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
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.ml.features import (
    build_feature_vector,
    build_training_dataset,
    FEATURE_COLS as NEW_FEATURE_COLS,
    DEFAULT_FEATURES,
)

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "./data/models/xgb_v1.joblib"))
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = "2.0.0"

# ---------------------------------------------------------------------------
# Feature engineering — delegates to features.py (single source of truth)
# ---------------------------------------------------------------------------

FEATURE_COLS = NEW_FEATURE_COLS


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
    Produces the canonical 18 FEATURE_COLS expected by the current model.
    """
    random.seed(42)
    np.random.seed(42)
    rows = []
    for _ in range(n):
        ta = random.choice(TEAMS)
        tb = random.choice([t for t in TEAMS if t != ta])
        mt = random.choice(["T20", "ODI", "Test"])

        ta_elo = random.uniform(1300, 1700)
        tb_elo = random.uniform(1300, 1700)
        ta_strength = random.uniform(20, 90)
        tb_strength = random.uniform(20, 90)
        ta_form = random.uniform(0.2, 0.8)
        tb_form = random.uniform(0.2, 0.8)

        h2h_a = random.randint(0, 15)
        h2h_tot = h2h_a + random.randint(0, 15)
        h2h_win_rate = h2h_a / max(h2h_tot, 1)

        venue_bat = random.uniform(0.8, 1.3)
        venue_spin = random.uniform(0.7, 1.4)
        toss_bat = random.choice([0.0, 1.0])
        toss_is_a = random.choice([0.0, 1.0])
        mt_enc = float(_encode_match_type(mt))

        ta_bat_rating = random.uniform(30, 90)
        tb_bat_rating = random.uniform(30, 90)
        ta_bowl_rating = random.uniform(30, 90)
        tb_bowl_rating = random.uniform(30, 90)

        # Deterministic label with noise
        score_a = (
            ta_elo * 0.01 + ta_strength * 0.5 + ta_form * 20
            + h2h_win_rate * 15 + ta_bat_rating * 0.3
        )
        score_b = (
            tb_elo * 0.01 + tb_strength * 0.5 + tb_form * 20
            + (1 - h2h_win_rate) * 15 + tb_bat_rating * 0.3
        )
        label = 1 if score_a > score_b else 0
        # Add 10% label noise for realism
        if random.random() < 0.10:
            label = 1 - label

        rows.append({
            "team_a_elo": ta_elo,
            "team_b_elo": tb_elo,
            "elo_diff": ta_elo - tb_elo,
            "team_a_strength": ta_strength,
            "team_b_strength": tb_strength,
            "team_a_form_last10": ta_form,
            "team_b_form_last10": tb_form,
            "h2h_team_a_win_rate": h2h_win_rate,
            "h2h_total": float(h2h_tot),
            "venue_batting_factor": venue_bat,
            "venue_spin_factor": venue_spin,
            "toss_winner_batting": toss_bat,
            "toss_winner_is_team_a": toss_is_a,
            "match_type_encoded": mt_enc,
            "team_a_top_batsman_rating": ta_bat_rating,
            "team_b_top_batsman_rating": tb_bat_rating,
            "team_a_top_bowler_rating": ta_bowl_rating,
            "team_b_top_bowler_rating": tb_bowl_rating,
            "label": label,         # 1 = team_a wins
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DB-to-features conversion
# ---------------------------------------------------------------------------

def _load_features_from_db() -> Optional[pd.DataFrame]:
    """
    Extract features from the live DB using the features.py pipeline.
    Returns None if insufficient data.
    """
    try:
        from src.data.db import get_session
        session = get_session()
        try:
            df = build_training_dataset(session)
        finally:
            session.close()
        return df
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

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    accuracy = float((y_pred == y_test).mean())
    model_log_loss = float(log_loss(y_test, y_proba))
    model_brier = float(brier_score_loss(y_test, y_proba[:, 1]))

    logger.info(
        "[train] Accuracy=%.3f  LogLoss=%.4f  Brier=%.4f",
        accuracy, model_log_loss, model_brier,
    )

    trained_at = datetime.now(timezone.utc).isoformat()

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": trained_at,
        "accuracy": accuracy,
        "log_loss": model_log_loss,
        "brier_score": model_brier,
        "feature_cols": FEATURE_COLS,
        "n_samples": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model_path": str(MODEL_PATH),
    }

    payload = {"model": clf, "metadata": metadata}
    joblib.dump(payload, MODEL_PATH)
    logger.info("[train] Model saved → %s", MODEL_PATH)

    # Save metadata sidecar
    meta_path = MODEL_PATH.with_suffix(".json")
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    # Persist ModelRecord to DB
    try:
        from src.data.db import get_session, ModelRecord
        session = get_session()
        try:
            existing = (
                session.query(ModelRecord)
                .filter(ModelRecord.version == MODEL_VERSION)
                .first()
            )
            if existing:
                existing.trained_at = trained_at
                existing.accuracy = accuracy
                existing.log_loss = model_log_loss
                existing.brier_score = model_brier
                existing.n_train = len(X_train)
                existing.n_test = len(X_test)
                existing.feature_cols_json = json.dumps(FEATURE_COLS)
                existing.model_path = str(MODEL_PATH)
            else:
                record = ModelRecord(
                    version=MODEL_VERSION,
                    trained_at=trained_at,
                    accuracy=accuracy,
                    log_loss=model_log_loss,
                    brier_score=model_brier,
                    n_train=len(X_train),
                    n_test=len(X_test),
                    feature_cols_json=json.dumps(FEATURE_COLS),
                    model_path=str(MODEL_PATH),
                )
                session.add(record)
            session.commit()
            logger.info("[train] ModelRecord upserted for version %s", MODEL_VERSION)
        except Exception as db_exc:
            session.rollback()
            logger.warning("[train] ModelRecord DB upsert failed: %s", db_exc)
        finally:
            session.close()
    except Exception as exc:
        logger.warning("[train] Could not persist ModelRecord: %s", exc)

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

    # Build feature vector from DB using the features pipeline
    match_dict = {
        "team_a": team_a,
        "team_b": team_b,
        "match_type": match_type,
        "venue": venue,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
    }
    try:
        from src.data.db import get_session
        session = get_session()
        try:
            features, _ = build_feature_vector(match_dict, session)
        finally:
            session.close()
    except Exception as exc:
        logger.warning("[predict] Feature build failed, using defaults: %s", exc)
        features = np.array(
            [[DEFAULT_FEATURES[col] for col in FEATURE_COLS]],
            dtype=np.float64,
        )

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
