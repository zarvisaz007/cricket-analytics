"""
backtester.py — Time-series backtesting for match prediction model.

Trains on data before a cutoff date, evaluates on data after.
Reports accuracy, log-loss, Brier score.

Usage
-----
    from src.ml.backtester import run_backtest, evaluate_current_model

    results = run_backtest(train_before="2022-01-01", test_after="2022-01-01")
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

_EMPTY_METRICS: Dict[str, Any] = {
    "n_train": 0,
    "n_test": 0,
    "accuracy": 0,
    "log_loss": 1.0,
    "brier_score": 0.25,
    "calibration_data": [],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_by_date(
    session,
    train_before: str,
    test_after: str,
) -> Tuple[Optional[object], Optional[object]]:
    """
    Split the training dataset into train/test portions by match date.

    Returns (train_df, test_df). Either may be None on failure.
    """
    from src.ml.features import build_training_dataset, FEATURE_COLS
    from src.data.db import Match as MatchModel

    all_df = build_training_dataset(session)
    if all_df is None or len(all_df) == 0:
        return None, None

    # We need match dates — rebuild via direct DB query to attach dates
    # build_training_dataset doesn't carry match_date so we re-query here
    matches = (
        session.query(MatchModel)
        .filter(MatchModel.winner.isnot(None))
        .order_by(MatchModel.match_date.asc())
        .all()
    )

    # Build a date-indexed dict keyed by (team_a, team_b, match_date)
    # For simplicity, rely on row ordering being consistent with query order
    if len(all_df) != len(matches):
        # Can happen if build_training_dataset skipped some rows
        # Fall back: split by index position
        logger.debug("_split_by_date: row count mismatch (%d vs %d), splitting by index", len(all_df), len(matches))
        split_idx = int(len(all_df) * 0.8)
        return all_df.iloc[:split_idx], all_df.iloc[split_idx:]

    dates = [m.match_date or "" for m in matches]
    all_df = all_df.copy()
    all_df["_match_date"] = dates

    train_df = all_df[all_df["_match_date"] < train_before].drop(columns=["_match_date"])
    test_df = all_df[all_df["_match_date"] >= test_after].drop(columns=["_match_date"])

    return (
        train_df if len(train_df) > 0 else None,
        test_df if len(test_df) > 0 else None,
    )


def _compute_metrics(
    clf: XGBClassifier,
    X_test: object,
    y_test: object,
) -> Dict[str, Any]:
    """Compute accuracy, log_loss, brier_score and calibration_data."""
    from sklearn.metrics import log_loss, brier_score_loss
    import numpy as np

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    accuracy = float((y_pred == y_test).mean())
    model_log_loss = float(log_loss(y_test, y_proba))
    model_brier = float(brier_score_loss(y_test, y_proba[:, 1]))

    calibration_data: List[Tuple[float, int]] = [
        (float(p), int(a)) for p, a in zip(y_proba[:, 1].tolist(), y_test.tolist())
    ]

    return {
        "accuracy": round(accuracy, 4),
        "log_loss": round(model_log_loss, 4),
        "brier_score": round(model_brier, 4),
        "calibration_data": calibration_data,
    }


def _train_xgb(X_train, y_train) -> XGBClassifier:
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest(
    train_before: str = "2022-01-01",
    test_after: str = "2022-01-01",
    session=None,
) -> Dict[str, Any]:
    """
    Time-series backtest: train on data before cutoff, evaluate on data after.

    Parameters
    ----------
    train_before : str
        ISO date string (YYYY-MM-DD). Matches before this date form the
        training set.
    test_after : str
        ISO date string (YYYY-MM-DD). Matches on or after this date form the
        test set.
    session :
        SQLAlchemy session. Created and closed internally if None.

    Returns
    -------
    dict with keys: n_train, n_test, accuracy, log_loss, brier_score,
                    calibration_data
    """
    _own_session = False
    if session is None:
        try:
            from src.data.db import get_session
            session = get_session()
            _own_session = True
        except Exception as exc:
            logger.warning("run_backtest: could not open DB session: %s", exc)
            return dict(_EMPTY_METRICS)

    try:
        from src.ml.features import FEATURE_COLS

        train_df, test_df = _split_by_date(session, train_before, test_after)

        if train_df is None or len(train_df) < 10:
            logger.info("run_backtest: insufficient training data")
            return dict(_EMPTY_METRICS)

        if test_df is None or len(test_df) < 5:
            logger.info("run_backtest: insufficient test data")
            return dict(_EMPTY_METRICS)

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df["label"].values
        X_test = test_df[FEATURE_COLS].values
        y_test = test_df["label"].values

        clf = _train_xgb(X_train, y_train)
        metrics = _compute_metrics(clf, X_test, y_test)

        return {
            "n_train": len(X_train),
            "n_test": len(X_test),
            **metrics,
        }

    except Exception as exc:
        logger.warning("run_backtest failed: %s", exc)
        return dict(_EMPTY_METRICS)
    finally:
        if _own_session and session is not None:
            session.close()


def evaluate_current_model(session=None) -> Dict[str, Any]:
    """
    Evaluate the currently saved model on the most recent 20% of matches.

    Parameters
    ----------
    session :
        SQLAlchemy session. Created internally if None.

    Returns
    -------
    dict with keys: n_train, n_test, accuracy, log_loss, brier_score,
                    calibration_data
    """
    _own_session = False
    if session is None:
        try:
            from src.data.db import get_session
            session = get_session()
            _own_session = True
        except Exception as exc:
            logger.warning("evaluate_current_model: could not open DB session: %s", exc)
            return dict(_EMPTY_METRICS)

    try:
        from src.ml.train import load_model
        from src.ml.features import build_training_dataset, FEATURE_COLS

        clf, _meta = load_model()

        df = build_training_dataset(session)
        if df is None or len(df) < 10:
            logger.info("evaluate_current_model: insufficient data")
            return dict(_EMPTY_METRICS)

        # Use the last 20% as test set
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:]

        if len(test_df) < 5:
            logger.info("evaluate_current_model: test set too small (%d rows)", len(test_df))
            return dict(_EMPTY_METRICS)

        X_test = test_df[FEATURE_COLS].values
        y_test = test_df["label"].values

        metrics = _compute_metrics(clf, X_test, y_test)

        return {
            "n_train": split_idx,
            "n_test": len(test_df),
            **metrics,
        }

    except Exception as exc:
        logger.warning("evaluate_current_model failed: %s", exc)
        return dict(_EMPTY_METRICS)
    finally:
        if _own_session and session is not None:
            session.close()


def should_promote_new_model(
    old_metrics: Dict[str, Any],
    new_metrics: Dict[str, Any],
    min_improvement: float = 0.02,
) -> bool:
    """
    Decide whether to promote a newly trained model over the current one.

    Returns True if the new model's accuracy is no worse than
    (old accuracy - min_improvement).  A 2% buffer prevents noisy degradation
    from triggering unwanted rollbacks.

    Parameters
    ----------
    old_metrics : dict
        Metrics from the current production model (must include 'accuracy').
    new_metrics : dict
        Metrics from the candidate new model.
    min_improvement : float
        Minimum improvement buffer (default 0.02 = 2 percentage points).

    Returns
    -------
    bool
    """
    old_acc = float(old_metrics.get("accuracy", 0.0))
    new_acc = float(new_metrics.get("accuracy", 0.0))
    threshold = old_acc - min_improvement
    promote = new_acc >= threshold
    logger.info(
        "should_promote_new_model: old=%.4f new=%.4f threshold=%.4f → %s",
        old_acc, new_acc, threshold, promote,
    )
    return promote
