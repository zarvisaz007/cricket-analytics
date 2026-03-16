"""
simulator.py — Monte Carlo match simulation.

Simulates N match outcomes by sampling from player performance distributions.
Used for: win probability confidence intervals, score distributions, player impact.

Performance distribution:
- Batting: runs ~ Gamma(shape=2, scale=expected_runs/2)
- Bowling: wickets ~ Poisson(lambda=wickets_per_match)

All simulations are vectorised using numpy (no Python loops over n_sims).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_EXPECTED_RUNS: float = 25.0
_DEFAULT_STD: float = 20.0
_GAMMA_SHAPE: float = 2.0
_DEFAULT_WICKETS_PER_MATCH: float = 0.5   # per bowler per match
_N_PLAYERS: int = 11


# ---------------------------------------------------------------------------
# Player data helpers
# ---------------------------------------------------------------------------

def get_player_expected_runs(
    team_name: str,
    format: str,
    session,
) -> List[Tuple[str, float, float]]:
    """
    Return expected batting performance for up to 11 players of a team.

    Returns
    -------
    List of (player_name, expected_runs, std) tuples — length 11.
    Falls back to synthetic players with expected_runs=25 when no data found.
    """
    try:
        from src.data.db import PlayerFeature, PlayerStat, Match as MatchModel
        from sqlalchemy import desc

        player_id_rows = (
            session.query(PlayerStat.player_id)
            .join(MatchModel, PlayerStat.match_id == MatchModel.id)
            .filter(
                PlayerStat.team == team_name,
                MatchModel.match_type == format,
            )
            .distinct()
            .all()
        )
        player_ids = [r[0] for r in player_id_rows if r[0] is not None]

        if not player_ids:
            raise ValueError(f"No player IDs found for {team_name}/{format}")

        result: List[Tuple[str, float, float]] = []
        from src.data.db import Player, PlayerFeature as PF

        for pid in player_ids:
            if len(result) >= _N_PLAYERS:
                break

            player = session.get(Player, pid)
            pname = player.name if player else f"player_{pid}"

            pf_row = (
                session.query(PF)
                .filter(
                    PF.player_id == pid,
                    PF.format == format,
                    PF.batting_avg.isnot(None),
                )
                .order_by(desc(PF.snapshot_date))
                .first()
            )

            if pf_row and pf_row.batting_avg is not None:
                expected = float(pf_row.batting_avg)
                std = float(pf_row.strike_rate or _DEFAULT_STD)
            else:
                expected = _DEFAULT_EXPECTED_RUNS
                std = _DEFAULT_STD

            result.append((pname, expected, std))

        # Pad to exactly 11 players if needed
        while len(result) < _N_PLAYERS:
            result.append((f"{team_name}_player_{len(result)+1}", _DEFAULT_EXPECTED_RUNS, _DEFAULT_STD))

        return result[:_N_PLAYERS]

    except Exception as exc:
        logger.debug("get_player_expected_runs failed for %s/%s: %s", team_name, format, exc)
        return [
            (f"{team_name}_player_{i+1}", _DEFAULT_EXPECTED_RUNS, _DEFAULT_STD)
            for i in range(_N_PLAYERS)
        ]


def _get_player_wickets_per_match(
    team_name: str,
    format: str,
    session,
) -> List[float]:
    """
    Return expected wickets per match for up to 11 players of a team.
    Falls back to _DEFAULT_WICKETS_PER_MATCH when no data is available.
    """
    try:
        from src.data.db import PlayerFeature, PlayerStat, Match as MatchModel, Player
        from sqlalchemy import desc

        player_id_rows = (
            session.query(PlayerStat.player_id)
            .join(MatchModel, PlayerStat.match_id == MatchModel.id)
            .filter(
                PlayerStat.team == team_name,
                MatchModel.match_type == format,
            )
            .distinct()
            .all()
        )
        player_ids = [r[0] for r in player_id_rows if r[0] is not None]

        wickets_list: List[float] = []
        for pid in player_ids:
            if len(wickets_list) >= _N_PLAYERS:
                break

            pf_row = (
                session.query(PlayerFeature)
                .filter(
                    PlayerFeature.player_id == pid,
                    PlayerFeature.format == format,
                    PlayerFeature.bowling_avg.isnot(None),
                )
                .order_by(desc(PlayerFeature.snapshot_date))
                .first()
            )

            if pf_row and pf_row.bowling_avg is not None and pf_row.bowling_avg > 0:
                # Approximate wickets_per_match from bowling_avg (runs per wicket)
                # Assume ~30 runs/match faced → wickets ≈ 30/bowling_avg
                wickets_lambda = min(30.0 / float(pf_row.bowling_avg), 3.0)
            else:
                wickets_lambda = _DEFAULT_WICKETS_PER_MATCH

            wickets_list.append(wickets_lambda)

        # Pad to 11
        while len(wickets_list) < _N_PLAYERS:
            wickets_list.append(_DEFAULT_WICKETS_PER_MATCH)

        return wickets_list[:_N_PLAYERS]

    except Exception as exc:
        logger.debug("_get_player_wickets_per_match failed for %s/%s: %s", team_name, format, exc)
        return [_DEFAULT_WICKETS_PER_MATCH] * _N_PLAYERS


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_match(
    team_a: str,
    team_b: str,
    format: str = "T20",
    venue: str = "",
    session=None,
    n_sims: int = 2000,
) -> Dict:
    """
    Monte Carlo match simulation.

    Parameters
    ----------
    team_a, team_b : str
        Team names.
    format : str
        Match format (T20, ODI, Test).
    venue : str
        Venue name (unused directly here; passed for future extension).
    session :
        SQLAlchemy session. Created internally if None.
    n_sims : int
        Number of simulation iterations (default 2000).

    Returns
    -------
    dict with keys:
        win_prob_a, win_prob_b,
        score_p10_a, score_p50_a, score_p90_a,
        score_p10_b, score_p50_b, score_p90_b,
        n_sims
    """
    _own_session = False
    if session is None:
        try:
            from src.data.db import get_session
            session = get_session()
            _own_session = True
        except Exception as exc:
            logger.warning("simulate_match: could not open DB session: %s", exc)
            session = None

    try:
        players_a = get_player_expected_runs(team_a, format, session) if session else [
            (f"{team_a}_p{i}", _DEFAULT_EXPECTED_RUNS, _DEFAULT_STD) for i in range(_N_PLAYERS)
        ]
        players_b = get_player_expected_runs(team_b, format, session) if session else [
            (f"{team_b}_p{i}", _DEFAULT_EXPECTED_RUNS, _DEFAULT_STD) for i in range(_N_PLAYERS)
        ]
    finally:
        if _own_session and session is not None:
            session.close()

    # -- Vectorised simulation ------------------------------------------------
    # runs_matrix shape: (n_sims, n_players)
    scales_a = np.array([max(er / _GAMMA_SHAPE, 1e-3) for _, er, _ in players_a])
    scales_b = np.array([max(er / _GAMMA_SHAPE, 1e-3) for _, er, _ in players_b])

    runs_matrix_a = np.random.gamma(
        shape=_GAMMA_SHAPE,
        scale=scales_a[np.newaxis, :],   # broadcast over sims
        size=(n_sims, _N_PLAYERS),
    )
    runs_matrix_b = np.random.gamma(
        shape=_GAMMA_SHAPE,
        scale=scales_b[np.newaxis, :],
        size=(n_sims, _N_PLAYERS),
    )

    # Team score per simulation = sum of player runs
    team_scores_a = runs_matrix_a.sum(axis=1)   # shape (n_sims,)
    team_scores_b = runs_matrix_b.sum(axis=1)

    wins_a = (team_scores_a > team_scores_b).sum()
    win_prob_a = float(wins_a) / n_sims
    win_prob_b = 1.0 - win_prob_a

    return {
        "win_prob_a": round(win_prob_a, 4),
        "win_prob_b": round(win_prob_b, 4),
        "score_p10_a": float(np.percentile(team_scores_a, 10)),
        "score_p50_a": float(np.percentile(team_scores_a, 50)),
        "score_p90_a": float(np.percentile(team_scores_a, 90)),
        "score_p10_b": float(np.percentile(team_scores_b, 10)),
        "score_p50_b": float(np.percentile(team_scores_b, 50)),
        "score_p90_b": float(np.percentile(team_scores_b, 90)),
        "n_sims": n_sims,
    }


# ---------------------------------------------------------------------------
# Player impact (PVOR via simulation)
# ---------------------------------------------------------------------------

def compute_player_impact(
    player_name: str,
    team: str,
    opponent: str,
    format: str = "T20",
    session=None,
    n_sims: int = 1000,
) -> Dict:
    """
    Estimate a player's impact by comparing win probability with vs without them.

    Simulates the match twice:
    1. With the player (their actual expected runs from DB).
    2. With the player replaced by a generic replacement (expected_runs=20).

    Returns
    -------
    dict with keys:
        player_name, team, pvor_win_prob_delta, base_win_prob, replacement_win_prob
    """
    _own_session = False
    if session is None:
        try:
            from src.data.db import get_session
            session = get_session()
            _own_session = True
        except Exception as exc:
            logger.warning("compute_player_impact: could not open DB session: %s", exc)
            session = None

    try:
        # -- Base simulation --------------------------------------------------
        base_result = simulate_match(team, opponent, format=format, session=session, n_sims=n_sims)
        base_win_prob = base_result["win_prob_a"]  # team is treated as team_a

        # -- Replacement simulation -------------------------------------------
        # Find the player in the team's lineup and substitute with replacement
        players = get_player_expected_runs(team, format, session) if session else [
            (f"{team}_p{i}", _DEFAULT_EXPECTED_RUNS, _DEFAULT_STD) for i in range(_N_PLAYERS)
        ]

        # Find position of this player (by name)
        replacement_expected = 20.0
        player_found = False
        replacement_players = []
        for pname, er, std in players:
            if pname == player_name and not player_found:
                replacement_players.append((pname, replacement_expected, std))
                player_found = True
            else:
                replacement_players.append((pname, er, std))

        if not player_found:
            logger.debug(
                "compute_player_impact: player '%s' not found in %s lineup; "
                "adding as replacement at tail",
                player_name, team,
            )
            # Replace last player as approximation
            if replacement_players:
                replacement_players[-1] = (player_name, replacement_expected, _DEFAULT_STD)

        # Build opponents lineup for comparison
        opp_players = get_player_expected_runs(opponent, format, session) if session else [
            (f"{opponent}_p{i}", _DEFAULT_EXPECTED_RUNS, _DEFAULT_STD) for i in range(_N_PLAYERS)
        ]

        # Vectorised replacement simulation
        scales_team = np.array([max(er / _GAMMA_SHAPE, 1e-3) for _, er, _ in replacement_players])
        scales_opp = np.array([max(er / _GAMMA_SHAPE, 1e-3) for _, er, _ in opp_players])

        runs_team = np.random.gamma(
            _GAMMA_SHAPE, scales_team[np.newaxis, :], size=(n_sims, _N_PLAYERS)
        ).sum(axis=1)
        runs_opp = np.random.gamma(
            _GAMMA_SHAPE, scales_opp[np.newaxis, :], size=(n_sims, _N_PLAYERS)
        ).sum(axis=1)

        replacement_win_prob = float((runs_team > runs_opp).sum()) / n_sims

    finally:
        if _own_session and session is not None:
            session.close()

    delta = base_win_prob - replacement_win_prob

    return {
        "player_name": player_name,
        "team": team,
        "pvor_win_prob_delta": round(delta, 4),
        "base_win_prob": round(base_win_prob, 4),
        "replacement_win_prob": round(replacement_win_prob, 4),
    }
