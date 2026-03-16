"""
cricsheet.py — Downloads and parses cricket data from cricsheet.org.

Cricsheet provides free, ball-by-ball data for all major international
matches and IPL as downloadable ZIP files — no API key, no scraping,
no bot detection. This is the primary historical data source.

Data: https://cricsheet.org/downloads/
"""
from __future__ import annotations

import csv
import io
import json
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterator, Dict, Optional

import requests

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/cricsheet_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cricsheet CSV2 format download URLs (all-time, updated weekly)
DOWNLOADS = {
    "t20i":  "https://cricsheet.org/downloads/t20s_male_csv2.zip",
    "odi":   "https://cricsheet.org/downloads/odis_male_csv2.zip",
    "test":  "https://cricsheet.org/downloads/tests_male_csv2.zip",
    "ipl":   "https://cricsheet.org/downloads/ipl_male_csv2.zip",
}

FORMAT_MAP = {"t20i": "T20", "odi": "ODI", "test": "Test", "ipl": "T20"}


def _download_zip(url: str, cache_path: Path) -> Optional[bytes]:
    """Download a zip file, using a cached copy if it exists and is < 24h old."""
    if cache_path.exists():
        age_hours = (datetime.now().timestamp() - cache_path.stat().st_mtime) / 3600
        if age_hours < 24:
            log.info("[cricsheet] Using cached %s (%.1fh old)", cache_path.name, age_hours)
            return cache_path.read_bytes()

    log.info("[cricsheet] Downloading %s …", url)
    try:
        resp = requests.get(url, timeout=120, headers={"User-Agent": "cricket-analytics/1.0"})
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        return resp.content
    except Exception as exc:
        log.error("[cricsheet] Download failed for %s: %s", url, exc)
        if cache_path.exists():
            log.warning("[cricsheet] Falling back to stale cache for %s", cache_path.name)
            return cache_path.read_bytes()
        return None


def iter_matches(formats: list = None) -> Iterator[Dict]:
    """
    Yield match dicts (one per match) from Cricsheet CSV2 format.

    Each yielded dict has keys matching the `matches` DB table plus
    a `deliveries` key with a list of ball-by-ball dicts.
    """
    if formats is None:
        formats = list(DOWNLOADS.keys())

    for fmt_key in formats:
        url = DOWNLOADS.get(fmt_key)
        if not url:
            continue

        cache_path = CACHE_DIR / f"{fmt_key}.zip"
        raw = _download_zip(url, cache_path)
        if not raw:
            continue

        fmt_label = FORMAT_MAP[fmt_key]
        yield from _parse_zip(raw, fmt_label)


def _parse_zip(raw: bytes, fmt_label: str) -> Iterator[Dict]:
    """Parse a Cricsheet CSV2 zip and yield one match dict per match."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except Exception as exc:
        log.error("[cricsheet] Zip parse error: %s", exc)
        return

    # CSV2 format: one CSV per match named <matchid>_info.csv and <matchid>.csv
    # Group files by match id
    info_files = {n.replace("_info.csv", ""): n for n in zf.namelist() if n.endswith("_info.csv")}
    ball_files = {n.replace(".csv", ""): n for n in zf.namelist()
                  if n.endswith(".csv") and not n.endswith("_info.csv")}

    for match_id, info_name in info_files.items():
        try:
            info = _parse_info_csv(zf.read(info_name).decode("utf-8"))
            deliveries = []
            if match_id in ball_files:
                deliveries = _parse_ball_csv(zf.read(ball_files[match_id]).decode("utf-8"))

            match = {
                "match_key":     f"cricsheet_{match_id}",
                "match_type":    fmt_label,
                "match_date":    info.get("date", ""),
                "venue":         info.get("venue", ""),
                "team_a":        info.get("team1", ""),
                "team_b":        info.get("team2", ""),
                "winner":        info.get("winner", None),
                "result_margin": info.get("winner_runs") or info.get("winner_wickets") or "",
                "toss_winner":   info.get("toss_winner", ""),
                "toss_decision": info.get("toss_decision", ""),
                "tournament":    info.get("event", fmt_label),
                "source":        "cricsheet",
                "deliveries":    deliveries,
                "cricsheet_id":  match_id,
            }
            yield match
        except Exception as exc:
            log.debug("[cricsheet] Skipped match %s: %s", match_id, exc)


def _parse_info_csv(text: str) -> Dict:
    """Parse a Cricsheet _info.csv file into a flat dict."""
    info: Dict = {}
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        if len(row) >= 3 and row[0] == "info":
            key, val = row[1], row[2]
            if key == "dates":
                info["date"] = val
            elif key == "teams":
                if "team1" not in info:
                    info["team1"] = val
                else:
                    info["team2"] = val
            else:
                info[key] = val
    return info


def _parse_ball_csv(text: str) -> list:
    """Parse a Cricsheet ball CSV into a list of delivery dicts."""
    deliveries = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            over_str = row.get("ball", "0.0")
            parts = over_str.split(".")
            over_num = int(parts[0])
            ball_num = int(parts[1]) if len(parts) > 1 else 0
            deliveries.append({
                "innings_number": int(row.get("innings", 1)),
                "over_number":    over_num,
                "ball_number":    ball_num,
                "batsman_name":   row.get("striker", ""),
                "bowler_name":    row.get("bowler", ""),
                "runs_scored":    int(row.get("runs_off_bat", 0)),
                "extras":         int(row.get("extras", 0)),
                "extra_type":     row.get("wides", "") and "wide" or row.get("noballs", "") and "nb" or None,
                "wicket_type":    row.get("wicket_type", "") or None,
                "fielder_name":   row.get("other_player_dismissed", "") or None,
                "is_boundary":    int(row.get("runs_off_bat", 0)) == 4,
                "is_six":         int(row.get("runs_off_bat", 0)) == 6,
            })
        except Exception:
            continue
    return deliveries


def ingest_to_db(formats: list = None, session=None, limit: int = None) -> int:
    """
    Download Cricsheet data and upsert into the database.

    Parameters
    ----------
    formats : list, optional
        Subset of ['t20i','odi','test','ipl']. Defaults to all.
    session : SQLAlchemy session, optional
        Creates its own if not provided.
    limit : int, optional
        Stop after this many matches (useful for quick tests).

    Returns
    -------
    int
        Number of new matches inserted.
    """
    import difflib
    from src.data.db import Match, Player, Innings, Delivery, PlayerStat, get_session

    own_session = session is None
    if own_session:
        session = get_session()

    new_count = 0
    try:
        # Build player name cache
        player_cache: Dict[str, int] = {
            p.name: p.id for p in session.query(Player).all()
        }

        total = 0
        for m in iter_matches(formats):
            total += 1
            if limit and total > limit:
                break

            # ── Upsert Match ──────────────────────────────────────────────
            existing = session.query(Match).filter_by(match_key=m["match_key"]).first()
            if existing:
                continue

            deliveries_data = m.pop("deliveries", [])
            m.pop("cricsheet_id", None)

            match_obj = Match(**{k: v for k, v in m.items()
                                  if k in Match.__table__.columns.keys()})
            session.add(match_obj)
            session.flush()  # get match_obj.id

            # ── Build innings summaries from deliveries ───────────────────
            innings_map: Dict[int, Dict] = {}
            for d in deliveries_data:
                inn_num = d["innings_number"]
                if inn_num not in innings_map:
                    # Determine batting team from innings number
                    batting = m["team_a"] if inn_num % 2 == 1 else m["team_b"]
                    bowling = m["team_b"] if inn_num % 2 == 1 else m["team_a"]
                    innings_map[inn_num] = {
                        "batting_team": batting, "bowling_team": bowling,
                        "total_runs": 0, "total_wickets": 0, "total_overs": 0.0,
                    }
                innings_map[inn_num]["total_runs"] += d["runs_scored"] + d["extras"]
                if d["wicket_type"]:
                    innings_map[inn_num]["total_wickets"] += 1

            innings_objs: Dict[int, object] = {}
            for inn_num, summary in innings_map.items():
                inn_obj = Innings(
                    match_id=match_obj.id,
                    innings_number=inn_num,
                    **summary,
                )
                session.add(inn_obj)
                session.flush()
                innings_objs[inn_num] = inn_obj

            # ── Insert Deliveries ─────────────────────────────────────────
            for d in deliveries_data:
                inn_obj = innings_objs.get(d["innings_number"])
                if not inn_obj:
                    continue

                def _resolve(name: str) -> Optional[int]:
                    if not name:
                        return None
                    if name in player_cache:
                        return player_cache[name]
                    close = difflib.get_close_matches(name, player_cache.keys(), n=1, cutoff=0.8)
                    if close:
                        return player_cache[close[0]]
                    # Create minimal player record
                    p = Player(name=name)
                    session.add(p)
                    session.flush()
                    player_cache[name] = p.id
                    return p.id

                delivery = Delivery(
                    innings_id=inn_obj.id,
                    over_number=d["over_number"],
                    ball_number=d["ball_number"],
                    batsman_id=_resolve(d["batsman_name"]),
                    bowler_id=_resolve(d["bowler_name"]),
                    batsman_name=d["batsman_name"],
                    bowler_name=d["bowler_name"],
                    runs_scored=d["runs_scored"],
                    extras=d["extras"],
                    extra_type=d["extra_type"],
                    wicket_type=d["wicket_type"],
                    fielder_name=d["fielder_name"],
                    is_boundary=d["is_boundary"],
                    is_six=d["is_six"],
                )
                session.add(delivery)

            session.commit()
            new_count += 1

            if new_count % 100 == 0:
                log.info("[cricsheet] %d new matches ingested…", new_count)

    except Exception as exc:
        session.rollback()
        log.error("[cricsheet] Ingest error: %s", exc)
    finally:
        if own_session:
            session.close()

    log.info("[cricsheet] Ingest complete — %d new matches.", new_count)
    return new_count
