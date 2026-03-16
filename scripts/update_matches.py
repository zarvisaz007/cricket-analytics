"""
update_matches.py — Daily updater: checks for new match results and adds them to DB.

Crontab: 0 6 * * * cd /path/to/Claude-cricket && python scripts/update_matches.py

Providers (in priority order):
1. CricAPI (if CRICKET_API_KEY is set)
2. Mock generator (new synthetic matches beyond last DB date)
"""
from __future__ import annotations
import sys, os, json, logging
from datetime import datetime, timezone, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

LOG_FILE = "./logs/agents.log"
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] update_matches — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("update_matches")

SCHEDULE_FILE = "run/update_schedule.json"
CRICKET_API_KEY = os.getenv("CRICKET_API_KEY", "")
CRICKET_API_PROVIDER = os.getenv("CRICKET_API_PROVIDER", "mock")


def _get_last_run() -> str | None:
    try:
        with open(SCHEDULE_FILE) as f:
            return json.load(f).get("last_run")
    except Exception:
        return None


def _set_last_run() -> None:
    os.makedirs("run", exist_ok=True)
    data = {}
    try:
        with open(SCHEDULE_FILE) as f:
            data = json.load(f)
    except Exception:
        pass
    data["last_run"] = datetime.now(timezone.utc).isoformat()
    with open(SCHEDULE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _fetch_new_via_cricapi(since: str) -> list:
    """
    TODO: Implement real CricAPI fetch for matches since *since* date.
    Endpoint: https://api.cricapi.com/v1/matches?apikey={CRICKET_API_KEY}&offset=0
    Replace this stub when CRICKET_API_KEY is available.
    """
    log.warning("CricAPI fetch not yet implemented — using mock. "
                "Set CRICKET_API_KEY and implement this function.")
    return []


def _generate_new_mock_matches(since: str) -> list:
    """Generate a small batch of synthetic 'new' matches beyond *since* date."""
    import random
    from scripts.populate_real_data import TEAMS_DATA, VENUES

    teams = [t["name"] for t in TEAMS_DATA if t["team_type"] == "international"]
    venues = [
        "Eden Gardens, Kolkata", "MCG, Melbourne", "Lord's, London",
        "Nassau County Stadium, New York", "Dubai International Cricket Stadium",
        "National Stadium, Karachi",
    ]

    try:
        since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
    except Exception:
        since_dt = datetime.now(timezone.utc) - timedelta(days=1)

    matches = []
    random.seed(int(datetime.now().timestamp()))
    for i in range(random.randint(1, 4)):
        match_date = (since_dt + timedelta(days=i + 1)).strftime("%Y-%m-%d")
        ta = random.choice(teams)
        tb = random.choice([t for t in teams if t != ta])
        matches.append({
            "match_key": f"auto_{match_date.replace('-','')}_{i:02d}",
            "team_a": ta,
            "team_b": tb,
            "venue": random.choice(venues),
            "match_date": match_date,
            "match_type": random.choice(["T20", "ODI", "Test"]),
            "tournament": "Bilateral Series",
            "winner": random.choice([ta, tb]),
            "result_margin": f"{random.randint(1, 150)} runs",
            "toss_winner": random.choice([ta, tb]),
            "toss_decision": random.choice(["bat", "field"]),
            "source": "auto_generated",
        })
    return matches


def main() -> None:
    from src.data.db import init_db, get_session, Match

    log.info("=== update_matches.py starting ===")
    init_db()

    # Determine since-date
    last_run = _get_last_run()
    session = get_session()
    latest_match = session.query(Match).order_by(Match.match_date.desc()).first()
    since = (
        latest_match.match_date
        if latest_match and latest_match.match_date
        else (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    )
    log.info("Fetching matches since: %s", since)

    # Fetch
    if CRICKET_API_KEY and CRICKET_API_PROVIDER == "cricapi":
        new_matches = _fetch_new_via_cricapi(since)
    else:
        new_matches = _generate_new_mock_matches(since)

    inserted = 0
    for m in new_matches:
        existing = session.query(Match).filter_by(match_key=m["match_key"]).first()
        if not existing:
            session.add(Match(**m))
            inserted += 1
            log.info("  + %s: %s vs %s (%s)", m["match_date"], m["team_a"], m["team_b"], m["winner"])
    session.commit()
    session.close()

    _set_last_run()
    log.info("Update complete: %d new matches added.", inserted)
    print(f"update_matches: {inserted} new matches added (since {since})")


if __name__ == "__main__":
    main()
