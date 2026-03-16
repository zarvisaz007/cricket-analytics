"""
populate_real_data.py — Seeds the cricket DB with real match data (2020–2026).

Data sourced from:
  - ICC T20 World Cup 2021, 2022, 2024
  - ICC ODI World Cup 2023
  - ICC Champions Trophy 2025
  - ICC World Test Championship Finals 2021, 2023
  - Major bilateral series

Run: python scripts/populate_real_data.py
"""
from __future__ import annotations
import sys, os, json, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("populate")

from src.data.db import init_db, get_session, Match, Player, PlayerStat, Team

# ─────────────────────────────────────────────────────────────────────────────
# TEAMS
# ─────────────────────────────────────────────────────────────────────────────
TEAMS_DATA = [
    {"name": "India",        "country": "India",        "team_type": "international"},
    {"name": "Australia",    "country": "Australia",    "team_type": "international"},
    {"name": "England",      "country": "England",      "team_type": "international"},
    {"name": "Pakistan",     "country": "Pakistan",     "team_type": "international"},
    {"name": "South Africa", "country": "South Africa", "team_type": "international"},
    {"name": "New Zealand",  "country": "New Zealand",  "team_type": "international"},
    {"name": "West Indies",  "country": "West Indies",  "team_type": "international"},
    {"name": "Sri Lanka",    "country": "Sri Lanka",    "team_type": "international"},
    {"name": "Bangladesh",   "country": "Bangladesh",   "team_type": "international"},
    {"name": "Afghanistan",  "country": "Afghanistan",  "team_type": "international"},
    {"name": "Zimbabwe",     "country": "Zimbabwe",     "team_type": "international"},
    {"name": "Ireland",      "country": "Ireland",      "team_type": "international"},
    {"name": "Scotland",     "country": "Scotland",     "team_type": "international"},
    {"name": "Namibia",      "country": "Namibia",      "team_type": "international"},
    {"name": "Netherlands",  "country": "Netherlands",  "team_type": "international"},
    {"name": "UAE",          "country": "UAE",          "team_type": "international"},
    # IPL teams
    {"name": "Mumbai Indians",          "country": "India", "team_type": "ipl"},
    {"name": "Chennai Super Kings",     "country": "India", "team_type": "ipl"},
    {"name": "Royal Challengers Bangalore", "country": "India", "team_type": "ipl"},
    {"name": "Kolkata Knight Riders",   "country": "India", "team_type": "ipl"},
    {"name": "Delhi Capitals",          "country": "India", "team_type": "ipl"},
    {"name": "Rajasthan Royals",        "country": "India", "team_type": "ipl"},
    {"name": "Sunrisers Hyderabad",     "country": "India", "team_type": "ipl"},
    {"name": "Punjab Kings",            "country": "India", "team_type": "ipl"},
]

# ─────────────────────────────────────────────────────────────────────────────
# PLAYERS  (real stats as of early 2026)
# ─────────────────────────────────────────────────────────────────────────────
PLAYERS_DATA = [
    # India
    {"name": "Virat Kohli",    "country": "India",   "role": "batsman",      "batting_style": "Right-hand bat",  "bowling_style": "Right-arm medium",       "dob": "1988-11-05"},
    {"name": "Rohit Sharma",   "country": "India",   "role": "batsman",      "batting_style": "Right-hand bat",  "bowling_style": "Right-arm off-break",    "dob": "1987-04-30"},
    {"name": "Jasprit Bumrah", "country": "India",   "role": "bowler",       "batting_style": "Right-hand bat",  "bowling_style": "Right-arm fast",         "dob": "1993-12-06"},
    {"name": "Hardik Pandya",  "country": "India",   "role": "all-rounder",  "batting_style": "Right-hand bat",  "bowling_style": "Right-arm fast-medium",  "dob": "1993-10-11"},
    {"name": "KL Rahul",       "country": "India",   "role": "batsman",      "batting_style": "Right-hand bat",  "bowling_style": "Right-arm off-break",    "dob": "1992-04-18"},
    {"name": "Ravindra Jadeja","country": "India",   "role": "all-rounder",  "batting_style": "Left-hand bat",   "bowling_style": "Slow left-arm orthodox", "dob": "1988-12-06"},
    {"name": "Shubman Gill",   "country": "India",   "role": "batsman",      "batting_style": "Right-hand bat",  "bowling_style": "Right-arm off-break",    "dob": "1999-09-08"},
    {"name": "Rishabh Pant",   "country": "India",   "role": "wicket-keeper","batting_style": "Left-hand bat",   "bowling_style": None,                     "dob": "1997-10-04"},
    {"name": "Mohammed Siraj", "country": "India",   "role": "bowler",       "batting_style": "Right-hand bat",  "bowling_style": "Right-arm fast-medium",  "dob": "1994-03-13"},
    {"name": "Axar Patel",     "country": "India",   "role": "all-rounder",  "batting_style": "Left-hand bat",   "bowling_style": "Slow left-arm orthodox", "dob": "1994-01-20"},

    # Australia
    {"name": "Steve Smith",    "country": "Australia","role": "batsman",      "batting_style": "Right-hand bat",  "bowling_style": "Right-arm leg-break",    "dob": "1989-06-02"},
    {"name": "Pat Cummins",    "country": "Australia","role": "bowler",       "batting_style": "Right-hand bat",  "bowling_style": "Right-arm fast",         "dob": "1993-05-08"},
    {"name": "David Warner",   "country": "Australia","role": "batsman",      "batting_style": "Left-hand bat",   "bowling_style": "Right-arm off-break",    "dob": "1986-10-27"},
    {"name": "Travis Head",    "country": "Australia","role": "batsman",      "batting_style": "Left-hand bat",   "bowling_style": "Right-arm off-break",    "dob": "1993-12-29"},
    {"name": "Mitchell Starc", "country": "Australia","role": "bowler",       "batting_style": "Left-hand bat",   "bowling_style": "Left-arm fast",          "dob": "1990-01-30"},
    {"name": "Glenn Maxwell",  "country": "Australia","role": "all-rounder",  "batting_style": "Right-hand bat",  "bowling_style": "Right-arm off-break",    "dob": "1988-10-14"},
    {"name": "Marnus Labuschagne","country":"Australia","role":"batsman",     "batting_style": "Right-hand bat",  "bowling_style": "Right-arm leg-break",    "dob": "1994-06-22"},

    # England
    {"name": "Joe Root",       "country": "England", "role": "batsman",      "batting_style": "Right-hand bat",  "bowling_style": "Right-arm off-break",    "dob": "1990-12-30"},
    {"name": "Ben Stokes",     "country": "England", "role": "all-rounder",  "batting_style": "Left-hand bat",   "bowling_style": "Right-arm fast-medium",  "dob": "1991-06-04"},
    {"name": "Jos Buttler",    "country": "England", "role": "wicket-keeper","batting_style": "Right-hand bat",  "bowling_style": None,                     "dob": "1990-09-08"},
    {"name": "Jofra Archer",   "country": "England", "role": "bowler",       "batting_style": "Right-hand bat",  "bowling_style": "Right-arm fast",         "dob": "1995-04-01"},
    {"name": "Jonny Bairstow", "country": "England", "role": "wicket-keeper","batting_style": "Right-hand bat",  "bowling_style": None,                     "dob": "1989-09-26"},

    # Pakistan
    {"name": "Babar Azam",     "country": "Pakistan","role": "batsman",      "batting_style": "Right-hand bat",  "bowling_style": "Right-arm off-break",    "dob": "1994-10-15"},
    {"name": "Shaheen Afridi", "country": "Pakistan","role": "bowler",       "batting_style": "Left-hand bat",   "bowling_style": "Left-arm fast",          "dob": "2000-04-06"},
    {"name": "Mohammad Rizwan","country": "Pakistan","role": "wicket-keeper","batting_style": "Right-hand bat",  "bowling_style": None,                     "dob": "1992-06-01"},
    {"name": "Fakhar Zaman",   "country": "Pakistan","role": "batsman",      "batting_style": "Left-hand bat",   "bowling_style": "Right-arm off-break",    "dob": "1990-04-10"},

    # South Africa
    {"name": "Kagiso Rabada",  "country": "South Africa","role":"bowler",    "batting_style": "Right-hand bat",  "bowling_style": "Right-arm fast",         "dob": "1995-05-25"},
    {"name": "Quinton de Kock","country": "South Africa","role":"wicket-keeper","batting_style":"Left-hand bat", "bowling_style": None,                     "dob": "1992-12-17"},
    {"name": "Temba Bavuma",   "country": "South Africa","role":"batsman",   "batting_style": "Right-hand bat",  "bowling_style": "Right-arm medium",       "dob": "1990-05-17"},
    {"name": "Heinrich Klaasen","country":"South Africa","role":"batsman",   "batting_style": "Right-hand bat",  "bowling_style": None,                     "dob": "1991-07-08"},

    # New Zealand
    {"name": "Kane Williamson","country": "New Zealand","role":"batsman",    "batting_style": "Right-hand bat",  "bowling_style": "Right-arm off-break",    "dob": "1990-08-08"},
    {"name": "Trent Boult",    "country": "New Zealand","role":"bowler",     "batting_style": "Right-hand bat",  "bowling_style": "Left-arm fast-medium",   "dob": "1989-07-22"},
    {"name": "Devon Conway",   "country": "New Zealand","role":"batsman",    "batting_style": "Left-hand bat",   "bowling_style": None,                     "dob": "1991-07-08"},
    {"name": "Rachin Ravindra","country": "New Zealand","role":"all-rounder","batting_style": "Left-hand bat",   "bowling_style": "Slow left-arm orthodox", "dob": "2000-02-18"},

    # West Indies
    {"name": "Kieron Pollard", "country": "West Indies","role":"all-rounder","batting_style":"Right-hand bat",   "bowling_style": "Right-arm medium-fast",  "dob": "1987-05-12"},
    {"name": "Nicholas Pooran","country": "West Indies","role":"wicket-keeper","batting_style":"Left-hand bat",  "bowling_style": None,                     "dob": "1995-10-02"},

    # Sri Lanka
    {"name": "Wanindu Hasaranga","country":"Sri Lanka","role":"all-rounder", "batting_style": "Right-hand bat",  "bowling_style": "Right-arm leg-break",    "dob": "1997-07-29"},
    {"name": "Pathum Nissanka","country": "Sri Lanka","role":"batsman",      "batting_style": "Right-hand bat",  "bowling_style": "Right-arm medium",       "dob": "1998-02-15"},

    # Afghanistan
    {"name": "Rashid Khan",    "country": "Afghanistan","role":"all-rounder","batting_style":"Right-hand bat",   "bowling_style": "Right-arm leg-break",    "dob": "1998-09-20"},
    {"name": "Mohammad Nabi",  "country": "Afghanistan","role":"all-rounder","batting_style":"Right-hand bat",   "bowling_style": "Right-arm off-break",    "dob": "1985-01-01"},
    {"name": "Ibrahim Zadran", "country": "Afghanistan","role":"batsman",    "batting_style":"Right-hand bat",   "bowling_style": None,                     "dob": "2002-09-26"},

    # Bangladesh
    {"name": "Shakib Al Hasan","country": "Bangladesh","role":"all-rounder","batting_style": "Left-hand bat",   "bowling_style": "Slow left-arm orthodox", "dob": "1987-03-24"},
    {"name": "Litton Das",     "country": "Bangladesh","role":"wicket-keeper","batting_style":"Right-hand bat",  "bowling_style": None,                     "dob": "1994-10-13"},
]

# ─────────────────────────────────────────────────────────────────────────────
# REAL MATCHES  (2020–2026)
# ─────────────────────────────────────────────────────────────────────────────
MATCHES_DATA = [
    # ── ICC T20 World Cup 2021 (UAE) ─────────────────────────────────────────
    {"match_key":"t20wc21_001","team_a":"Pakistan","team_b":"India","venue":"Dubai International Cricket Stadium","match_date":"2021-10-24","match_type":"T20","tournament":"ICC T20 World Cup 2021","winner":"Pakistan","result_margin":"10 wickets","toss_winner":"Pakistan","toss_decision":"field","source":"real"},
    {"match_key":"t20wc21_002","team_a":"Australia","team_b":"South Africa","venue":"Sheikh Zayed Stadium, Abu Dhabi","match_date":"2021-10-23","match_type":"T20","tournament":"ICC T20 World Cup 2021","winner":"Australia","result_margin":"5 wickets","toss_winner":"Australia","toss_decision":"field","source":"real"},
    {"match_key":"t20wc21_003","team_a":"England","team_b":"West Indies","venue":"Dubai International Cricket Stadium","match_date":"2021-10-23","match_type":"T20","tournament":"ICC T20 World Cup 2021","winner":"England","result_margin":"6 wickets","toss_winner":"England","toss_decision":"field","source":"real"},
    {"match_key":"t20wc21_004","team_a":"New Zealand","team_b":"Pakistan","venue":"Sharjah Cricket Stadium","match_date":"2021-10-26","match_type":"T20","tournament":"ICC T20 World Cup 2021","winner":"Pakistan","result_margin":"5 wickets","toss_winner":"Pakistan","toss_decision":"field","source":"real"},
    {"match_key":"t20wc21_005","team_a":"India","team_b":"New Zealand","venue":"Dubai International Cricket Stadium","match_date":"2021-10-31","match_type":"T20","tournament":"ICC T20 World Cup 2021","winner":"New Zealand","result_margin":"8 wickets","toss_winner":"New Zealand","toss_decision":"field","source":"real"},
    {"match_key":"t20wc21_006","team_a":"Australia","team_b":"England","venue":"Dubai International Cricket Stadium","match_date":"2021-10-30","match_type":"T20","tournament":"ICC T20 World Cup 2021","winner":"England","result_margin":"8 wickets","toss_winner":"England","toss_decision":"field","source":"real"},
    {"match_key":"t20wc21_sf1","team_a":"Pakistan","team_b":"Australia","venue":"Dubai International Cricket Stadium","match_date":"2021-11-11","match_type":"T20","tournament":"ICC T20 World Cup 2021","winner":"Australia","result_margin":"5 wickets","toss_winner":"Pakistan","toss_decision":"bat","source":"real"},
    {"match_key":"t20wc21_sf2","team_a":"England","team_b":"New Zealand","venue":"Sheikh Zayed Stadium, Abu Dhabi","match_date":"2021-11-10","match_type":"T20","tournament":"ICC T20 World Cup 2021","winner":"England","result_margin":"5 wickets","toss_winner":"New Zealand","toss_decision":"bat","source":"real"},
    {"match_key":"t20wc21_final","team_a":"Australia","team_b":"New Zealand","venue":"Dubai International Cricket Stadium","match_date":"2021-11-14","match_type":"T20","tournament":"ICC T20 World Cup 2021","winner":"Australia","result_margin":"8 wickets","toss_winner":"Australia","toss_decision":"field","source":"real"},

    # ── ICC T20 World Cup 2022 (Australia) ───────────────────────────────────
    {"match_key":"t20wc22_001","team_a":"Australia","team_b":"New Zealand","venue":"Sydney Cricket Ground","match_date":"2022-10-22","match_type":"T20","tournament":"ICC T20 World Cup 2022","winner":"New Zealand","result_margin":"89 runs","toss_winner":"Australia","toss_decision":"bat","source":"real"},
    {"match_key":"t20wc22_002","team_a":"India","team_b":"Pakistan","venue":"Melbourne Cricket Ground","match_date":"2022-10-23","match_type":"T20","tournament":"ICC T20 World Cup 2022","winner":"India","result_margin":"4 wickets","toss_winner":"India","toss_decision":"field","source":"real"},
    {"match_key":"t20wc22_003","team_a":"England","team_b":"Afghanistan","venue":"Perth Stadium","match_date":"2022-10-22","match_type":"T20","tournament":"ICC T20 World Cup 2022","winner":"England","result_margin":"5 wickets","toss_winner":"England","toss_decision":"field","source":"real"},
    {"match_key":"t20wc22_004","team_a":"South Africa","team_b":"Bangladesh","venue":"Sydney Cricket Ground","match_date":"2022-10-27","match_type":"T20","tournament":"ICC T20 World Cup 2022","winner":"South Africa","result_margin":"104 runs","toss_winner":"South Africa","toss_decision":"bat","source":"real"},
    {"match_key":"t20wc22_005","team_a":"Pakistan","team_b":"India","venue":"Melbourne Cricket Ground","match_date":"2022-10-23","match_type":"T20","tournament":"ICC T20 World Cup 2022","winner":"India","result_margin":"4 wickets","toss_winner":"India","toss_decision":"field","source":"real"},
    {"match_key":"t20wc22_sf1","team_a":"India","team_b":"England","venue":"Adelaide Oval","match_date":"2022-11-10","match_type":"T20","tournament":"ICC T20 World Cup 2022","winner":"England","result_margin":"10 wickets","toss_winner":"India","toss_decision":"bat","source":"real"},
    {"match_key":"t20wc22_sf2","team_a":"Pakistan","team_b":"New Zealand","venue":"Sydney Cricket Ground","match_date":"2022-11-09","match_type":"T20","tournament":"ICC T20 World Cup 2022","winner":"New Zealand","result_margin":"7 wickets","toss_winner":"Pakistan","toss_decision":"bat","source":"real"},
    {"match_key":"t20wc22_final","team_a":"England","team_b":"Pakistan","venue":"Melbourne Cricket Ground","match_date":"2022-11-13","match_type":"T20","tournament":"ICC T20 World Cup 2022","winner":"England","result_margin":"5 wickets","toss_winner":"Pakistan","toss_decision":"bat","source":"real"},

    # ── ICC T20 World Cup 2024 (West Indies & USA) ───────────────────────────
    {"match_key":"t20wc24_001","team_a":"India","team_b":"Ireland","venue":"Nassau County International Cricket Stadium, New York","match_date":"2024-06-05","match_type":"T20","tournament":"ICC T20 World Cup 2024","winner":"India","result_margin":"8 wickets","toss_winner":"Ireland","toss_decision":"bat","source":"real"},
    {"match_key":"t20wc24_002","team_a":"India","team_b":"Pakistan","venue":"Nassau County International Cricket Stadium, New York","match_date":"2024-06-09","match_type":"T20","tournament":"ICC T20 World Cup 2024","winner":"India","result_margin":"6 runs","toss_winner":"India","toss_decision":"bat","source":"real"},
    {"match_key":"t20wc24_003","team_a":"India","team_b":"USA","venue":"Nassau County International Cricket Stadium, New York","match_date":"2024-06-12","match_type":"T20","tournament":"ICC T20 World Cup 2024","winner":"India","result_margin":"7 wickets","toss_winner":"USA","toss_decision":"bat","source":"real"},
    {"match_key":"t20wc24_004","team_a":"Australia","team_b":"England","venue":"Kensington Oval, Bridgetown","match_date":"2024-06-08","match_type":"T20","tournament":"ICC T20 World Cup 2024","winner":"Australia","result_margin":"36 runs","toss_winner":"Australia","toss_decision":"bat","source":"real"},
    {"match_key":"t20wc24_005","team_a":"Afghanistan","team_b":"New Zealand","venue":"Arnos Vale Ground, Kingstown","match_date":"2024-06-21","match_type":"T20","tournament":"ICC T20 World Cup 2024","winner":"Afghanistan","result_margin":"84 runs","toss_winner":"Afghanistan","toss_decision":"bat","source":"real"},
    {"match_key":"t20wc24_sf1","team_a":"India","team_b":"England","venue":"Providence Stadium, Guyana","match_date":"2024-06-27","match_type":"T20","tournament":"ICC T20 World Cup 2024","winner":"India","result_margin":"68 runs","toss_winner":"India","toss_decision":"bat","source":"real"},
    {"match_key":"t20wc24_sf2","team_a":"South Africa","team_b":"Afghanistan","venue":"Brian Lara Cricket Academy, Trinidad","match_date":"2024-06-26","match_type":"T20","tournament":"ICC T20 World Cup 2024","winner":"South Africa","result_margin":"9 wickets","toss_winner":"Afghanistan","toss_decision":"bat","source":"real"},
    {"match_key":"t20wc24_final","team_a":"India","team_b":"South Africa","venue":"Kensington Oval, Bridgetown","match_date":"2024-06-29","match_type":"T20","tournament":"ICC T20 World Cup 2024","winner":"India","result_margin":"7 runs","toss_winner":"India","toss_decision":"bat","source":"real"},

    # ── ICC ODI World Cup 2023 (India) ────────────────────────────────────────
    {"match_key":"odiwc23_001","team_a":"India","team_b":"Australia","venue":"MA Chidambaram Stadium, Chennai","match_date":"2023-10-08","match_type":"ODI","tournament":"ICC ODI World Cup 2023","winner":"India","result_margin":"6 wickets","toss_winner":"Australia","toss_decision":"bat","source":"real"},
    {"match_key":"odiwc23_002","team_a":"Pakistan","team_b":"Netherlands","venue":"Rajiv Gandhi International Stadium, Hyderabad","match_date":"2023-10-06","match_type":"ODI","tournament":"ICC ODI World Cup 2023","winner":"Pakistan","result_margin":"81 runs","toss_winner":"Pakistan","toss_decision":"bat","source":"real"},
    {"match_key":"odiwc23_003","team_a":"England","team_b":"New Zealand","venue":"Narendra Modi Stadium, Ahmedabad","match_date":"2023-10-05","match_type":"ODI","tournament":"ICC ODI World Cup 2023","winner":"New Zealand","result_margin":"9 wickets","toss_winner":"England","toss_decision":"bat","source":"real"},
    {"match_key":"odiwc23_004","team_a":"India","team_b":"Pakistan","venue":"Narendra Modi Stadium, Ahmedabad","match_date":"2023-10-14","match_type":"ODI","tournament":"ICC ODI World Cup 2023","winner":"India","result_margin":"7 wickets","toss_winner":"India","toss_decision":"field","source":"real"},
    {"match_key":"odiwc23_005","team_a":"South Africa","team_b":"Sri Lanka","venue":"Arun Jaitley Stadium, Delhi","match_date":"2023-10-07","match_type":"ODI","tournament":"ICC ODI World Cup 2023","winner":"South Africa","result_margin":"102 runs","toss_winner":"Sri Lanka","toss_decision":"field","source":"real"},
    {"match_key":"odiwc23_006","team_a":"India","team_b":"New Zealand","venue":"Himachal Pradesh Cricket Association Stadium, Dharamsala","match_date":"2023-10-22","match_type":"ODI","tournament":"ICC ODI World Cup 2023","winner":"India","result_margin":"4 wickets","toss_winner":"New Zealand","toss_decision":"bat","source":"real"},
    {"match_key":"odiwc23_007","team_a":"Afghanistan","team_b":"England","venue":"Arun Jaitley Stadium, Delhi","match_date":"2023-10-15","match_type":"ODI","tournament":"ICC ODI World Cup 2023","winner":"Afghanistan","result_margin":"69 runs","toss_winner":"England","toss_decision":"field","source":"real"},
    {"match_key":"odiwc23_sf1","team_a":"India","team_b":"New Zealand","venue":"Wankhede Stadium, Mumbai","match_date":"2023-11-15","match_type":"ODI","tournament":"ICC ODI World Cup 2023","winner":"India","result_margin":"70 runs","toss_winner":"India","toss_decision":"bat","source":"real"},
    {"match_key":"odiwc23_sf2","team_a":"Australia","team_b":"South Africa","venue":"Eden Gardens, Kolkata","match_date":"2023-11-16","match_type":"ODI","tournament":"ICC ODI World Cup 2023","winner":"Australia","result_margin":"3 wickets","toss_winner":"South Africa","toss_decision":"bat","source":"real"},
    {"match_key":"odiwc23_final","team_a":"India","team_b":"Australia","venue":"Narendra Modi Stadium, Ahmedabad","match_date":"2023-11-19","match_type":"ODI","tournament":"ICC ODI World Cup 2023","winner":"Australia","result_margin":"6 wickets","toss_winner":"Australia","toss_decision":"field","source":"real"},

    # ── ICC Champions Trophy 2025 (Pakistan/UAE) ─────────────────────────────
    {"match_key":"ct25_001","team_a":"Pakistan","team_b":"New Zealand","venue":"National Stadium, Karachi","match_date":"2025-02-19","match_type":"ODI","tournament":"ICC Champions Trophy 2025","winner":"Pakistan","result_margin":"7 wickets","toss_winner":"New Zealand","toss_decision":"bat","source":"real"},
    {"match_key":"ct25_002","team_a":"India","team_b":"Bangladesh","venue":"Dubai International Cricket Stadium","match_date":"2025-02-20","match_type":"ODI","tournament":"ICC Champions Trophy 2025","winner":"India","result_margin":"6 wickets","toss_winner":"Bangladesh","toss_decision":"bat","source":"real"},
    {"match_key":"ct25_003","team_a":"Australia","team_b":"England","venue":"Gaddafi Stadium, Lahore","match_date":"2025-02-22","match_type":"ODI","tournament":"ICC Champions Trophy 2025","winner":"Australia","result_margin":"7 wickets","toss_winner":"England","toss_decision":"bat","source":"real"},
    {"match_key":"ct25_004","team_a":"India","team_b":"Pakistan","venue":"Dubai International Cricket Stadium","match_date":"2025-02-23","match_type":"ODI","tournament":"ICC Champions Trophy 2025","winner":"India","result_margin":"6 wickets","toss_winner":"Pakistan","toss_decision":"bat","source":"real"},
    {"match_key":"ct25_005","team_a":"New Zealand","team_b":"Bangladesh","venue":"Rawalpindi Cricket Stadium","match_date":"2025-02-24","match_type":"ODI","tournament":"ICC Champions Trophy 2025","winner":"New Zealand","result_margin":"5 wickets","toss_winner":"Bangladesh","toss_decision":"bat","source":"real"},
    {"match_key":"ct25_006","team_a":"South Africa","team_b":"Afghanistan","venue":"National Stadium, Karachi","match_date":"2025-02-25","match_type":"ODI","tournament":"ICC Champions Trophy 2025","winner":"South Africa","result_margin":"7 wickets","toss_winner":"Afghanistan","toss_decision":"bat","source":"real"},
    {"match_key":"ct25_007","team_a":"India","team_b":"New Zealand","venue":"Dubai International Cricket Stadium","match_date":"2025-03-02","match_type":"ODI","tournament":"ICC Champions Trophy 2025","winner":"India","result_margin":"7 wickets","toss_winner":"New Zealand","toss_decision":"bat","source":"real"},
    {"match_key":"ct25_sf1","team_a":"India","team_b":"Australia","venue":"Dubai International Cricket Stadium","match_date":"2025-03-04","match_type":"ODI","tournament":"ICC Champions Trophy 2025","winner":"India","result_margin":"4 wickets","toss_winner":"Australia","toss_decision":"bat","source":"real"},
    {"match_key":"ct25_sf2","team_a":"New Zealand","team_b":"South Africa","venue":"Gaddafi Stadium, Lahore","match_date":"2025-03-05","match_type":"ODI","tournament":"ICC Champions Trophy 2025","winner":"New Zealand","result_margin":"50 runs","toss_winner":"New Zealand","toss_decision":"bat","source":"real"},
    {"match_key":"ct25_final","team_a":"India","team_b":"New Zealand","venue":"Dubai International Cricket Stadium","match_date":"2025-03-09","match_type":"ODI","tournament":"ICC Champions Trophy 2025","winner":"India","result_margin":"4 wickets","toss_winner":"New Zealand","toss_decision":"bat","source":"real"},

    # ── ICC World Test Championship Finals ───────────────────────────────────
    {"match_key":"wtc21_final","team_a":"New Zealand","team_b":"India","venue":"Hampshire Bowl, Southampton","match_date":"2021-06-18","match_type":"Test","tournament":"ICC World Test Championship Final 2021","winner":"New Zealand","result_margin":"8 wickets","toss_winner":"New Zealand","toss_decision":"bat","source":"real"},
    {"match_key":"wtc23_final","team_a":"Australia","team_b":"India","venue":"The Oval, London","match_date":"2023-06-07","match_type":"Test","tournament":"ICC World Test Championship Final 2023","winner":"Australia","result_margin":"209 runs","toss_winner":"Australia","toss_decision":"bat","source":"real"},

    # ── The Ashes 2021-22 (Australia) ─────────────────────────────────────────
    {"match_key":"ashes2122_001","team_a":"Australia","team_b":"England","venue":"The Gabba, Brisbane","match_date":"2021-12-08","match_type":"Test","tournament":"The Ashes 2021-22","winner":"Australia","result_margin":"9 wickets","toss_winner":"Australia","toss_decision":"bat","source":"real"},
    {"match_key":"ashes2122_002","team_a":"Australia","team_b":"England","venue":"Adelaide Oval","match_date":"2021-12-16","match_type":"Test","tournament":"The Ashes 2021-22","winner":"Australia","result_margin":"275 runs","toss_winner":"England","toss_decision":"bat","source":"real"},
    {"match_key":"ashes2122_003","team_a":"Australia","team_b":"England","venue":"Melbourne Cricket Ground","match_date":"2021-12-26","match_type":"Test","tournament":"The Ashes 2021-22","winner":"Australia","result_margin":"Innings & 14 runs","toss_winner":"Australia","toss_decision":"bat","source":"real"},

    # ── The Ashes 2023 (England) ──────────────────────────────────────────────
    {"match_key":"ashes23_001","team_a":"England","team_b":"Australia","venue":"Edgbaston, Birmingham","match_date":"2023-06-16","match_type":"Test","tournament":"The Ashes 2023","winner":"England","result_margin":"2 wickets","toss_winner":"England","toss_decision":"bat","source":"real"},
    {"match_key":"ashes23_002","team_a":"England","team_b":"Australia","venue":"Lord's Cricket Ground, London","match_date":"2023-06-28","match_type":"Test","tournament":"The Ashes 2023","winner":"Australia","result_margin":"43 runs","toss_winner":"Australia","toss_decision":"bat","source":"real"},
    {"match_key":"ashes23_003","team_a":"England","team_b":"Australia","venue":"Headingley, Leeds","match_date":"2023-07-06","match_type":"Test","tournament":"The Ashes 2023","winner":"England","result_margin":"3 wickets","toss_winner":"Australia","toss_decision":"bat","source":"real"},

    # ── India bilateral series ────────────────────────────────────────────────
    {"match_key":"ind_aus_t20_2022_01","team_a":"India","team_b":"Australia","venue":"VCA Stadium, Nagpur","match_date":"2022-09-20","match_type":"T20","tournament":"India vs Australia T20 Series 2022","winner":"India","result_margin":"6 wickets","toss_winner":"Australia","toss_decision":"bat","source":"real"},
    {"match_key":"ind_sa_t20_2022_01","team_a":"India","team_b":"South Africa","venue":"Arun Jaitley Stadium, Delhi","match_date":"2022-09-28","match_type":"T20","tournament":"India vs South Africa T20 Series 2022","winner":"South Africa","result_margin":"7 wickets","toss_winner":"South Africa","toss_decision":"field","source":"real"},
    {"match_key":"ind_sl_odi_2023_01","team_a":"India","team_b":"Sri Lanka","venue":"Rajiv Gandhi International Stadium, Hyderabad","match_date":"2023-01-10","match_type":"ODI","tournament":"India vs Sri Lanka ODI Series 2023","winner":"India","result_margin":"67 runs","toss_winner":"India","toss_decision":"bat","source":"real"},
    {"match_key":"ind_nz_t20_2023_01","team_a":"India","team_b":"New Zealand","venue":"Sawai Mansingh Stadium, Jaipur","match_date":"2023-01-27","match_type":"T20","tournament":"India vs New Zealand T20 Series 2023","winner":"India","result_margin":"21 runs","toss_winner":"India","toss_decision":"bat","source":"real"},
    {"match_key":"ind_eng_test_2024_01","team_a":"India","team_b":"England","venue":"Rajiv Gandhi International Stadium, Hyderabad","match_date":"2024-01-25","match_type":"Test","tournament":"India vs England Test Series 2024","winner":"England","result_margin":"28 runs","toss_winner":"India","toss_decision":"bat","source":"real"},
    {"match_key":"ind_eng_test_2024_02","team_a":"India","team_b":"England","venue":"ACA-VDCA Cricket Stadium, Visakhapatnam","match_date":"2024-02-02","match_type":"Test","tournament":"India vs England Test Series 2024","winner":"India","result_margin":"106 runs","toss_winner":"England","toss_decision":"bat","source":"real"},
    {"match_key":"ind_eng_test_2024_03","team_a":"India","team_b":"England","venue":"Saurashtra Cricket Association Stadium, Rajkot","match_date":"2024-02-15","match_type":"Test","tournament":"India vs England Test Series 2024","winner":"India","result_margin":"434 runs","toss_winner":"India","toss_decision":"bat","source":"real"},
    {"match_key":"ind_eng_test_2024_04","team_a":"India","team_b":"England","venue":"JSCA International Stadium Complex, Ranchi","match_date":"2024-02-23","match_type":"Test","tournament":"India vs England Test Series 2024","winner":"Draw","result_margin":"Draw","toss_winner":"India","toss_decision":"bat","source":"real"},
    {"match_key":"ind_eng_test_2024_05","team_a":"India","team_b":"England","venue":"MA Chidambaram Stadium, Chennai","match_date":"2024-03-07","match_type":"Test","tournament":"India vs England Test Series 2024","winner":"India","result_margin":"Innings & 64 runs","toss_winner":"India","toss_decision":"bat","source":"real"},

    # ── Asia Cup 2022 & 2023 ──────────────────────────────────────────────────
    {"match_key":"ac22_001","team_a":"India","team_b":"Pakistan","venue":"Dubai International Cricket Stadium","match_date":"2022-08-28","match_type":"T20","tournament":"Asia Cup 2022","winner":"India","result_margin":"5 wickets","toss_winner":"India","toss_decision":"field","source":"real"},
    {"match_key":"ac22_final","team_a":"Sri Lanka","team_b":"Pakistan","venue":"Dubai International Cricket Stadium","match_date":"2022-09-11","match_type":"T20","tournament":"Asia Cup 2022","winner":"Sri Lanka","result_margin":"23 runs","toss_winner":"Pakistan","toss_decision":"field","source":"real"},
    {"match_key":"ac23_001","team_a":"India","team_b":"Pakistan","venue":"Pallekele International Cricket Stadium","match_date":"2023-09-02","match_type":"ODI","tournament":"Asia Cup 2023","winner":"India","result_margin":"228 runs","toss_winner":"Pakistan","toss_decision":"bat","source":"real"},
    {"match_key":"ac23_final","team_a":"India","team_b":"Sri Lanka","venue":"R.Premadasa Stadium, Colombo","match_date":"2023-09-17","match_type":"ODI","tournament":"Asia Cup 2023","winner":"India","result_margin":"10 wickets","toss_winner":"India","toss_decision":"field","source":"real"},

    # ── BGT 2024-25 (Australia vs India) ─────────────────────────────────────
    {"match_key":"bgt2425_001","team_a":"Australia","team_b":"India","venue":"Perth Stadium","match_date":"2024-11-22","match_type":"Test","tournament":"Border-Gavaskar Trophy 2024-25","winner":"India","result_margin":"295 runs","toss_winner":"Australia","toss_decision":"bat","source":"real"},
    {"match_key":"bgt2425_002","team_a":"Australia","team_b":"India","venue":"Adelaide Oval","match_date":"2024-12-06","match_type":"Test","tournament":"Border-Gavaskar Trophy 2024-25","winner":"Australia","result_margin":"10 wickets","toss_winner":"India","toss_decision":"bat","source":"real"},
    {"match_key":"bgt2425_003","team_a":"Australia","team_b":"India","venue":"The Gabba, Brisbane","match_date":"2024-12-14","match_type":"Test","tournament":"Border-Gavaskar Trophy 2024-25","winner":"Draw","result_margin":"Draw","toss_winner":"Australia","toss_decision":"bat","source":"real"},
    {"match_key":"bgt2425_004","team_a":"Australia","team_b":"India","venue":"Melbourne Cricket Ground","match_date":"2024-12-26","match_type":"Test","tournament":"Border-Gavaskar Trophy 2024-25","winner":"Australia","result_margin":"184 runs","toss_winner":"India","toss_decision":"field","source":"real"},
    {"match_key":"bgt2425_005","team_a":"Australia","team_b":"India","venue":"Sydney Cricket Ground","match_date":"2025-01-03","match_type":"Test","tournament":"Border-Gavaskar Trophy 2024-25","winner":"Australia","result_margin":"6 wickets","toss_winner":"Australia","toss_decision":"bat","source":"real"},
]

# ─────────────────────────────────────────────────────────────────────────────
# SEEDING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def seed_teams(session) -> dict:
    name_to_id = {}
    new_count = 0
    for t in TEAMS_DATA:
        existing = session.query(Team).filter_by(name=t["name"]).first()
        if not existing:
            obj = Team(**t)
            session.add(obj)
            session.flush()
            name_to_id[t["name"]] = obj.id
            new_count += 1
        else:
            name_to_id[t["name"]] = existing.id
    session.commit()
    log.info("Teams: %d new / %d total", new_count, len(TEAMS_DATA))
    return name_to_id


def seed_players(session) -> dict:
    name_to_id = {}
    new_count = 0
    for p in PLAYERS_DATA:
        existing = session.query(Player).filter_by(name=p["name"]).first()
        if not existing:
            obj = Player(**p)
            session.add(obj)
            session.flush()
            name_to_id[p["name"]] = obj.id
            new_count += 1
        else:
            name_to_id[p["name"]] = existing.id
    session.commit()
    log.info("Players: %d new / %d total", new_count, len(PLAYERS_DATA))
    return name_to_id


def seed_matches(session) -> int:
    new_count = 0
    for m in MATCHES_DATA:
        existing = session.query(Match).filter_by(match_key=m["match_key"]).first()
        if not existing:
            obj = Match(**m)
            session.add(obj)
            new_count += 1
        else:
            # Update winner / source if changed
            existing.winner = m.get("winner", existing.winner)
            existing.source = m.get("source", existing.source)
    session.commit()
    log.info("Matches: %d new / %d total", new_count, len(MATCHES_DATA))
    return new_count


def seed_player_stats(session, player_ids: dict, match_ids: dict) -> None:
    """Insert sample real-ish stats for key players in key matches."""
    # Virat Kohli in T20 WC 2024 Final
    vid = player_ids.get("Virat Kohli")
    mid = match_ids.get("t20wc24_final")
    if vid and mid:
        exists = session.query(PlayerStat).filter_by(player_id=vid, match_id=mid).first()
        if not exists:
            session.add(PlayerStat(
                player_id=vid, match_id=mid, team="India",
                runs=76, balls_faced=59, fours=6, sixes=2,
                strike_rate=128.81, wickets=0, overs_bowled=0,
                runs_conceded=0, economy_rate=0, catches=1,
            ))

    # Jasprit Bumrah in T20 WC 2024 Final
    bid = player_ids.get("Jasprit Bumrah")
    if bid and mid:
        exists = session.query(PlayerStat).filter_by(player_id=bid, match_id=mid).first()
        if not exists:
            session.add(PlayerStat(
                player_id=bid, match_id=mid, team="India",
                runs=0, balls_faced=3, fours=0, sixes=0, strike_rate=0,
                wickets=2, overs_bowled=4.0, runs_conceded=18, economy_rate=4.5, catches=0,
            ))

    # Travis Head in WTC 2023 Final
    thid = player_ids.get("Travis Head")
    wtc_mid = match_ids.get("wtc23_final")
    if thid and wtc_mid:
        exists = session.query(PlayerStat).filter_by(player_id=thid, match_id=wtc_mid).first()
        if not exists:
            session.add(PlayerStat(
                player_id=thid, match_id=wtc_mid, team="Australia",
                runs=163, balls_faced=174, fours=17, sixes=4,
                strike_rate=93.68, wickets=0, overs_bowled=0, runs_conceded=0,
                economy_rate=0, catches=1,
            ))

    # Steve Smith in WTC 2023 Final
    ssid = player_ids.get("Steve Smith")
    if ssid and wtc_mid:
        exists = session.query(PlayerStat).filter_by(player_id=ssid, match_id=wtc_mid).first()
        if not exists:
            session.add(PlayerStat(
                player_id=ssid, match_id=wtc_mid, team="Australia",
                runs=121, balls_faced=293, fours=10, sixes=1,
                strike_rate=41.30, wickets=0, overs_bowled=0, runs_conceded=0,
                economy_rate=0, catches=2,
            ))

    session.commit()
    log.info("Player stats seeded for key matches.")


def setup_auto_update():
    """Create a cron schedule hint file for daily DB updates."""
    hint = {
        "cron": "0 6 * * *",
        "command": "python scripts/update_matches.py",
        "description": "Daily cricket data updater — fetches new match results",
        "last_run": None,
    }
    import json
    with open("run/update_schedule.json", "w") as f:
        json.dump(hint, f, indent=2)
    log.info("Auto-update schedule written to run/update_schedule.json")
    log.info("Add to crontab: 0 6 * * * cd $(pwd) && python scripts/update_matches.py")


def main():
    log.info("=== Populating cricket DB with real historical data (2020–2026) ===")
    init_db()
    session = get_session()

    try:
        team_ids = seed_teams(session)
        player_ids = seed_players(session)
        seed_matches(session)

        # Build match_key -> id map for stats
        match_ids = {}
        for m in session.query(Match).all():
            match_ids[m.match_key] = m.id

        seed_player_stats(session, player_ids, match_ids)
        setup_auto_update()

        total_matches = session.query(Match).count()
        total_players = session.query(Player).count()
        total_teams = session.query(Team).count()
        total_stats = session.query(PlayerStat).count()

        print("\n" + "="*55)
        print("  Cricket DB seeded successfully!")
        print(f"  Teams:        {total_teams}")
        print(f"  Players:      {total_players}")
        print(f"  Matches:      {total_matches}  (2020–2026)")
        print(f"  Player stats: {total_stats}")
        print("="*55)
        print("\nTournaments included:")
        print("  • ICC T20 World Cup 2021 (UAE)")
        print("  • ICC T20 World Cup 2022 (Australia)")
        print("  • ICC T20 World Cup 2024 (WI & USA)")
        print("  • ICC ODI World Cup 2023 (India)")
        print("  • ICC Champions Trophy 2025 (Pakistan/UAE)")
        print("  • ICC WTC Finals 2021 & 2023")
        print("  • The Ashes 2021-22 & 2023")
        print("  • Border-Gavaskar Trophy 2024-25")
        print("  • Asia Cup 2022 & 2023")
        print("  • Major bilateral series\n")
    finally:
        session.close()


if __name__ == "__main__":
    main()
