-- Cricket Analytics DB Schema
-- Run via: sqlite3 data/cricket.db < src/data/schema.sql

CREATE TABLE IF NOT EXISTS teams (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    country     TEXT,
    team_type   TEXT DEFAULT 'international',  -- international | ipl | domestic
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS players (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    country         TEXT,
    role            TEXT,   -- batsman | bowler | all-rounder | wicket-keeper
    batting_style   TEXT,
    bowling_style   TEXT,
    dob             TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS matches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    match_key       TEXT UNIQUE,            -- external identifier
    team_a          TEXT NOT NULL,
    team_b          TEXT NOT NULL,
    venue           TEXT,
    match_date      TEXT,
    match_type      TEXT,                   -- T20 | ODI | Test
    tournament      TEXT,
    winner          TEXT,
    result_margin   TEXT,
    toss_winner     TEXT,
    toss_decision   TEXT,
    scorecard_json  TEXT,                   -- full scorecard as JSON string
    source          TEXT DEFAULT 'mock',    -- mock | cricapi | scraped
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS player_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id       INTEGER REFERENCES players(id),
    match_id        INTEGER REFERENCES matches(id),
    team            TEXT,
    runs            INTEGER DEFAULT 0,
    balls_faced     INTEGER DEFAULT 0,
    fours           INTEGER DEFAULT 0,
    sixes           INTEGER DEFAULT 0,
    strike_rate     REAL DEFAULT 0.0,
    wickets         INTEGER DEFAULT 0,
    overs_bowled    REAL DEFAULT 0.0,
    runs_conceded   INTEGER DEFAULT 0,
    economy_rate    REAL DEFAULT 0.0,
    catches         INTEGER DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id            INTEGER REFERENCES matches(id),
    model_version       TEXT,
    team_a_win_prob     REAL,
    team_b_win_prob     REAL,
    predicted_winner    TEXT,
    confidence          REAL,
    key_features_json   TEXT,   -- JSON dict of feature importances
    explanation         TEXT,   -- LLM or fallback explanation
    created_at          TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS telegram_users (
    -- NOTE: Only stores user_id (no PII like name, username stored persistently)
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id     INTEGER NOT NULL UNIQUE,
    notify_enabled  INTEGER DEFAULT 1,
    language        TEXT DEFAULT 'en',
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_matches_date     ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_matches_teams    ON matches(team_a, team_b);
CREATE INDEX IF NOT EXISTS idx_player_stats_pid ON player_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_predictions_mid  ON predictions(match_id);
