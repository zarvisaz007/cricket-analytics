#!/usr/bin/env bash
# setup_db.sh — Create data directories and initialise the SQLite database.

set -euo pipefail

echo "==> Creating data directories …"
mkdir -p data/models data/context_archives data/llm_cache
mkdir -p logs run/queue

echo "==> Creating empty data files …"
touch data/context_summaries.jsonl logs/agents.log
echo '{}' > run/metrics.json

echo "==> Initialising SQLite database …"
DB_PATH="${DATABASE_URL:-sqlite:///./data/cricket.db}"
# Strip sqlite:/// prefix
DB_FILE="${DB_PATH#sqlite:///}"
DB_FILE="${DB_FILE#./}"

sqlite3 "$DB_FILE" < src/data/schema.sql
echo "    Database: $DB_FILE"

echo "==> Running Phase 2 migrations …"
PYTHON=$(command -v .venv/bin/python3 || command -v python3)
$PYTHON -m src.data.migrations.add_phase2_tables
echo "    Phase 2 tables created/verified."

echo "==> Done. Run 'bash scripts/start_all_agents.sh' to launch."
