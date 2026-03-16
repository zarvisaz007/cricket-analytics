#!/usr/bin/env bash
# start_bot.sh — Launch only the Telegram bot (no orchestrator overhead).

set -euo pipefail
cd "$(dirname "$0")/.."

source .env 2>/dev/null || true

echo "==> Initialising DB …"
python -c "from src.data.db import init_db; init_db()"

echo "==> Starting Telegram bot …"
python -m src.bot.main
