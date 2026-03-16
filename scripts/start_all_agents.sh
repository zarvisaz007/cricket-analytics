#!/usr/bin/env bash
# start_all_agents.sh — Launch the full pipeline via the orchestrator.

set -euo pipefail
cd "$(dirname "$0")/.."

source .env 2>/dev/null || true

echo "==> Starting Cricket Analytics orchestrator …"
python -m src.agents.orchestrator "$@"
