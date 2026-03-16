# 🏏 Cricket Analytics Bot

A multi-agent cricket prediction and analytics system with a Telegram bot interface,
XGBoost match predictions, LLM-powered reports, and automatic context management with
purge/archive.

---

## Quickstart

```bash
# 1. Clone / enter the project directory
cd Claude-cricket

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and set your keys (see Configuration section below)

# 4. Initialise the database and folder structure
bash scripts/setup_db.sh

# 5. Seed the database with real historical data (2020–2026)
python scripts/populate_real_data.py

# 6. Run the end-to-end demo (no Telegram needed)
python dev_demo.py

# 7. Launch the full pipeline (ingestion → training → bot)
bash scripts/start_all_agents.sh

# --- OR --- launch only the bot (for development)
bash scripts/start_bot.sh
```

---

## Configuration

All configuration is in `.env` (copy from `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | — | From @BotFather |
| `OPENROUTER_API_KEY` | (empty) | Optional — enables AI reports & explanations |
| `CRICKET_API_KEY` | (empty) | Optional — enables live CricAPI data |
| `CRICKET_API_PROVIDER` | `mock` | `mock` \| `cricapi` \| `espn_scrape` |
| `DATABASE_URL` | `sqlite:///./data/cricket.db` | SQLite path or Postgres URL |
| `MODEL_PATH` | `./data/models/xgb_v1.joblib` | XGBoost model output |
| `MODEL_MAX_TOKENS` | `200000` | LLM context window size |
| `CONTEXT_PURGE_PERCENT` | `50` | Purge at this % of MODEL_MAX_TOKENS |
| `CONTEXT_HARD_LIMIT` | `100000` | Purge if tokens exceed this absolute value |
| `FORCE_KEEP_EPHEMERAL` | `false` | Keep messages in memory after purge (debug) |
| `LLM_MODEL` | `gpt-4o-mini` | Model for tiktoken estimation + OpenRouter |
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` |

### Changing the token window

```bash
# Example: use a 128k model instead of 200k
MODEL_MAX_TOKENS=128000   # in .env
```

The purge threshold auto-adjusts: at 50% that becomes 64,000 tokens.

---

## Context Management & Purge Rules

The system uses an in-memory context list shared by all agents.

### Purge trigger (OR rule)

A purge fires when **either** condition is true:

```
tokens > MODEL_MAX_TOKENS × CONTEXT_PURGE_PERCENT / 100
   OR
tokens > CONTEXT_HARD_LIMIT
```

With defaults: purge fires when `tokens > 100,000` (50% of 200k) **or** `tokens > 100,000` (hard limit).

To make purges less frequent, raise both:

```bash
CONTEXT_PURGE_PERCENT=70
CONTEXT_HARD_LIMIT=150000
```

### Manual purge

```python
from src.agents.context_manager import context_manager
summary = context_manager.purge_and_archive("manual", "user_triggered")
print(summary["archive_path"])
```

Or via the demo script:
```bash
python dev_demo.py   # adds messages until threshold, shows purge output
```

### Where summaries are stored

| Artifact | Path |
|---|---|
| Full message archives | `./data/context_archives/<timestamp>_<phase>.jsonl` |
| Summary index (one JSON per line) | `./data/context_summaries.jsonl` |
| Phase completion pointers | `./run/phase_status.json` |
| Structured metrics / events | `./run/metrics.json` |
| Agent logs | `./logs/agents.log` |

---

## Pipeline Phases

```
phase_ingestion_seed   → fetch/seed match data into DB
phase_model_train      → train XGBoost model
phase_bot_ready        → launch Telegram bot
```

Run a single phase for development:
```bash
python -m src.agents.orchestrator --phase-only phase_model_train
```

---

## Telegram Bot

Start a conversation with your bot and type `/start`.

```
Main Menu
├── 📅 Upcoming Matches  → list matches → tap to predict
├── 🔮 Predict           → same flow
├── 👤 Player Report     → AI-generated narrative (LLM or fallback)
└── ⚙️ Settings          → notification preferences
```

On any prediction result, tap **❓ Why?** to get a short LLM explanation
of which features drove the outcome.

---

## Privacy

- Only `telegram_id` (an integer) and notification preferences are stored.
- No usernames, first names, phone numbers, or other PII are persisted.
- See `src/data/db.py` → `TelegramUser` for the exact schema.

---

## Keeping the Database Updated

A scheduled updater runs daily to fetch new match results:

```bash
# Manual run
python scripts/update_matches.py

# Add to crontab (runs at 06:00 daily)
0 6 * * * cd /path/to/Claude-cricket && python scripts/update_matches.py
```

---

## Project Structure

```
Claude-cricket/
├── .env.example
├── requirements.txt
├── dev_demo.py                  ← end-to-end synthetic demo
├── src/
│   ├── agents/
│   │   ├── context_manager.py  ← shared in-memory context + purge
│   │   ├── orchestrator.py     ← phase runner + IPC queue
│   │   ├── ingestion_agent.py  ← match data fetcher
│   │   ├── ml_agent.py         ← model trainer
│   │   ├── nlp_agent.py        ← LLM wrapper
│   │   └── bot_agent.py        ← Telegram bot launcher
│   ├── bot/
│   │   ├── main.py             ← Application builder
│   │   └── handlers.py         ← command + callback handlers
│   ├── data/
│   │   ├── db.py               ← SQLAlchemy models + session
│   │   └── schema.sql          ← raw SQL schema
│   ├── ml/
│   │   └── train.py            ← XGBoost training + inference
│   ├── nlp/
│   │   └── llm_client.py       ← OpenRouter wrapper + cache
│   └── utils/
│       └── token_utils.py      ← tiktoken helpers
├── scripts/
│   ├── setup_db.sh
│   ├── start_all_agents.sh
│   ├── start_bot.sh
│   ├── populate_real_data.py   ← seeds 2020–2026 historical data
│   └── update_matches.py       ← daily updater
├── data/
│   ├── cricket.db
│   ├── models/
│   ├── context_archives/
│   ├── llm_cache/
│   └── context_summaries.jsonl
├── logs/agents.log
└── run/
    ├── queue/
    ├── metrics.json
    └── phase_status.json
```

---

## LLM Fallbacks

All LLM calls (via OpenRouter) have deterministic fallback strings that are cached
to `./data/llm_cache/` so behaviour stays consistent whether or not an API key is set.
Set `OPENROUTER_API_KEY` to enable real AI-generated content.

---

MVP ready — how would you like the first demo flow to behave?
