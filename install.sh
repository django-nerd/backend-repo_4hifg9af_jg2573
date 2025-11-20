#!/usr/bin/env bash
set -euo pipefail

# Easy installer for the backend (FastAPI)
# - Creates Python venv
# - Installs pip dependencies
# - Writes a default .env if missing
#
# Usage: bash install.sh

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[!] Python not found. Please install Python 3.10+ and retry."
  exit 1
fi

# Create venv if missing
if [ ! -d .venv ]; then
  echo "[+] Creating virtual environment (.venv)"
  "$PYTHON_BIN" -m venv .venv
fi

# Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate

# Upgrade pip + install deps
echo "[+] Upgrading pip and installing backend dependencies"
pip install --upgrade pip
pip install -r requirements.txt

# Create .env with sensible defaults if missing
if [ ! -f .env ]; then
  cat > .env <<'EOF'
# FastAPI Backend .env
# Change OLLAMA_BASE_URL to a reachable Ollama server if different
OLLAMA_BASE_URL=http://localhost:11434

# Optional: configure MongoDB for persistence
# DATABASE_URL=mongodb+srv://<user>:<pass>@<cluster>/
# DATABASE_NAME=voice_assistant

# Optional tuning
# TOKEN_TTL_DEFAULT=300
# READ_MAX_BYTES=1048576
# RUN_TIMEOUT_MS=8000
# OUTPUT_MAX_BYTES=1048576
# SANDBOX_DIRS=/tmp,/var/tmp
# RUN_WHITELIST=echo,dir,ls
EOF
  echo "[+] Wrote backend .env with defaults"
else
  echo "[i] Existing backend .env detected — leaving it unchanged"
fi

cat <<'NEXT'

✅ Backend install complete.

Next steps:
1) Activate venv:
   source .venv/bin/activate
2) Start the server:
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
3) Verify health:
   open http://localhost:8000/health
   (ollama may show unreachable until it is running)

Tip: To use chat, start Ollama and pull a model:
   ollama pull llama3.1
   curl http://localhost:11434/api/tags
NEXT
