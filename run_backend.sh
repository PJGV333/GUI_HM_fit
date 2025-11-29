#!/usr/bin/env bash

# Activate the virtual environment
source "$(dirname "$0")/venv/bin/activate"

# Start the FastAPI backend on the expected port
HOST="${HM_FIT_BACKEND_HOST:-127.0.0.1}"
PORT="${HM_FIT_BACKEND_PORT:-8001}"
uvicorn backend_fastapi.main:app --host "$HOST" --port "$PORT"
