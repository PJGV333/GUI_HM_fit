#!/usr/bin/env bash

# Activate the virtual environment
source "$(dirname "$0")/venv/bin/activate"

# Start the FastAPI backend on the expected port
uvicorn backend_fastapi.main:app --host 127.0.0.1 --port 8001
