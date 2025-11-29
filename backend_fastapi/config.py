"""Configuration helpers for the FastAPI backend.

This module centralizes environment-driven settings so switching machines
only requires updating environment variables (no code edits).
"""

import os
from typing import List, Optional

# Environment: "dev" enables permissive localhost CORS for Vite; any other
# value (e.g., "prod") tightens to the packaged Tauri origins.
HM_FIT_ENV = os.getenv("HM_FIT_ENV", "dev").lower()

# Host/port for the FastAPI server. Defaults match the existing workspace
# conventions but can be overridden per machine.
BACKEND_HOST = os.getenv("HM_FIT_BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("HM_FIT_BACKEND_PORT", "8001"))

# CORS settings depend on environment. In dev we allow common localhost
# variants plus a regex to cover arbitrary localhost ports. In prod we
# explicitly allow the Tauri origin.
DEV_ORIGINS: List[str] = [
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:1420",  # Tauri dev server
    "http://127.0.0.1",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:1420",
    "tauri://localhost",  # Some Tauri builds keep this origin even in dev
]

PROD_ORIGINS: List[str] = [
    "tauri://localhost",
]

if HM_FIT_ENV == "dev":
    # Allow enumerated localhost hosts plus any localhost/127 port via regex so
    # the app "just works" across machines and dev setups without CORS edits.
    CORS_ALLOW_ORIGINS: List[str] = DEV_ORIGINS
    CORS_ALLOW_ORIGIN_REGEX: Optional[str] = r"(https?|tauri)://(localhost|127\.0\.0\.1)(:\d+)?"
else:
    # Production: restrict to the packaged Tauri origin.
    CORS_ALLOW_ORIGINS = PROD_ORIGINS
    CORS_ALLOW_ORIGIN_REGEX = r"tauri://localhost"

# Shared CORS flags
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]
CORS_ALLOW_CREDENTIALS = True
