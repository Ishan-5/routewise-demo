"""Shared config: reads ROUTEWISE_API_KEY / ROUTEWISE_BASE_URL from the environment or a local .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

DEFAULT_BASE_URL = "https://llm-router-d2b2.onrender.com"


def api_key() -> str:
    key = os.environ.get("ROUTEWISE_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "No ROUTEWISE_API_KEY set. Copy .env.example to .env and fill in your routewise API key."
        )
    return key


def base_url() -> str:
    return os.environ.get("ROUTEWISE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
