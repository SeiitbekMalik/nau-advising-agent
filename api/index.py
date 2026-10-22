"""Vercel entry point: exposes the FastAPI app as a serverless function.

Every request is routed here by vercel.json. Locally, keep using:
    uvicorn app.main:app --reload
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app  # noqa: E402,F401
