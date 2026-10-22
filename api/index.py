"""Vercel entry point: exposes the FastAPI app as a serverless function.

vercel.json rewrites every request to /api/index?__path=/<original path>.
Depending on the runtime, the app may see either the original path or
"/api/index", so this small wrapper restores the original path before
FastAPI routes the request.

Locally, keep using:  uvicorn app.main:app --reload
"""

import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app as fastapi_app  # noqa: E402

PREFIX = "/api/index"


async def app(scope, receive, send):
    if scope["type"] in ("http", "websocket"):
        query = parse_qsl(scope.get("query_string", b"").decode(), keep_blank_values=True)
        original = next((v for k, v in query if k == "__path"), None)
        path = scope["path"]
        if original is not None:
            path = "/" + original.lstrip("/")
        elif path.startswith(PREFIX):
            path = path[len(PREFIX):] or "/"
        rest = [(k, v) for k, v in query if k != "__path"]
        scope = dict(scope, path=path, raw_path=path.encode(), query_string=urlencode(rest).encode())
    await fastapi_app(scope, receive, send)
