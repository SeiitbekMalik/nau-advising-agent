"""Settings, read from environment variables (or a .env file).

Everything has a safe default so the app runs with no configuration at all:
no API key -> the offline "extractive" answer mode is used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines) so no extra dependency is needed."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


@dataclass
class Settings:
    # Where the knowledge-base markdown files live (seed files + ingested web pages).
    kb_dir: Path = field(default_factory=lambda: Path(os.getenv("KB_DIR", ROOT / "data" / "kb")))
    # SQLite file for the conversation log and escalation queue.
    db_path: Path = field(default_factory=lambda: Path(os.getenv("DB_PATH", ROOT / "data" / "advising.db")))

    # LLM provider: "extractive" (offline, no key), "anthropic" or "openai".
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "").lower())
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # Retrieval settings.
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "4")))
    # Minimum share of the question's key terms that must appear in the best source.
    # Below this the agent says it doesn't know and offers a human advisor.
    min_coverage: float = field(default_factory=lambda: float(os.getenv("MIN_COVERAGE", "0.25")))

    def resolved_provider(self) -> str:
        """Pick the provider: explicit setting wins, else whichever key is present, else offline."""
        if self.llm_provider in {"extractive", "anthropic", "openai"}:
            return self.llm_provider
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        return "extractive"


settings = Settings()
