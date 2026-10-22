import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Tests always run offline and never touch the real database.
os.environ["LLM_PROVIDER"] = "extractive"
os.environ["DB_PATH"] = ":memory:"

from app.agent.core import AdvisingAgent  # noqa: E402
from app.agent.llm import ExtractiveGenerator  # noqa: E402
from app.kb.loader import load_kb  # noqa: E402
from app.kb.retriever import BM25Retriever  # noqa: E402
from app.storage import Store  # noqa: E402


@pytest.fixture(scope="session")
def chunks():
    return load_kb(ROOT / "data" / "kb")


@pytest.fixture
def store():
    return Store(":memory:")


@pytest.fixture
def agent(chunks, store):
    return AdvisingAgent(BM25Retriever(chunks), ExtractiveGenerator(), store)
