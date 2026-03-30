import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def stable_test_env(monkeypatch):
    """Provide stable defaults so tests are less dependent on local shell env."""
    monkeypatch.setenv("ENV", os.getenv("ENV", "dev"))
    monkeypatch.setenv("QDRANT_COLLECTION", os.getenv("QDRANT_COLLECTION", "divination_master_collection"))
    monkeypatch.setenv("EMBEDDINGS_API", "openai")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDINGS_DIMENSION", "384")
