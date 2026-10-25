"""
backend/tests/conftest.py
=========================
Pytest fixtures for ARIA backend test suite.
"""

import os
import shutil
import pytest
from pathlib import Path


@pytest.fixture
def isolated_chroma(tmp_path, monkeypatch):
    """
    Isolates ChromaDB vector store into a temporary directory for unit testing.

    Sets CHROMA_PERSIST_PATH to a subfolder under pytest's tmp_path, sets
    EMBEDDING_MODEL_NAME to 'all-MiniLM-L6-v2', and cleans client caches so
    tests never touch or mutate the production chroma_db/ directory.

    If Windows file locks interfere with temp cleanup, pytest ignores cleanup errors.
    """
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("CHROMA_PERSIST_PATH", str(chroma_dir))
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    # Clear cached clients in vector_store module
    from backend.pipeline import vector_store
    vector_store._clients.clear()

    yield chroma_dir

    # Teardown: clear client references to avoid lingering Windows file locks
    vector_store._clients.clear()
