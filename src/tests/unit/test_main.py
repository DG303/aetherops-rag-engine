# tests/unit/test_main.py
from unittest.mock import MagicMock

import rag_core.main as main_module
from rag_core.main import run_indexer

# ---------------------------------------------------------------------------
# Shared fake objects
# ---------------------------------------------------------------------------

FAKE_DOCS = [MagicMock(name="doc1"), MagicMock(name="doc2")]
FAKE_CHUNKS = [MagicMock(name="chunk1"), MagicMock(name="chunk2"), MagicMock(name="chunk3")]
FAKE_EMBEDDED = [MagicMock(name="emb1"), MagicMock(name="emb2"), MagicMock(name="emb3")]


def _patch_pipeline(monkeypatch, docs=None, chunks=None, embedded=None):
    """Replace all four pipeline steps with controllable mocks."""
    docs = docs if docs is not None else FAKE_DOCS
    chunks = chunks if chunks is not None else FAKE_CHUNKS
    embedded = embedded if embedded is not None else FAKE_EMBEDDED

    mock_load = MagicMock(return_value=docs)
    mock_chunk = MagicMock(return_value=chunks)
    mock_embed = MagicMock(return_value=embedded)
    mock_store = MagicMock(return_value=None)

    # Patch the names as imported in rag_core.main, not their source modules
    monkeypatch.setattr(main_module, "load_repo_files", mock_load)
    monkeypatch.setattr(main_module, "chunk_documents", mock_chunk)
    monkeypatch.setattr(main_module, "embed_chunks", mock_embed)
    monkeypatch.setattr(main_module, "store_embeddings", mock_store)

    return mock_load, mock_chunk, mock_embed, mock_store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_indexer_calls_load_repo_files(monkeypatch):
    """load_repo_files should be called with the repo URL and branch."""
    mock_load, _, _, _ = _patch_pipeline(monkeypatch)

    run_indexer("https://github.com/org/repo.git", "main")

    mock_load.assert_called_once_with("https://github.com/org/repo.git", "main")


def test_run_indexer_passes_docs_to_chunk_documents(monkeypatch):
    """The documents returned by load_repo_files must be passed to chunk_documents."""
    mock_load, mock_chunk, _, _ = _patch_pipeline(monkeypatch)

    run_indexer("https://github.com/org/repo.git", "main")

    mock_chunk.assert_called_once_with(mock_load.return_value)


def test_run_indexer_passes_chunks_to_embed_chunks(monkeypatch):
    """The chunks returned by chunk_documents must be passed to embed_chunks."""
    _, mock_chunk, mock_embed, _ = _patch_pipeline(monkeypatch)

    run_indexer("https://github.com/org/repo.git", "main")

    mock_embed.assert_called_once_with(mock_chunk.return_value)


def test_run_indexer_passes_embeddings_to_store(monkeypatch):
    """The embedded chunks returned by embed_chunks must be passed to store_embeddings."""
    _, _, mock_embed, mock_store = _patch_pipeline(monkeypatch)

    run_indexer("https://github.com/org/repo.git", "main")

    mock_store.assert_called_once_with(mock_embed.return_value)


def test_run_indexer_executes_steps_in_order(monkeypatch):
    """All four pipeline steps should run in the correct sequence."""
    call_order = []

    def fake_load(*_):
        call_order.append("load")
        return FAKE_DOCS

    def fake_chunk(*_):
        call_order.append("chunk")
        return FAKE_CHUNKS

    def fake_embed(*_):
        call_order.append("embed")
        return FAKE_EMBEDDED

    def fake_store(*_):
        call_order.append("store")

    monkeypatch.setattr(main_module, "load_repo_files", fake_load)
    monkeypatch.setattr(main_module, "chunk_documents", fake_chunk)
    monkeypatch.setattr(main_module, "embed_chunks", fake_embed)
    monkeypatch.setattr(main_module, "store_embeddings", fake_store)

    run_indexer("https://github.com/org/repo.git", "main")

    assert call_order == ["load", "chunk", "embed", "store"]


def test_run_indexer_empty_repo_runs_without_error(monkeypatch):
    """If the repo has no files, run_indexer should complete without raising."""
    _patch_pipeline(monkeypatch, docs=[], chunks=[], embedded=[])

    run_indexer("https://github.com/org/empty-repo.git", "main")  # should not raise
