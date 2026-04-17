# tests/unit/test_qdrant_store.py
from unittest.mock import MagicMock

import numpy as np

from rag_core.config.config import config
from rag_core.models.embedded_chunk import EmbeddedChunk
from rag_core.storage.qdrant_store import ensure_collection_exists, store_embeddings


def make_embedded_chunk(content="hello world", embedding=None):
    return EmbeddedChunk(
        repo_name="test-repo",
        branch="main",
        source_path="src/foo.py",
        language="python",
        content=content,
        embedding=embedding if embedding is not None else np.array([0.1, 0.2, 0.3]),
    )


def make_mock_client(existing_collections=None):
    """Build a fake QdrantClient with controllable collection state."""
    client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.name = existing_collections[0] if existing_collections else ""
    client.get_collections.return_value.collections = (
        [mock_collection] if existing_collections else []
    )
    return client


def test_store_embeddings_calls_upsert():
    """store_embeddings should call upsert at least once with our points."""
    client = make_mock_client()
    chunks = [make_embedded_chunk()]
    store_embeddings(chunks, client)
    assert client.upsert.called


def test_store_embeddings_empty_list_does_not_upsert():
    """Empty input should return early without calling upsert."""
    client = make_mock_client()
    store_embeddings([], client)
    client.upsert.assert_not_called()


def test_store_embeddings_creates_collection_when_missing():
    """If the collection doesn't exist, create_collection should be called."""
    client = make_mock_client(existing_collections=[])
    chunks = [make_embedded_chunk()]
    store_embeddings(chunks, client)
    client.create_collection.assert_called_once()


def test_store_embeddings_skips_collection_creation_when_exists():
    """If the collection already exists, create_collection should NOT be called."""
    client = make_mock_client(existing_collections=[config.QDRANT_COLLECTION_NAME])
    chunks = [make_embedded_chunk()]
    store_embeddings(chunks, client)
    client.create_collection.assert_not_called()


def test_ensure_collection_exists_creates_with_correct_vector_size():
    """create_collection should be called with the correct vector dimensions."""
    client = make_mock_client(existing_collections=[])
    chunk = make_embedded_chunk(embedding=np.array([0.1, 0.2, 0.3]))  # size 3
    ensure_collection_exists([chunk], client)
    call_kwargs = client.create_collection.call_args.kwargs
    assert call_kwargs["vectors_config"].size == 3
