# tests/unit/test_embedder.py
from unittest.mock import MagicMock

import numpy as np

from rag_core.embedding.embedder import embed_chunks
from rag_core.models.chunk import Chunk


def test_embed_chunks_returns_embedded_chunks():
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    chunks = [
        Chunk(repo_name="test-repo", branch="main", source_path="src/tests/test_data/test.txt", language="text", content="line 1 AetherOps RAG Engine testing chunking"),
        Chunk(repo_name="test-repo", branch="main", source_path="src/tests/test_data/test.txt", language="text", content="line 2 AetherOps RAG Engine testing chunking"),
    ]

    embedded_chunks = embed_chunks(chunks, fake_model)

    assert len(embedded_chunks) == 2

    np.testing.assert_array_almost_equal(embedded_chunks[0].embedding, [0.1, 0.2, 0.3])
    np.testing.assert_array_almost_equal(embedded_chunks[1].embedding, [0.4, 0.5, 0.6])

    assert embedded_chunks[0].repo_name == "test-repo"
    assert embedded_chunks[0].branch == "main"
    assert embedded_chunks[0].source_path == "src/tests/test_data/test.txt"
    assert embedded_chunks[0].language == "text"
    assert embedded_chunks[0].content == "line 1 AetherOps RAG Engine testing chunking"

    assert embedded_chunks[1].repo_name == "test-repo"
    assert embedded_chunks[1].branch == "main"
    assert embedded_chunks[1].source_path == "src/tests/test_data/test.txt"
    assert embedded_chunks[1].language == "text"
    assert embedded_chunks[1].content == "line 2 AetherOps RAG Engine testing chunking"

    fake_model.encode.assert_called_once_with(
        ["line 1 AetherOps RAG Engine testing chunking", "line 2 AetherOps RAG Engine testing chunking"],
        normalize_embeddings=True,
    )
