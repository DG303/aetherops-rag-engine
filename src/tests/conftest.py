# tests/conftest.py
import pytest

from rag_core.models.document import Document


@pytest.fixture
def simple_document():
    """A minimal document with predictable line count for testing chunking."""
    content = "\n".join(f"line {i} AetherOps RAG Engine testing chunking" for i in range(1, 101))
    return Document(
        repo_name="test-repo",
        branch="main",
        path="src/tests/test_data/test.txt",
        language="text",
        content=content,
    )
