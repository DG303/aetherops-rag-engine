# tests/unit/test_retriever.py
from unittest.mock import MagicMock

import numpy as np

import rag_core.retrieval.retriever as retriever_module
from rag_core.models.retrieval_result import RetrievalResult
from rag_core.retrieval.retriever import embed_query, get_client, get_model, retrieve


def _make_mock_point(repo_name="my-repo", branch="main", source_path="src/main.py",
                     language="python", content="print('hello')", score=0.95):
    point = MagicMock()
    point.payload = {
        "repo_name": repo_name,
        "branch": branch,
        "source_path": source_path,
        "language": language,
        "content": content,
    }
    point.score = score
    return point


# --- embed_query ---

def test_embed_query_calls_model_encode(monkeypatch):
    """embed_query should delegate to model.encode with normalize_embeddings=True."""
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    monkeypatch.setattr(retriever_module, "_model", fake_model)

    result = embed_query("test query")

    fake_model.encode.assert_called_once_with("test query", normalize_embeddings=True)
    assert result == [0.1, 0.2, 0.3]


def test_embed_query_returns_list(monkeypatch):
    """embed_query should always return a plain Python list, not an ndarray."""
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([0.5, 0.6])
    monkeypatch.setattr(retriever_module, "_model", fake_model)

    result = embed_query("hello")

    assert isinstance(result, list)


# --- retrieve ---

def test_retrieve_returns_retrieval_results(monkeypatch):
    """retrieve should map Qdrant hits to RetrievalResult objects."""
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    monkeypatch.setattr(retriever_module, "_model", fake_model)

    fake_client = MagicMock()
    fake_client.query_points.return_value.points = [_make_mock_point()]
    monkeypatch.setattr(retriever_module, "_client", fake_client)

    results = retrieve("test query", limit=1)

    assert len(results) == 1
    assert isinstance(results[0], RetrievalResult)
    assert results[0].repo_name == "my-repo"
    assert results[0].branch == "main"
    assert results[0].source_path == "src/main.py"
    assert results[0].language == "python"
    assert results[0].content == "print('hello')"
    assert results[0].score == 0.95


def test_retrieve_empty_results(monkeypatch):
    """retrieve should return an empty list when Qdrant returns no hits."""
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    monkeypatch.setattr(retriever_module, "_model", fake_model)

    fake_client = MagicMock()
    fake_client.query_points.return_value.points = []
    monkeypatch.setattr(retriever_module, "_client", fake_client)

    results = retrieve("nothing here", limit=5)

    assert results == []


def test_retrieve_multiple_results(monkeypatch):
    """retrieve should return all hits returned by Qdrant."""
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    monkeypatch.setattr(retriever_module, "_model", fake_model)

    points = [
        _make_mock_point(content="chunk A", score=0.9),
        _make_mock_point(content="chunk B", score=0.8),
        _make_mock_point(content="chunk C", score=0.7),
    ]
    fake_client = MagicMock()
    fake_client.query_points.return_value.points = points
    monkeypatch.setattr(retriever_module, "_client", fake_client)

    results = retrieve("query", limit=3)

    assert len(results) == 3
    assert [r.content for r in results] == ["chunk A", "chunk B", "chunk C"]
    assert [r.score for r in results] == [0.9, 0.8, 0.7]


def test_retrieve_passes_limit_to_qdrant(monkeypatch):
    """retrieve should forward the limit argument to Qdrant."""
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    monkeypatch.setattr(retriever_module, "_model", fake_model)

    fake_client = MagicMock()
    fake_client.query_points.return_value.points = []
    monkeypatch.setattr(retriever_module, "_client", fake_client)

    retrieve("query", limit=10)

    call_kwargs = fake_client.query_points.call_args.kwargs
    assert call_kwargs["limit"] == 10


# --- lazy initializers ---

def test_get_model_returns_same_instance(monkeypatch):
    """get_model should return the same object on repeated calls (singleton)."""
    monkeypatch.setattr(retriever_module, "_model", None)
    fake_model = MagicMock()

    monkeypatch.setattr(retriever_module, "_model", fake_model)
    m1 = get_model()
    m2 = get_model()

    assert m1 is m2


def test_get_client_returns_same_instance(monkeypatch):
    """get_client should return the same object on repeated calls (singleton)."""
    fake_client = MagicMock()
    monkeypatch.setattr(retriever_module, "_client", fake_client)

    c1 = get_client()
    c2 = get_client()

    assert c1 is c2
