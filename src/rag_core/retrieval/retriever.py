# rag_core/retrieval/retriever.py
import logging

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag_core.config.config import config
from rag_core.config.logging_config import setup_logging
from rag_core.models.retrieval_result import RetrievalResult

setup_logging()
logger = logging.getLogger(__name__)

_client = None
_model = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
        )
    return _client


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def retrieve(query: str, limit: int = 5) -> list[RetrievalResult]:
    logger.info("Retrieving %s results for query: %s", limit, query)

    query_vector = embed_query(query)

    results = (
        get_client()
        .query_points(
            collection_name=config.QDRANT_COLLECTION_NAME,
            query=query_vector,
            limit=limit,
        )
        .points
    )

    retrieval_results = []

    for result in results:
        payload = result.payload or {}
        retrieval_results.append(
            RetrievalResult(
                repo_name=payload.get("repo_name", ""),
                branch=payload.get("branch", ""),
                source_path=payload.get("source_path", ""),
                language=payload.get("language", ""),
                content=payload.get("content", ""),
                score=float(result.score),
            )
        )

    logger.info("Retrieved %s results for query: %s", len(retrieval_results), query)
    return retrieval_results


def embed_query(query: str) -> list[float]:
    logger.debug("Embedding query using model: %s", config.EMBED_MODEL)

    embeddings = get_model().encode(
        query,
        normalize_embeddings=True,
    )

    return embeddings.tolist()
