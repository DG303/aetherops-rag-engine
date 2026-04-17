# src/rag_core/storage/qdrant_store.py
import logging
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag_core.config.config import config
from rag_core.config.logging_config import setup_logging
from rag_core.models.embedded_chunk import EmbeddedChunk

setup_logging()
logger = logging.getLogger(__name__)

_client = None

def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
        )
    return _client

def _batched(items: list[PointStruct], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]  # pyright: ignore[reportUndefinedVariable]

def store_embeddings(embedded_chunks: list[EmbeddedChunk], client: QdrantClient = None) -> None:
    if client is None:
        client = get_client()

    logger.info("Storing %s embedded chunks in Qdrant collection '%s'", len(embedded_chunks), config.QDRANT_COLLECTION_NAME)

    if not embedded_chunks:
        logger.warning("No embedded chunks to store")
        return

    ensure_collection_exists(embedded_chunks, client)

    points = []

    for chunk in embedded_chunks:
        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=chunk.embedding,
                payload={
                    "repo_name": chunk.repo_name,
                    "branch": chunk.branch,
                    "source_path": chunk.source_path,
                    "language": chunk.language,
                    "content": chunk.content,
                }
            )
        )

    batch_size = max(1, config.QDRANT_BATCH_SIZE)
    total_batches = (len(points) + batch_size -1) // batch_size

    for batch_index, batch in enumerate(_batched(points, batch_size), start=1):
        client.upsert(
            collection_name=config.QDRANT_COLLECTION_NAME,
            points=batch,
        )
        logger.debug(
            "Upsetted batch %s/%s (%s points)",
            batch_index,
            total_batches,
            len(batch),
        )

    logger.info("Stored %s embedded chunks in Qdrant collection '%s'", len(embedded_chunks), config.QDRANT_COLLECTION_NAME)

def ensure_collection_exists(embedded_chunks: list[EmbeddedChunk], client: QdrantClient = None) -> None:
    if client is None:
        client = get_client()

    existing = client.get_collections().collections
    existing_names = {c.name for c in existing}

    if config.QDRANT_COLLECTION_NAME in existing_names:
        logger.debug("Collection '%s' already exists", config.QDRANT_COLLECTION_NAME)
        return

    vector_size = len(embedded_chunks[0].embedding)

    logger.info("Creating collection '%s' with vector size %s", config.QDRANT_COLLECTION_NAME, vector_size)

    client.create_collection(
        collection_name=config.QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )
    logger.info("Created collection '%s' with vector size %s", config.QDRANT_COLLECTION_NAME, vector_size)

