# src/rag_core/embedding/embedder.py
import logging

from sentence_transformers import SentenceTransformer

from rag_core.config.config import config
from rag_core.config.logging_config import setup_logging
from rag_core.models.chunk import Chunk
from rag_core.models.embedded_chunk import EmbeddedChunk

setup_logging()
logger = logging.getLogger(__name__)


# model = SentenceTransformer(config.EMBED_MODEL)
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def embed_chunks(chunks: list[Chunk], model: SentenceTransformer = None) -> list[EmbeddedChunk]:
    logger.info("starting embedding process for %s chunks", len(chunks))

    if model is None:
        model = get_model()

    texts = [chunk.content for chunk in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append(
            EmbeddedChunk(
                repo_name=chunk.repo_name,
                branch=chunk.branch,
                source_path=chunk.source_path,
                language=chunk.language,
                content=chunk.content,
                embedding=embedding,
            )
        )

    logger.info("embedded %s chunks", len(embedded_chunks))
    return embedded_chunks
