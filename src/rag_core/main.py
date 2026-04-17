# src/rag_core/main.py
import logging

from rag_core.chunking.chunk import chunk_documents
from rag_core.config.logging_config import setup_logging
from rag_core.embedding.embedder import embed_chunks
from rag_core.indexing.github_loader import load_repo_files
from rag_core.storage.qdrant_store import store_embeddings

setup_logging()
logger = logging.getLogger(__name__)

def run_indexer(repo_url: str, branch: str):
    logger.info("Running indexer for repo '%s' on branch '%s'", repo_url, branch)

    docs = load_repo_files(repo_url, branch)
    logger.info("Loaded %s documents from repo '%s' on branch '%s'", len(docs), repo_url, branch)

    chunks = chunk_documents(docs)
    logger.info("Chunked %s documents into %s chunks", len(docs), len(chunks))

    embedded_chunks = embed_chunks(chunks)
    logger.info("Embedded %s chunks", len(embedded_chunks))

    store_embeddings(embedded_chunks)
    logger.info("Stored %s embedded chunks", len(embedded_chunks))

    logger.info("Indexer completed successfully")
