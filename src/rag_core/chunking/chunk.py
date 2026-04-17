# src/rag_core/chunking/chunk.py
import logging

from rag_core.config.config import config
from rag_core.config.logging_config import setup_logging
from rag_core.models.chunk import Chunk
from rag_core.models.document import Document

setup_logging()
logger = logging.getLogger(__name__)

def chunk_documents(documents: list[Document]) -> list[Chunk]:
    logger.info("Starting document chunking process")

    chunks: list[Chunk] = []

    for doc in documents:
        doc_chunks = chunk_document(doc)
        chunks.extend(doc_chunks)

    logger.info("Created %s chunks", len(chunks))
    return chunks

def chunk_document(document: Document, chunk_lines: int = config.CHUNK_LINES, chunk_overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    lines = document.content.splitlines()
    chunks = []
    start = 0
    total_lines = len(lines)

    while start < total_lines:
        end = start + chunk_lines
        # Skip a trailing partial window when overlap is active and we already
        # have at least one chunk (the partial content was already covered by
        # the previous chunk's overlap tail).
        if end > total_lines and chunks and chunk_overlap > 0:
            break
        window = lines[start:min(end, total_lines)]
        chunk_text = "\n".join(window)
        chunks.append(
            Chunk(
                id=generate_chunk_id(document, len(chunks)),
                repo_name=document.repo_name,
                branch=document.branch,
                source_path=document.path,
                language=document.language,
                content=chunk_text,
            )
        )
        if end >= total_lines:
            break
        start = end - chunk_overlap

    logger.debug(
        "Chunked document %s into %s chunks",
        document.path,
        len(chunks)
    )

    return chunks

def generate_chunk_id(document: Document, chunk_index: int) -> str:
    return f"{document.repo_name}|{document.branch}|{document.path}|{chunk_index}"
