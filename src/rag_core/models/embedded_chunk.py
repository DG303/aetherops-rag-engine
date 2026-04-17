# src/rag_core/models/embedded_chunk.py
from dataclasses import dataclass


@dataclass
class EmbeddedChunk:
    repo_name: str
    branch: str
    source_path: str
    language: str
    content: str
    embedding: list[float]
