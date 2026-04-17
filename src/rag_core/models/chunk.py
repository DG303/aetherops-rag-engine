# src/rag_core/models/chunk.py
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    repo_name: str
    branch: str
    source_path: str
    language: str
    content: str
    id: str = field(default="")
