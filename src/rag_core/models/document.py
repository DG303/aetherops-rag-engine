# src/rag_core/models/document.py
from dataclasses import dataclass


@dataclass
class Document:
    repo_name: str
    branch: str
    path: str
    language: str
    content: str
