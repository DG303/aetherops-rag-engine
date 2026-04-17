# rag_core/models/retrieval_result.py
from pydantic import BaseModel


class RetrievalResult(BaseModel):
    repo_name: str
    branch: str
    source_path: str
    language: str
    content: str
    score: float
