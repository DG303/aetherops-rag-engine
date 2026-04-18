# src/rag_core/config/config.py
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_int(name: str, default: int) -> int:
    try:
        value = os.getenv(name)
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def parse_allowed_extensions(raw: str) -> tuple[list[str], set[str], set[str]]:
    """Split env list into extension tokens, suffixes (with leading dot), and exact filenames."""
    tokens = [item.strip() for item in raw.split(",") if item.strip()]
    suffixes: set[str] = set()
    filenames: set[str] = set()
    for item in tokens:
        lower = item.lower()
        if lower.startswith("."):
            suffixes.add(lower)
        elif "." in lower:
            filenames.add(lower)
        else:
            suffixes.add(f".{lower}")
    return tokens, suffixes, filenames


class Config:
    APP_NAME: str = os.getenv("APP_NAME", "AetherOps RAG")
    APP_ENV: str = os.getenv("APP_ENV", "local")
    APP_VERSION: str = os.getenv("APP_VERSION", "0.1.0")
    APP_DESCRIPTION: str = os.getenv("APP_DESCRIPTION", "AetherOps RAG")
    APP_AUTHOR: str = os.getenv("APP_AUTHOR", "AetherOps")
    APP_AUTHOR_EMAIL: str = os.getenv("APP_AUTHOR_EMAIL", "support@aetherops.ai")
    APP_AUTHOR_URL: str = os.getenv("APP_AUTHOR_URL", "https://aetherops.ai")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # QDRANT CONFIG
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = get_int("QDRANT_PORT", 6333)
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "aetherops_rag")
    QDRANT_BATCH_SIZE: int = get_int("QDRANT_BATCH_SIZE", 100)

    # INDEXER
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    GITHUB_REPO: str = os.getenv("GITHUB_REPO", "").strip()
    GITHUB_BRANCH: str = os.getenv("GITHUB_BRANCH", "main").strip() or "main"
    CHUNK_LINES: int = get_int("CHUNK_LINES", 30)
    CHUNK_OVERLAP: int = get_int("CHUNK_OVERLAP", 5)

    ALLOWED_EXTENSIONS: list[str]
    ALLOWED_SUFFIXES: set[str]
    ALLOWED_FILENAMES: set[str]

    ALLOWED_EXTENSIONS, ALLOWED_SUFFIXES, ALLOWED_FILENAMES = parse_allowed_extensions(
        os.getenv(
            "ALLOWED_EXTENSIONS",
            "py,js,ts,jsx,tsx,html,css,json,yml,yaml,md,txt,csv,ini,cfg,conf,log,sql,db,sqlite,sqlite3,sqlite3.db,sqlite3.db.gz,sqlite3.db.tar,sqlite3.db.tar.gz,sqlite3.db.tar.bz2,sqlite3.db.tar.xz,sqlite3.db.tar.lzma,sqlite3.db.tar.lzma.gz,sqlite3.db.tar.lzma.bz2,sqlite3.db.tar.lzma.xz",
        )
    )

    EXCLUDED_DIRS: list[str] = os.getenv(
        "EXCLUDED_DIRS", ".git,.DS_Store,.tmp,.venv,.env,.gitignore,.gitmodules,.gitignore"
    ).split(",")
    EXCLUDED_DIR_NAMES: set[str] = {d.lower().strip("/") for d in EXCLUDED_DIRS}

    # Keys: file extension from os.path.splitext (".py") or lowercased basename for extensionless names.
    LANGUAGE_MAP: dict[str, str] = {
        ".py": "python",
        ".pyw": "python",
        ".pyi": "python",
        ".js": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".txt": "text",
        ".csv": "csv",
        ".ini": "ini",
        ".cfg": "config",
        ".conf": "config",
        ".log": "log",
        ".sql": "sql",
        ".tf": "terraform",
        ".tfvars": "terraform",
        ".tfstate": "terraform",
        ".sh": "shell",
        ".bash": "shell",
        ".zsh": "shell",
        ".db": "database",
        ".sqlite": "database",
        ".sqlite3": "database",
        "dockerfile": "docker",
        "docker-compose.yml": "docker",
        "docker-compose.yaml": "docker",
        "makefile": "make",
    }

    READ_ENCODINGS: list[str] = [
        item.strip()
        for item in os.getenv("READ_ENCODINGS", "utf-8,utf-8-sig,utf-16,cp1252").split(",")
        if item.strip()
    ]

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"


config = Config()
