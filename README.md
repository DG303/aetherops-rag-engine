# AetherOps RAG Engine

A retrieval-augmented generation (RAG) pipeline that indexes source code from GitHub repositories into a Qdrant vector store and retrieves semantically relevant chunks in response to natural language queries.

---

## How it works

```
GitHub repo  →  load files  →  chunk  →  embed  →  Qdrant
                                                        ↑
query  →  embed query  →  vector search  ──────────────┘  →  results
```

1. **Indexer** — clones (or pulls) a GitHub repo, walks the file tree, chunks each file into overlapping line windows, embeds the chunks with a sentence-transformer model, and upserts them into Qdrant.
2. **Retriever** — embeds a natural language query with the same model, runs a nearest-neighbour search in Qdrant, and returns the top-k matching code chunks.

---

## Project structure

```
aetherops-rag-engine/
├── src/
│   ├── rag_core/
│   │   ├── chunking/       # chunk_document, chunk_documents
│   │   ├── config/         # Config (env-driven), logging setup
│   │   ├── embedding/      # embed_chunks (sentence-transformers)
│   │   ├── indexing/       # github_loader — clone/pull/walk/read
│   │   ├── models/         # Document, Chunk, EmbeddedChunk, RetrievalResult
│   │   ├── retrieval/      # retriever — embed_query, retrieve
│   │   ├── storage/        # qdrant_store — ensure_collection, store_embeddings
│   │   └── main.py         # run_indexer pipeline orchestrator
│   ├── indexer_app/
│   │   └── cli.py          # aetherops-rag-engine-indexer entry point
│   ├── retriever_app/
│   │   └── cli.py          # aetherops-rag-engine-retriever entry point
│   └── tests/              # pytest unit tests (95%+ coverage)
├── docker-compose.yml       # Qdrant service
├── pyproject.toml
├── pytest.ini
└── .pre-commit-config.yaml
```

---

## Setup

### 1. Start Qdrant

```bash
docker compose up -d
```

Qdrant will be available at `http://localhost:6333`.

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the package

```bash
# Runtime only
pip install -e .

# Runtime + dev tools (pytest, ruff, pre-commit)
pip install -e ".[dev]"
```

### 4. Configure environment variables

Copy the example below into a `.env` file at the project root and fill in your values:

```env
# Required
GITHUB_REPO=https://github.com/your-org/your-repo.git
GITHUB_BRANCH=main

# Qdrant (defaults match docker-compose)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=aetherops_rag

# Embedding model (any sentence-transformers model name)
EMBED_MODEL=all-MiniLM-L6-v2

# Chunking
CHUNK_LINES=30
CHUNK_OVERLAP=5
```

### 5. (Optional) Install pre-commit hooks

```bash
pre-commit install
```

After this, `ruff` lint and format checks run automatically on every `git commit`.

---

## Usage

### Index a repository

```bash
aetherops-rag-engine-indexer --repo-url https://github.com/org/repo.git --branch main
```

Or override the repo configured in `.env`:

```bash
aetherops-rag-engine-indexer \
  --repo-url https://github.com/your-org/your-repo.git \
  --branch main
```

### Query the index

```bash
aetherops-rag-engine-retriever "how does authentication work" --limit 5
```

Each result shows the file path, language, similarity score, and up to 1000 characters of the matching chunk.

---

## Development

### Run tests

```bash
# All tests with coverage report
.venv/bin/pytest src/tests/

# Single file
.venv/bin/pytest src/tests/unit/test_chunk.py

# With enforced coverage floor
.venv/bin/pytest src/tests/ --cov-fail-under=90
```

See [`src/tests/README.md`](src/tests/README.md) for full test documentation.

### Lint and format

```bash
# Check for issues
.venv/bin/ruff check src/

# Auto-fix
.venv/bin/ruff check src/ --fix

# Format (like black)
.venv/bin/ruff format src/
```

### Run pre-commit manually (without committing)

```bash
pre-commit run --all-files
```

---

## Configuration reference

All settings are read from environment variables (or `.env`). Every value has a sensible default so the engine works out of the box with the docker-compose Qdrant instance.

| Variable | Default | Description |
|---|---|---|
| `GITHUB_REPO` | *(empty)* | Repository URL to index |
| `GITHUB_BRANCH` | `main` | Branch to clone/pull |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_COLLECTION_NAME` | `aetherops_rag` | Qdrant collection |
| `QDRANT_BATCH_SIZE` | `100` | Upsert batch size |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `CHUNK_LINES` | `30` | Lines per chunk window |
| `CHUNK_OVERLAP` | `5` | Overlapping lines between chunks |
| `ALLOWED_EXTENSIONS` | `py,js,ts,...` | File extensions to index |
| `EXCLUDED_DIRS` | `.git,.venv,.tmp,...` | Directory names to skip |
| `LOG_LEVEL` | `INFO` | Python logging level |
