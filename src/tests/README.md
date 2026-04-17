# Tests

Unit tests for the AetherOps RAG Engine. All tests are isolated — no running Qdrant instance, no network access, and no embedding model is loaded. External dependencies are replaced with mocks.

**Current coverage: 95%**

---

## Running the tests

```bash
# Full suite with coverage report (from project root)
.venv/bin/pytest src/tests/

# Single test file
.venv/bin/pytest src/tests/unit/test_chunk.py

# Single test by name
.venv/bin/pytest src/tests/unit/test_chunk.py::test_chunk_count_no_overlap

# With enforced coverage floor
.venv/bin/pytest src/tests/ --cov-fail-under=90

# HTML coverage report (opens in browser)
.venv/bin/pytest src/tests/ --cov-report=html
open htmlcov/index.html
```

---

## Structure

```
src/tests/
├── conftest.py          # Shared fixtures (simple_document)
└── unit/
    ├── test_chunk.py        # Chunking algorithm
    ├── test_embedder.py     # Embedding pipeline
    ├── test_github_loader.py # File loading, git sync, path utilities
    ├── test_main.py         # Pipeline orchestration (run_indexer)
    ├── test_qdrant_store.py # Vector storage
    └── test_retriever.py    # Query embedding and retrieval
```

---

## What each file covers

### `test_chunk.py` — 14 tests
Covers `chunk_document`, `chunk_documents`, and `generate_chunk_id`.

| Test | What it verifies |
|---|---|
| `test_chunk_count_no_overlap` | 100 lines / 20-line chunks = 5 chunks |
| `test_chunk_count_with_overlap` | 5-line overlap produces 6 complete windows |
| `test_chunk_content_is_correct` | Each chunk holds the exact line window |
| `test_overlap_produces_repeated_lines` | Shared lines between consecutive chunks match |
| `test_chunk_preserves_metadata` | repo, branch, path, language copied to every chunk |
| `test_empty_document_returns_no_chunks` | Empty content → zero chunks |
| `test_single_chunk_when_content_fits` | Document ≤ chunk size → single chunk |
| `test_chunk_lines_too_large_for_document` | Oversized window → single chunk with all content |
| `test_chunk_id_format` | ID format is `repo\|branch\|path\|index` |
| `test_chunk_id_uniqueness` | No two chunks share an ID |
| `test_chunk_documents_batches_multiple_docs` | Batch wrapper sums chunks from all documents |

### `test_embedder.py` — 1 test
Covers `embed_chunks` with a fake model injected directly.

| Test | What it verifies |
|---|---|
| `test_embed_chunks_returns_embedded_chunks` | Chunk fields and embeddings are mapped correctly; `encode` called with right args |

### `test_github_loader.py` — 43 tests
Covers every function in `github_loader.py` (98% line coverage). External calls — `subprocess.run`, `shutil.rmtree`, GitPython — are monkeypatched.

| Group | Tests |
|---|---|
| `extract_repo_name` | HTTPS, SSH, trailing slash, empty URL, empty name after `.git` strip |
| `normalize_path` | Backslash → forward slash, lowercase, trailing slash removal |
| `should_exclude_path` | node_modules, .git, .tmp, .venv; normal path passes |
| `is_allowed_extension` | .py .js .html .css .json by suffix; exact filename match |
| `read_text_with_fallback` | UTF-8 text, binary (null bytes), unreadable file, all encodings fail |
| `detect_language` | Extension-based (.py → python) and exact filename (Dockerfile → docker) |
| `read_repo_files` | Loads .py files, skips excluded dirs, skips binary, raises on missing dir |
| `load_repo_files` | Calls prepare → sync → read in order |
| `prepare_tmp_dir` | Creates directory; re-raises OSError |
| `sync_repo` | Pulls when .git exists; rmtree+clone when dir has no .git; fresh clone |
| `clone_repo` | Succeeds on returncode 0; raises RuntimeError on nonzero |
| `pull_latest_changes` | Checkout existing branch; create missing branch; re-clone on URL change; re-clone on GitCommandError; re-raise unexpected exception |
| `get_git_dependencies` | Raises RuntimeError when GitPython is not installed |

### `test_main.py` — 6 tests
Covers `run_indexer` by mocking all four pipeline steps (100% coverage).

| Test | What it verifies |
|---|---|
| `test_run_indexer_calls_load_repo_files` | URL and branch forwarded correctly |
| `test_run_indexer_passes_docs_to_chunk_documents` | Output of step 1 → input of step 2 |
| `test_run_indexer_passes_chunks_to_embed_chunks` | Output of step 2 → input of step 3 |
| `test_run_indexer_passes_embeddings_to_store` | Output of step 3 → input of step 4 |
| `test_run_indexer_executes_steps_in_order` | Steps run in the correct sequence |
| `test_run_indexer_empty_repo_runs_without_error` | Zero documents completes without raising |

### `test_qdrant_store.py` — 5 tests
Covers `store_embeddings` and `ensure_collection_exists` with a fake QdrantClient.

| Test | What it verifies |
|---|---|
| `test_store_embeddings_calls_upsert` | `upsert` is called for non-empty input |
| `test_store_embeddings_empty_list_does_not_upsert` | Early return on empty list |
| `test_store_embeddings_creates_collection_when_missing` | `create_collection` called when collection absent |
| `test_store_embeddings_skips_collection_creation_when_exists` | No `create_collection` call when collection exists |
| `test_ensure_collection_exists_creates_with_correct_vector_size` | `vectors_config.size` matches embedding dimension |

### `test_retriever.py` — 8 tests
Covers `embed_query`, `retrieve`, `get_model`, and `get_client`. The `_model` and `_client` singletons are replaced via `monkeypatch` so no real model or Qdrant connection is used.

| Test | What it verifies |
|---|---|
| `test_embed_query_calls_model_encode` | `encode` called with `normalize_embeddings=True`; result returned as list |
| `test_embed_query_returns_list` | Return type is `list`, not ndarray |
| `test_retrieve_returns_retrieval_results` | Qdrant payload mapped to `RetrievalResult` fields |
| `test_retrieve_empty_results` | Empty Qdrant response → empty list |
| `test_retrieve_multiple_results` | All hits returned in order with correct scores |
| `test_retrieve_passes_limit_to_qdrant` | `limit` arg forwarded to `query_points` |
| `test_get_model_returns_same_instance` | Singleton: same object on repeated calls |
| `test_get_client_returns_same_instance` | Singleton: same object on repeated calls |

---

## Key patterns

**Monkeypatching module-level names**
Functions are patched where they are *used*, not where they are defined:
```python
import rag_core.main as main_module
monkeypatch.setattr(main_module, "load_repo_files", mock_load)
```

**Injecting fake singletons**
Lazy `_model` / `_client` globals are replaced before the function under test runs:
```python
monkeypatch.setattr(retriever_module, "_model", fake_model)
```

**Faking GitPython**
`get_git_dependencies` is monkeypatched to return a fake `Repo` class and a custom `GitCommandError` type, keeping all git tests fully in-memory.

**Forcing ImportError**
Setting `sys.modules["git"] = None` makes the import machinery raise `ImportError` inside `get_git_dependencies`:
```python
monkeypatch.setitem(sys.modules, "git", None)
```
