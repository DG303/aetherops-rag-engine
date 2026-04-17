# tests/unit/test_chunk.py
from rag_core.chunking.chunk import chunk_document, generate_chunk_id
from rag_core.models.chunk import Chunk
from rag_core.models.document import Document


def test_chunk_count_no_overlap(simple_document):
    """100 lines, chunk_lines=20, overlap=0 should produce 5 chunks"""
    chunks = chunk_document(simple_document, chunk_lines=20, chunk_overlap=0)
    assert len(chunks) == 5

def test_chunk_count_with_overlap(simple_document):
    """100 lines, chunk_lines=20, overlap=5 should produce 6 chunks"""
    chunks = chunk_document(simple_document, chunk_lines=20, chunk_overlap=5)
    assert len(chunks) == 6

def test_chunk_content(simple_document):
    """chunk_document should return a non-empty list of Chunk objects"""
    chunks = chunk_document(simple_document, chunk_lines=20, chunk_overlap=0)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)

def test_chunk_preserves_metadata(simple_document):
    """Every chunk should carry the document's metadata"""
    chunks = chunk_document(simple_document, chunk_lines=20, chunk_overlap=0)
    for chunk in chunks:
        assert chunk.repo_name == simple_document.repo_name
        assert chunk.branch == simple_document.branch
        assert chunk.source_path == simple_document.path
        assert chunk.language == simple_document.language

def test_chunk_content_is_correct(simple_document):
    """The content of each chunk should be the correct window of lines"""
    chunks = chunk_document(simple_document, chunk_lines=20, chunk_overlap=0)
    for i, chunk in enumerate(chunks):
        expected_content = "\n".join(f"line {j} AetherOps RAG Engine testing chunking" for j in range(i*20+1, (i+1)*20+1))
        assert chunk.content == expected_content

def test_overlap_produces_repeated_lines(simple_document):
    """With overlap, the shared lines between consecutive chunks should match"""
    chunks = chunk_document(simple_document, chunk_lines=20, chunk_overlap=5)
    for i, chunk in enumerate(chunks):
        if i > 0:
            assert chunk.content.splitlines()[:5] == chunks[i - 1].content.splitlines()[-5:]
        if i < len(chunks) - 1:
            assert chunk.content.splitlines()[-5:] == chunks[i + 1].content.splitlines()[:5]

def test_single_chunk_when_content_fits(simple_document):
    """With content shorter than chunk_lines, should produce a single chunk"""
    chunks = chunk_document(simple_document, chunk_lines=100, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0].content == simple_document.content

def test_single_chunk_when_content_fits_with_overlap(simple_document):
    """When document fits exactly in one chunk, overlap should not produce a second chunk"""
    chunks = chunk_document(simple_document, chunk_lines=100, chunk_overlap=5)
    assert len(chunks) == 1
    assert chunks[0].content == simple_document.content

def test_empty_document_returns_no_chunks():
    """An empty document should return no chunks"""
    empty_doc = Document(
        repo_name="test-repo",
        branch="main",
        path="src/empty.txt",
        language="text",
        content="",
    )
    chunks = chunk_document(empty_doc, chunk_lines=20, chunk_overlap=0)
    assert len(chunks) == 0

def test_chunk_lines_too_large_for_document(simple_document):
    """If chunk_lines is larger than the document, should produce a single chunk"""
    chunks = chunk_document(simple_document, chunk_lines=1000, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0].content == simple_document.content

def test_generate_chunk_id_format(simple_document):
    """chunk.id should match generate_chunk_id output for each index"""
    chunks = chunk_document(simple_document, chunk_lines=20, chunk_overlap=0)
    for i, chunk in enumerate(chunks):
        assert chunk.id == generate_chunk_id(simple_document, i)

def test_chunk_id_uniqueness(simple_document):
    """Each chunk should have a unique ID"""
    chunks = chunk_document(simple_document, chunk_lines=20, chunk_overlap=0)
    ids = [chunk.id for chunk in chunks]
    assert len(ids) == len(set(ids))

def test_chunk_id_format(simple_document):
    """The chunk ID should be in the format 'repo_name|branch|path|chunk_index'"""
    chunks = chunk_document(simple_document, chunk_lines=20, chunk_overlap=0)
    for i, chunk in enumerate(chunks):
        assert chunk.id == f"{simple_document.repo_name}|{simple_document.branch}|{simple_document.path}|{i}"

def test_chunk_documents_batches_multiple_docs(simple_document):
    from rag_core.chunking.chunk import chunk_document, chunk_documents
    docs = [simple_document, simple_document]
    chunks = chunk_documents(docs)
    # chunk_documents uses config defaults, so derive the expected count
    # rather than hardcoding a number that would break if config changes
    expected_per_doc = len(chunk_document(simple_document))
    assert len(chunks) == expected_per_doc * 2
