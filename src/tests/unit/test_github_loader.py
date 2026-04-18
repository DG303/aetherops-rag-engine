# tests/unit/test_github_loader.py
import os
import shutil
import subprocess
from unittest.mock import MagicMock

import pytest

from rag_core.indexing.github_loader import (
    clone_repo,
    detect_language,
    extract_repo_name,
    get_git_dependencies,
    is_allowed_extension,
    load_repo_files,
    normalize_path,
    prepare_tmp_dir,
    pull_latest_changes,
    read_repo_files,
    read_text_with_fallback,
    should_exclude_path,
    sync_repo,
)


# --- extract_repo_name ---
def test_extract_repo_name_https_ssh():
    assert extract_repo_name("git@github.com:test/test.git") == "test"
    assert extract_repo_name("https://github.com/test/test.git") == "test"


def test_extract_repo_name_strips_git_suffix():
    assert extract_repo_name("https://github.com/test/test.git") == "test"


def test_extract_repo_name_strips_trailing_slash():
    assert extract_repo_name("https://github.com/test/test.git/") == "test"


def test_extract_repo_name_empty_raises():
    with pytest.raises(ValueError):
        extract_repo_name("")


# --- normalize_path ---
def test_normalize_path_converts_backslashes_to_forwardslashes():
    assert (
        normalize_path("C:\\Users\\test\\Documents\\test.txt") == "c:/users/test/documents/test.txt"
    )


def test_normalize_path_lowercases_path():
    assert normalize_path("C:/Users/Test/MAIN.py") == "c:/users/test/main.py"


def test_normalize_path_removes_trailing_slash():
    assert (
        normalize_path("C:\\Users\\test\\Documents\\test.txt/")
        == "c:/users/test/documents/test.txt"
    )


# --- should_exclude_path ---
def test_should_exclude_path_excludes_node_modules():
    assert should_exclude_path("project/node_modules/foo.js", {"node_modules"})


def test_should_exclude_path_allows_normal_path():
    assert not should_exclude_path("project/src/main.py", {"node_modules"})


def test_should_exclude_path_excludes_dot_git():
    assert should_exclude_path("project/.git/foo.js", {".git"})


def test_should_exclude_path_excludes_dot_tmp():
    assert should_exclude_path("project/.tmp/foo.js", {".tmp"})


def test_should_exclude_path_excludes_dot_venv():
    assert should_exclude_path("project/.venv/foo.js", {".venv"})


# --- is_allowed_extension ---
def test_is_allowed_extension_allows_python_files():
    assert is_allowed_extension("project/src/main.py", {".py"}, set())


def test_is_allowed_extension_allows_javascript_files():
    assert is_allowed_extension("project/src/main.js", {".js"}, set())


def test_is_allowed_extension_allows_html_files():
    assert is_allowed_extension("project/src/index.html", {".html"}, set())


def test_is_allowed_extension_allows_css_files():
    assert is_allowed_extension("project/src/style.css", {".css"}, set())


def test_is_allowed_extension_allows_json_files():
    assert is_allowed_extension("project/src/config.json", {".json"}, set())


# --- read_text_with_fallback ---
def test_read_text_with_fallback_reads_text_file(tmp_path):
    f = tmp_path / "main.py"
    f.write_text("print('Hello, World!')", encoding="utf-8")
    assert read_text_with_fallback(str(f)) == "print('Hello, World!')"


def test_read_text_with_fallback_returns_none_for_binary_file(tmp_path):
    f = tmp_path / "main.bin"
    f.write_bytes(b"\x00\x01\x02")
    assert read_text_with_fallback(str(f)) is None


def test_read_text_with_fallback_returns_none_for_unreadable_file(tmp_path):
    assert read_text_with_fallback(str(tmp_path / "nonexistent.txt")) is None


# --- read_repo_files ---
def test_read_repo_files_loads_allowed_files(tmp_path):
    """Should load .py files from a fake repo directory."""
    (tmp_path / "main.py").write_text("print('hello')", encoding="utf-8")
    docs = read_repo_files(str(tmp_path), "my-repo", "main")
    assert len(docs) == 1
    assert docs[0].repo_name == "my-repo"
    assert docs[0].branch == "main"
    assert docs[0].content == "print('hello')"


def test_read_repo_files_skips_excluded_dirs(tmp_path):
    """Files inside excluded directories should be ignored."""
    excluded = tmp_path / ".venv"
    excluded.mkdir()
    (excluded / "activate.py").write_text("# venv activate", encoding="utf-8")
    docs = read_repo_files(str(tmp_path), "my-repo", "main")
    assert len(docs) == 0


def test_read_repo_files_skips_binary_files(tmp_path):
    """Binary files should be skipped silently."""
    (tmp_path / "data.py").write_bytes(b"\x00\x01\x02")
    docs = read_repo_files(str(tmp_path), "my-repo", "main")
    assert len(docs) == 0


def test_read_repo_files_raises_if_dir_missing(tmp_path):
    """Should raise FileNotFoundError if repo dir doesn't exist."""
    with pytest.raises(FileNotFoundError):
        read_repo_files(str(tmp_path / "nonexistent"), "my-repo", "main")


def test_read_repo_files_detects_language(tmp_path):
    (tmp_path / "main.py").write_text("print('Hello, World!')", encoding="utf-8")
    docs = read_repo_files(str(tmp_path), "my-repo", "main")
    assert len(docs) == 1
    assert docs[0].language == "python"
    assert docs[0].content == "print('Hello, World!')"
    assert docs[0].repo_name == "my-repo"
    assert docs[0].branch == "main"
    assert docs[0].path == "main.py"


# --- extract_repo_name (empty name after .git strip, line 164) ---


def test_extract_repo_name_raises_when_name_empty_after_cleaning():
    with pytest.raises(ValueError):
        extract_repo_name("https://github.com/.git")


# --- load_repo_files ---


def test_load_repo_files_calls_pipeline_in_order(monkeypatch):
    import rag_core.indexing.github_loader as gl

    fake_docs = [MagicMock()]
    mock_prepare = MagicMock()
    mock_sync = MagicMock()
    mock_read = MagicMock(return_value=fake_docs)
    monkeypatch.setattr(gl, "prepare_tmp_dir", mock_prepare)
    monkeypatch.setattr(gl, "sync_repo", mock_sync)
    monkeypatch.setattr(gl, "read_repo_files", mock_read)

    result = load_repo_files("https://github.com/org/myrepo.git", "main")

    mock_prepare.assert_called_once()
    mock_sync.assert_called_once()
    mock_read.assert_called_once()
    assert result == fake_docs


# --- prepare_tmp_dir ---


def test_prepare_tmp_dir_creates_directory(monkeypatch, tmp_path):
    import rag_core.indexing.github_loader as gl

    target = str(tmp_path / "new_tmp")
    monkeypatch.setattr(gl, "TMP_DIR", target)
    prepare_tmp_dir()
    assert os.path.isdir(target)


def test_prepare_tmp_dir_raises_on_os_error(monkeypatch):
    monkeypatch.setattr(os, "makedirs", MagicMock(side_effect=OSError("permission denied")))
    with pytest.raises(OSError):
        prepare_tmp_dir()


# --- sync_repo ---


def test_sync_repo_pulls_when_git_dir_exists(tmp_path, monkeypatch):
    import rag_core.indexing.github_loader as gl

    (tmp_path / ".git").mkdir()
    mock_pull = MagicMock()
    monkeypatch.setattr(gl, "pull_latest_changes", mock_pull)
    monkeypatch.setattr(gl, "clone_repo", MagicMock())

    sync_repo("https://github.com/org/repo.git", "main", str(tmp_path))

    mock_pull.assert_called_once_with("https://github.com/org/repo.git", "main", str(tmp_path))


def test_sync_repo_removes_and_clones_when_dir_exists_without_git(tmp_path, monkeypatch):
    import rag_core.indexing.github_loader as gl

    mock_rmtree = MagicMock()
    mock_clone = MagicMock()
    monkeypatch.setattr(shutil, "rmtree", mock_rmtree)
    monkeypatch.setattr(gl, "clone_repo", mock_clone)

    sync_repo("https://github.com/org/repo.git", "main", str(tmp_path))

    mock_rmtree.assert_called_once_with(str(tmp_path))
    mock_clone.assert_called_once_with("https://github.com/org/repo.git", "main", str(tmp_path))


def test_sync_repo_clones_when_dir_does_not_exist(tmp_path, monkeypatch):
    import rag_core.indexing.github_loader as gl

    repo_dir = str(tmp_path / "nonexistent")
    mock_clone = MagicMock()
    mock_rmtree = MagicMock()
    monkeypatch.setattr(gl, "clone_repo", mock_clone)
    monkeypatch.setattr(shutil, "rmtree", mock_rmtree)

    sync_repo("https://github.com/org/repo.git", "main", repo_dir)

    mock_clone.assert_called_once_with("https://github.com/org/repo.git", "main", repo_dir)
    mock_rmtree.assert_not_called()


# --- clone_repo ---


def test_clone_repo_succeeds_when_returncode_is_zero(monkeypatch):
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=MagicMock(returncode=0)))
    clone_repo("https://github.com/org/repo.git", "main", "/tmp/repo")  # no exception = success


def test_clone_repo_raises_on_nonzero_returncode(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(return_value=MagicMock(returncode=1, stderr="fatal: repo not found")),
    )
    with pytest.raises(RuntimeError, match="Git clone failed"):
        clone_repo("https://github.com/org/repo.git", "main", "/tmp/repo")


# --- pull_latest_changes ---
# get_git_dependencies is monkeypatched to return a fake Repo class and a fake
# GitCommandError type so no real git operations are performed.


def _fake_git_deps(fake_repo, FakeGitCommandError):
    return lambda: (MagicMock(return_value=fake_repo), FakeGitCommandError)


def test_pull_checks_out_and_pulls_existing_branch(monkeypatch):
    import rag_core.indexing.github_loader as gl

    URL = "https://github.com/org/repo.git"
    FakeGCE = type("GitCommandError", (Exception,), {})
    fake_repo = MagicMock()
    fake_repo.remotes.origin.urls = iter([URL])
    fake_repo.heads = ["main"]
    monkeypatch.setattr(gl, "get_git_dependencies", _fake_git_deps(fake_repo, FakeGCE))

    pull_latest_changes(URL, "main", "/tmp/repo")

    fake_repo.remotes.origin.fetch.assert_called_once()
    fake_repo.git.checkout.assert_called_once_with("main")
    fake_repo.remotes.origin.pull.assert_called_once_with("main")


def test_pull_creates_branch_when_not_in_heads(monkeypatch):
    import rag_core.indexing.github_loader as gl

    URL = "https://github.com/org/repo.git"
    FakeGCE = type("GitCommandError", (Exception,), {})
    fake_repo = MagicMock()
    fake_repo.remotes.origin.urls = iter([URL])
    fake_repo.heads = []  # "feature" not present
    monkeypatch.setattr(gl, "get_git_dependencies", _fake_git_deps(fake_repo, FakeGCE))

    pull_latest_changes(URL, "feature", "/tmp/repo")

    fake_repo.git.checkout.assert_called_once_with("-B", "feature", "origin/feature")


def test_pull_reclones_when_remote_url_changes(monkeypatch):
    import rag_core.indexing.github_loader as gl

    OLD_URL = "https://github.com/org/old-repo.git"
    NEW_URL = "https://github.com/org/new-repo.git"
    FakeGCE = type("GitCommandError", (Exception,), {})
    fake_repo = MagicMock()
    fake_repo.remotes.origin.urls = iter([OLD_URL])
    monkeypatch.setattr(gl, "get_git_dependencies", _fake_git_deps(fake_repo, FakeGCE))
    mock_rmtree = MagicMock()
    mock_clone = MagicMock()
    monkeypatch.setattr(shutil, "rmtree", mock_rmtree)
    monkeypatch.setattr(gl, "clone_repo", mock_clone)

    pull_latest_changes(NEW_URL, "main", "/tmp/repo")

    mock_rmtree.assert_called_once_with("/tmp/repo")
    mock_clone.assert_called_once_with(NEW_URL, "main", "/tmp/repo")


def test_pull_reclones_on_git_command_error(monkeypatch):
    import rag_core.indexing.github_loader as gl

    URL = "https://github.com/org/repo.git"
    FakeGCE = type("GitCommandError", (Exception,), {})
    fake_repo = MagicMock()
    fake_repo.remotes.origin.urls = iter([URL])
    fake_repo.heads = ["main"]
    fake_repo.remotes.origin.pull.side_effect = FakeGCE("pull failed")
    monkeypatch.setattr(gl, "get_git_dependencies", _fake_git_deps(fake_repo, FakeGCE))
    mock_rmtree = MagicMock()
    mock_clone = MagicMock()
    monkeypatch.setattr(shutil, "rmtree", mock_rmtree)
    monkeypatch.setattr(gl, "clone_repo", mock_clone)

    pull_latest_changes(URL, "main", "/tmp/repo")

    mock_rmtree.assert_called_once_with("/tmp/repo")
    mock_clone.assert_called_once()


def test_pull_reraises_unexpected_exception(monkeypatch):
    import rag_core.indexing.github_loader as gl

    URL = "https://github.com/org/repo.git"
    FakeGCE = type("GitCommandError", (Exception,), {})
    fake_repo = MagicMock()
    fake_repo.remotes.origin.urls = iter([URL])
    fake_repo.heads = ["main"]
    fake_repo.remotes.origin.fetch.side_effect = RuntimeError("network down")
    monkeypatch.setattr(gl, "get_git_dependencies", _fake_git_deps(fake_repo, FakeGCE))

    with pytest.raises(RuntimeError, match="network down"):
        pull_latest_changes(URL, "main", "/tmp/repo")


# --- get_git_dependencies (ImportError branch, lines 169-175) ---


def test_get_git_dependencies_raises_if_gitpython_missing(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "git", None)
    with pytest.raises(RuntimeError, match="GitPython is required"):
        get_git_dependencies()


# --- is_allowed_extension (exact filename branch, line 181) ---


def test_is_allowed_extension_allows_exact_filename():
    assert is_allowed_extension("project/docker-compose.yml", set(), {"docker-compose.yml"}) is True


# --- detect_language (exact filename match, line 218) ---


def test_detect_language_matches_exact_filename():
    assert detect_language("/path/to/Dockerfile") == "docker"
    assert detect_language("/path/to/Makefile") == "make"


# --- read_text_with_fallback (all encodings fail, lines 210-212) ---


def test_read_text_with_fallback_returns_none_when_all_encodings_fail(tmp_path, monkeypatch):
    import rag_core.indexing.github_loader as gl

    # Restrict to ascii-only so the byte 0x81 causes UnicodeDecodeError on every attempt
    monkeypatch.setattr(gl.config, "READ_ENCODINGS", ["ascii"])
    f = tmp_path / "weird.py"
    f.write_bytes(b"\x81")
    assert read_text_with_fallback(str(f)) is None
