# src/rag_core/indexing/github_loader.py
import logging
import os
import shutil
import subprocess

from rag_core.config.config import config
from rag_core.config.logging_config import setup_logging
from rag_core.models.document import Document

setup_logging()
logger = logging.getLogger(__name__)

TMP_DIR = ".tmp"


def load_repo_files(repo_url: str = config.GITHUB_REPO, branch: str = config.GITHUB_BRANCH) -> list[Document]:
    repo_name = extract_repo_name(repo_url)
    repo_dir = os.path.join(TMP_DIR, repo_name)
    logger.info(
        "Loading files from repo '%s' on branch '%s' into '%s'",
        repo_name,
        branch,
        repo_dir,
    )

    prepare_tmp_dir()
    sync_repo(repo_url, branch, repo_dir)
    return read_repo_files(repo_dir, repo_name, branch)


def prepare_tmp_dir() -> None:
    try:
        os.makedirs(TMP_DIR, exist_ok=True)
    except OSError:
        logger.exception("Unable to prepare temporary directory at %s", TMP_DIR)
        raise


def sync_repo(repo_url: str, branch: str, repo_dir: str) -> None:
    if os.path.isdir(os.path.join(repo_dir, ".git")):
        logger.debug("Repository exists at %s. Pulling latest changes.", repo_dir)
        pull_latest_changes(repo_url, branch, repo_dir)
        return

    if os.path.exists(repo_dir):
        logger.warning(
            "Path exists but is not a git repository. Recreating directory: %s", repo_dir
        )
        shutil.rmtree(repo_dir)

    clone_repo(repo_url, branch, repo_dir)


def clone_repo(repo_url: str, branch: str, repo_dir: str) -> None:
    logger.debug("Cloning repo into %s", repo_dir)
    try:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_TEMPLATE_DIR": ""}
        result = subprocess.run(
            ["git", "clone", "--branch", branch, repo_url, repo_dir],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Git clone failed for URL '{repo_url}' and branch '{branch}':\n{result.stderr}"
            )
    except Exception:
        logger.exception("Unexpected error while cloning repo '%s'", repo_url)
        raise


def pull_latest_changes(repo_url: str, branch: str, repo_dir: str) -> None:
    Repo, GitCommandError = get_git_dependencies()

    try:
        repo = Repo(repo_dir)
        origin = repo.remotes.origin
        current_origin_url = next(iter(origin.urls), None)
        if current_origin_url and current_origin_url.rstrip("/") != repo_url.rstrip("/"):
            logger.warning(
                "Remote URL changed from %s to %s. Re-cloning repository.",
                current_origin_url,
                repo_url,
            )
            shutil.rmtree(repo_dir)
            clone_repo(repo_url, branch, repo_dir)
            return

        origin.fetch()
        if branch in repo.heads:
            repo.git.checkout(branch)
        else:
            repo.git.checkout("-B", branch, f"origin/{branch}")

        origin.pull(branch)
    except GitCommandError:
        logger.warning(
            "Pull failed for '%s' (%s). Re-cloning repository '%s'",
            repo_url,
            branch,
            repo_dir,
            exc_info=True,
        )
        shutil.rmtree(repo_dir)
        clone_repo(repo_url, branch, repo_dir)
    except Exception:
        logger.exception("Unexpected error while pulling latest changes for repo '%s'", repo_url)
        raise


def read_repo_files(repo_dir: str, repo_name: str, branch: str) -> list[Document]:
    if not os.path.isdir(repo_dir):
        raise FileNotFoundError(f"Cloned repository does not exist: {repo_dir}")

    allowed_suffixes = config.ALLOWED_SUFFIXES
    allowed_filenames = config.ALLOWED_FILENAMES
    excluded_dir_names = config.EXCLUDED_DIR_NAMES

    collected_files: list[Document] = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d.lower() not in excluded_dir_names]

        for file_name in files:
            file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(file_path, repo_dir)
            if should_exclude_path(relative_path, excluded_dir_names):
                continue
            if not is_allowed_extension(file_path, allowed_suffixes, allowed_filenames):
                continue

            content = read_text_with_fallback(file_path)
            if content is None:
                logger.debug("Skipping unreadable file: %s", file_path)
                continue

            language = detect_language(file_path)
            collected_files.append(
                Document(
                    repo_name=repo_name,
                    branch=branch,
                    path=relative_path,
                    language=language,
                    content=content,
                )
            )

    logger.info("Loaded %s files from %s", len(collected_files), repo_dir)
    return collected_files


def extract_repo_name(repo_url: str) -> str:
    cleaned_url = repo_url.strip().rstrip("/")
    if not cleaned_url:
        raise ValueError("Invalid repo URL")

    name_part = cleaned_url.split("/")[-1].split(":")[-1]
    if name_part.endswith(".git"):
        name_part = name_part[:-4]

    repo_name = name_part.strip()
    if not repo_name:
        raise ValueError(f"Invalid repo name from URL: {repo_url}")
    return repo_name


def get_git_dependencies():
    try:
        from git import GitCommandError, Repo
    except ImportError as exc:
        raise RuntimeError(
            "GitPython is required for GitHub loading. Install with `pip install GitPython`."
        ) from exc
    return Repo, GitCommandError


def is_allowed_extension(path: str, allowed_suffixes: set[str], allowed_filenames: set[str]) -> bool:
    file_name = os.path.basename(path).lower()
    if file_name in allowed_filenames:
        return True
    _, ext = os.path.splitext(file_name)
    return ext.lower() in allowed_suffixes


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").lower().rstrip("/")


def should_exclude_path(path: str, excluded_dir_names: set[str]) -> bool:
    normalized_path = normalize_path(path)
    path_parts = [part for part in normalized_path.split("/") if part]
    return any(part in excluded_dir_names for part in path_parts)


def read_text_with_fallback(file_path: str) -> str | None:
    try:
        with open(file_path, "rb") as file_obj:
            raw = file_obj.read()
    except OSError:
        logger.exception("Unable to read file: %s", file_path)
        return None

    if b"\x00" in raw:
        return None

    for encoding in config.READ_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def detect_language(file_path: str) -> str:
    file_name = os.path.basename(file_path).lower()
    if file_name in config.LANGUAGE_MAP:
        return config.LANGUAGE_MAP[file_name]
    _, ext = os.path.splitext(file_name)
    return config.LANGUAGE_MAP.get(ext.lower(), "unknown")
