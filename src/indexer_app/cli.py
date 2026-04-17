# indexer_app/cli.py
import argparse
import logging

from rag_core.config.logging_config import setup_logging
from rag_core.main import run_indexer

setup_logging()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Indexer CLI")
    parser.add_argument("--repo-url", type=str, required=True, help="URL of the repository to index")
    parser.add_argument("--branch", type=str, required=True, help="Branch of the repository to index")
    args = parser.parse_args()
    run_indexer(args.repo_url, args.branch)

if __name__ == "__main__":
    main()
