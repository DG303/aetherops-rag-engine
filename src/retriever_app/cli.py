# retriever_app/cli.py
import argparse
import logging

from rag_core.config.logging_config import setup_logging
from rag_core.retrieval.retriever import retrieve

setup_logging()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Retriever CLI")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    logger.info("Retrieving results for query: %s", args.query)
    results = retrieve(args.query, args.limit)

    if not results:
        print("No results found.")
        return

    for i, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"[{i}] {result.source_path} ({result.language}) score={result.score:.4f}")
        print("-" * 80)
        print(result.content[:1000])
        print()


if __name__ == "__main__":
    main()
