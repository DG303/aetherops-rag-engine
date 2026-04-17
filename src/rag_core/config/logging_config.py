# src/rag_core/config/logging_config.py
import logging

from rag_core.config.config import config


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s  -- %(name)s -- %(message)s',
    )
