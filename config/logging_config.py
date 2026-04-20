"""
Logging Configuration

Sets up structured logging for the entire project. Import and call
`setup_logging()` at application entry points (API server, training scripts).
Uses LOG_LEVEL from the environment (default: INFO).
"""

import logging
import logging.config
import os
from pathlib import Path


def setup_logging(log_level: str | None = None) -> None:
    """Configure project-wide logging.

    Args:
        log_level: Override log level string (e.g. "DEBUG", "INFO"). Falls back
            to the LOG_LEVEL environment variable, then "INFO".
    """
    level = log_level or os.getenv("LOG_LEVEL", "INFO")

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": level,
            "handlers": ["console"],
        },
    }

    logging.config.dictConfig(config)
