"""Logging configuration for the application."""

import logging
import logging.config
from pathlib import Path

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"

DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_level: str = "INFO",
    log_dir: str | None = None,
    log_file: str = "application.log",
    enable_console: bool = True,
) -> None:
    """Set up logging configuration.

    Args:
        log_level: Logging level as a string (e.g., "DEBUG", "INFO").
        log_dir: Optional directory to save log files. If None, file logging is disabled.
        log_file: Log file name if log_dir is provided.
        enable_console: Whether to enable console logging.

    """
    handlers = {}
    root_handlers: list[str] = []

    if enable_console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "default",
            "stream": "ext://sys.stdout",
        }
        root_handlers.append("console")

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "default",
            "filename": str(log_path / log_file),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "encoding": "utf-8",
        }
        root_handlers.append("file")

    logging_config = { # pyright: ignore[reportUnknownVariableType]
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": DEFAULT_LOG_FORMAT,
                "datefmt": DEFAULT_DATE_FORMAT,
            },
        },
        "handlers": handlers,
        "root": {
            "level": log_level,
            "handlers": root_handlers,
        },
        # 🔑 ВАЖНО: уровни сторонних логгеров
        "loggers": {
            "httpx": {
                "level": "WARNING",
            },
            "openai": {
                "level": "WARNING",
            },
            "urllib3": {
                "level": "WARNING",
            },
        },
    }

    logging.config.dictConfig(logging_config) # pyright: ignore[reportUnknownArgumentType]
