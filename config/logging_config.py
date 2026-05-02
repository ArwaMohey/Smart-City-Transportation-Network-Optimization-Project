"""Logging configuration for the Smart City Transportation Network System.

All modules must obtain their loggers via :func:`get_logger` so that the
application-wide format and level are consistently applied.  No module
outside this file should call :func:`logging.basicConfig` or manually
attach handlers to the root logger.

Usage example::

    from config.logging_config import setup_logging, get_logger

    setup_logging(level=logging.DEBUG, log_file="city_transport.log")
    logger = get_logger(__name__)
    logger.info("System started.")
"""

import logging
import logging.handlers
import sys
from typing import Optional


# Internal namespace used as the root for all application loggers.
_APP_ROOT = "smart_city"

# Default log-record format (ISO-8601 timestamp + level + origin + message).
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Rotate log files at 10 MB, keep at most 5 backups.
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5


def setup_logging(
    level: int = logging.DEBUG,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Configure application-wide logging.

    Must be called once at application start-up before any other module
    obtains a logger.  Calling it again replaces the existing handlers so
    the log level or file can be changed at runtime.

    Args:
        level:      Minimum severity to emit (default: ``logging.DEBUG``).
        log_file:   Optional path for a rotating file handler.  If *None*,
                    only the console (stdout) handler is registered.
        log_format: Optional override for the log-record format string.
                    Falls back to the built-in format when omitted.

    Returns:
        The configured application root logger (``smart_city``).
    """
    fmt = log_format or _DEFAULT_FORMAT
    formatter = logging.Formatter(fmt, datefmt=_DATE_FORMAT)

    app_logger = logging.getLogger(_APP_ROOT)
    app_logger.setLevel(level)

    # Remove stale handlers to prevent duplicate emission on repeated calls.
    app_logger.handlers.clear()

    # --- Console handler (stdout) -----------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    app_logger.addHandler(console_handler)

    # --- Optional rotating-file handler -----------------------------------
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        app_logger.addHandler(file_handler)

    app_logger.debug("Logging configured (level=%s, file=%s).", level, log_file)
    return app_logger


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger under the ``smart_city`` namespace.

    Args:
        name: Dotted module or component name, e.g. ``"core.graph"``.

    Returns:
        A :class:`logging.Logger` instance ready for use.

    Example::

        logger = get_logger("core.graph")
        logger.info("Graph initialised.")
    """
    return logging.getLogger(f"{_APP_ROOT}.{name}")
