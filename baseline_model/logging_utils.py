"""Logging setup shared by the engine modules."""
import logging
import os

# Module loggers live under this prefix, so setting its level covers all of them.
LOGGER_NAME = __name__.rsplit(".", 1)[0]
ENV_LEVEL = "LTX25_LOG_LEVEL"
DEFAULT_LEVEL = logging.INFO

_logger = logging.getLogger(LOGGER_NAME)

if not any(isinstance(handler, logging.NullHandler) for handler in _logger.handlers):
    _logger.addHandler(logging.NullHandler())

_requested_level = os.environ.get(ENV_LEVEL, "").strip().upper()
_logger.setLevel(logging.getLevelNamesMapping().get(_requested_level, DEFAULT_LEVEL))

def get_logger(name):
    """Return the logger for one engine module"""
    return logging.getLogger(name)
