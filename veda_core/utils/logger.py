# =============================================================================
# utils/logger.py
# VEDA POC — Centralised logging setup
#
# Usage in any pipeline file:
#   from utils.logger import get_logger
#   logger = get_logger(__name__)
#
# All pipeline steps write to logs/veda_pipeline.log (DEBUG+) and to the
# console (INFO+). Log file rotates at 10 MB, keeps 3 backups.
# =============================================================================

import logging
import os
from logging.handlers import RotatingFileHandler

_LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "veda_pipeline.log")
_FMT      = "%(asctime)s.%(msecs)03d | %(levelname)-5s | [%(name)s] %(message)s"
_DATEFMT  = "%Y-%m-%d %H:%M:%S"

_root_configured = False


def _configure_root() -> None:
    global _root_configured
    if _root_configured:
        return

    os.makedirs(_LOG_DIR, exist_ok=True)

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes   = 10 * 1024 * 1024,
        backupCount = 3,
        encoding   = "utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    root = logging.getLogger("veda")
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        root.addHandler(file_handler)
        root.addHandler(stream_handler)

    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger scoped to `name` under the 'veda' root logger.
    Call with __name__ from each pipeline module.
    """
    _configure_root()
    # Strip the module path prefix to keep names short (e.g. 'schema_scanner')
    short = name.split(".")[-1] if "." in name else name
    return logging.getLogger(f"veda.{short}")
