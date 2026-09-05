from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PySide6.QtCore import QStandardPaths


LOG_NAME = "sam3_auto_annotator.log"


def configure_logging():
    """Configure one rotating diagnostic log and return its path when available."""
    root = logging.getLogger()
    if getattr(root, "_sam3_auto_annotator_configured", False):
        return getattr(root, "_sam3_auto_annotator_log_path", None)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root.setLevel(logging.INFO)

    log_path = None
    try:
        data_dir = Path(QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation))
        data_dir.mkdir(parents=True, exist_ok=True)
        log_path = data_dir / LOG_NAME
        handler = RotatingFileHandler(
            log_path,
            maxBytes=2_000_000,
            backupCount=2,
            encoding="utf-8",
        )
    except OSError:
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    root.addHandler(handler)
    root._sam3_auto_annotator_configured = True
    root._sam3_auto_annotator_log_path = log_path
    return log_path
