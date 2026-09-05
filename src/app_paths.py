"""Filesystem locations owned by the installed desktop application."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping


APP_DIR_NAME = "SAM3-AutoAnnotator"
APP_HOME_ENV = "SAM3_AUTOANNOTATOR_HOME"


def user_data_dir(
    *,
    platform_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Return an OS-appropriate writable application-data directory."""
    platform_name = sys.platform if platform_name is None else platform_name
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else Path(home)

    override = environ.get(APP_HOME_ENV)
    if override:
        return Path(override).expanduser()

    if platform_name.startswith("win"):
        base = environ.get("LOCALAPPDATA") or environ.get("APPDATA")
        return (Path(base) if base else home / "AppData" / "Local") / APP_DIR_NAME

    if platform_name == "darwin":
        return home / "Library" / "Application Support" / APP_DIR_NAME

    base = environ.get("XDG_DATA_HOME")
    return (Path(base) if base else home / ".local" / "share") / "sam3-autoannotator"


APP_DATA_DIR = user_data_dir()
MODELS_DIR = APP_DATA_DIR / "models"
OUTPUTS_DIR = APP_DATA_DIR / "projects"


def discover_default_model():
    """Return the preferred checkpoint from the user-data model directory."""
    if not MODELS_DIR.is_dir():
        return None
    candidates = sorted(MODELS_DIR.glob("*.pt"))
    preferred = next(
        (path for path in candidates if path.name.casefold().startswith("sam3")),
        None,
    )
    return preferred or (candidates[0] if len(candidates) == 1 else None)
