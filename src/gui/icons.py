"""Bundled vector icons: the same symbols on Windows, macOS and Linux."""

from pathlib import Path
from PySide6.QtGui import QIcon

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
ICONS = {
    name: name
    for name in (
        "focus",
        "app",
        "image",
        "folder",
        "state",
        "save",
        "sam3",
        "draw",
        "trash",
        "export",
        "fit",
        "zoom_in",
        "zoom_out",
        "actual_size",
        "fullscreen",
        "setup",
        "annotate",
        "pan",
        "results",
        "preview",
        "reviewed",
        "undo",
        "redo",
        "reset",
        "warning",
        "previous",
        "next",
        "down",
        "up",
        "close",
        "panel_left",
        "panel_right",
        "more",
        "help",
    )
}


def icon(name, color=None, **_kwargs):
    variant = (
        f"{name}-dark"
        if color == "#09251e" and name in {"folder", "export", "reviewed", "sam3"}
        else name
    )
    path = RESOURCE_DIR / f"{variant}.svg"
    return QIcon(str(path)) if name in ICONS else QIcon()
