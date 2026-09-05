"""Semantic application icons resolved through Qt's native icon APIs."""

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QStyle


@dataclass(frozen=True)
class IconSpec:
    theme: QIcon.ThemeIcon | None = None
    fallback: QStyle.StandardPixmap | None = None
    resource: str | None = None


RESOURCE_DIR = Path(__file__).resolve().parent / "resources"


ICON_SPECS = {
    "app": IconSpec(QIcon.ThemeIcon.InsertImage, QStyle.SP_ComputerIcon),
    "image": IconSpec(QIcon.ThemeIcon.InsertImage, QStyle.SP_FileIcon),
    "folder": IconSpec(QIcon.ThemeIcon.FolderOpen, QStyle.SP_DirOpenIcon),
    "state": IconSpec(QIcon.ThemeIcon.DocumentOpenRecent, QStyle.SP_DialogOpenButton),
    "save": IconSpec(QIcon.ThemeIcon.DocumentSave, QStyle.SP_DialogSaveButton),
    "sam3": IconSpec(QIcon.ThemeIcon.MediaPlaybackStart, QStyle.SP_MediaPlay),
    # Qt has no cross-platform standard pixmap for these domain concepts.
    "draw": IconSpec(resource="draw-box.svg"),
    "trash": IconSpec(QIcon.ThemeIcon.EditDelete, QStyle.SP_TrashIcon),
    "export": IconSpec(QIcon.ThemeIcon.DocumentSaveAs, QStyle.SP_DialogSaveButton),
    "fit": IconSpec(QIcon.ThemeIcon.ZoomFitBest, QStyle.SP_TitleBarMaxButton),
    "setup": IconSpec(resource="setup.svg"),
    "annotate": IconSpec(resource="review-box.svg"),
    "results": IconSpec(resource="results.svg"),
    "preview": IconSpec(QIcon.ThemeIcon.DocumentPrintPreview, QStyle.SP_FileDialogContentsView),
    "reviewed": IconSpec(QIcon.ThemeIcon.MailMarkRead, QStyle.SP_DialogApplyButton),
    "reset": IconSpec(QIcon.ThemeIcon.EditUndo, QStyle.SP_ArrowBack),
    "warning": IconSpec(QIcon.ThemeIcon.DialogWarning, QStyle.SP_MessageBoxWarning),
    "previous": IconSpec(QIcon.ThemeIcon.GoPrevious, QStyle.SP_ArrowBack),
    "next": IconSpec(QIcon.ThemeIcon.GoNext, QStyle.SP_ArrowForward),
}

# Views use semantic names only and remain independent of any icon provider.
ICONS = {name: name for name in ICON_SPECS}


def icon(name, color=None, scale_factor=None):
    """Return a platform icon with a guaranteed QStyle fallback."""
    del color, scale_factor
    spec = ICON_SPECS.get(name)
    if spec is None:
        return QIcon()

    app = QApplication.instance()
    fallback = QIcon()
    if spec.resource:
        fallback = QIcon(str(RESOURCE_DIR / spec.resource))
    if fallback.isNull() and app is not None and spec.fallback is not None:
        fallback = app.style().standardIcon(spec.fallback)
    if spec.theme is None:
        return fallback
    return QIcon.fromTheme(spec.theme, fallback)
