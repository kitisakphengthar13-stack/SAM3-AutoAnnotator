from PySide6.QtGui import QIcon


try:
    import qtawesome as qta
except ImportError:  # pragma: no cover - optional GUI polish dependency
    qta = None


def icon(name, color="#334155", scale_factor=1.0):
    if qta is None:
        return QIcon()
    try:
        return qta.icon(name, color=color, scale_factor=scale_factor)
    except Exception:
        return QIcon()


ICONS = {
    "app": "fa5s.tags",
    "image": "fa5s.image",
    "folder": "fa5s.folder-open",
    "state": "fa5s.file-import",
    "save": "fa5s.save",
    "sam3": "fa5s.robot",
    "draw": "fa5s.vector-square",
    "trash": "fa5s.trash",
    "export": "fa5s.file-export",
    "fit": "fa5s.expand-arrows-alt",
    "setup": "fa5s.cog",
    "annotate": "fa5s.mouse-pointer",
    "results": "fa5s.file-alt",
    "preview": "fa5s.eye",
    "reviewed": "fa5s.check-circle",
    "warning": "fa5s.exclamation-triangle",
}
