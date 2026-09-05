"""GUI workflow coordinators that keep QMainWindow composition-focused."""

from gui.coordinators.annotation_history import AnnotationHistoryCoordinator
from gui.coordinators.export_dialog import ExportDialogCoordinator
from gui.coordinators.setup_dialog import SetupDialogCoordinator

__all__ = [
    "AnnotationHistoryCoordinator",
    "ExportDialogCoordinator",
    "SetupDialogCoordinator",
]
