"""GUI workflow coordinators that keep QMainWindow composition-focused."""

from sam3_auto_annotator.gui.coordinators.annotation_history import AnnotationHistoryCoordinator
from sam3_auto_annotator.gui.coordinators.annotation_interaction import AnnotationInteractionCoordinator
from sam3_auto_annotator.gui.coordinators.export_dialog import ExportDialogCoordinator
from sam3_auto_annotator.gui.coordinators.setup_dialog import SetupDialogCoordinator

__all__ = [
    "AnnotationHistoryCoordinator",
    "AnnotationInteractionCoordinator",
    "ExportDialogCoordinator",
    "SetupDialogCoordinator",
]
