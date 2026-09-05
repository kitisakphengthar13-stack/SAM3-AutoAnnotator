from __future__ import annotations

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtCore import QObject

from sam3_auto_annotator.gui.icons import ICONS, icon


class AppActions(QObject):
    """Single source of truth for commands, shortcuts and enabled state."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.open_image = self._action(
            "Open Image", "image", QKeySequence.StandardKey.Open,
            "Start a project from one image.",
        )
        self.open_folder = self._action(
            "Open Folder", "folder", "Ctrl+Shift+O",
            "Start a project from all supported images in one folder.",
        )
        self.open_state = self._action(
            "Open Project", "state", "Ctrl+Alt+O",
            "Resume a saved annotation_state.json project.",
        )
        self.import_yolo = self._action(
            "Import YOLO", "state", "Ctrl+I",
            "Import existing YOLO detection labels into the current project.",
        )
        self.save = self._action(
            "Save Project", "save", QKeySequence.StandardKey.Save,
            "Save the editable project state.",
        )
        self.run_current = self._action(
            "Run SAM3", "sam3", "F5",
            "Run SAM3 on the selected image.", role="primary",
        )
        self.run_remaining = self._action(
            "Run Pending", "sam3", "Shift+F5",
            "Run SAM3 on every not-predicted or failed image.",
        )
        self.draw_box = self._action(
            "Draw Box", "draw", "B",
            "Toggle manual box drawing on the image.", checkable=True,
        )
        self.delete_annotation = self._action(
            "Delete", "trash", QKeySequence.StandardKey.Delete,
            "Delete the selected annotation.", role="danger",
        )
        self.export = self._action(
            "Export Labels", "export", "Ctrl+E",
            "Save corrected CSV and YOLO labels.", role="export",
        )
        self.fit = self._action(
            "Fit", "fit", "F",
            "Fit the current image inside the canvas.",
        )
        self.previous_image = self._action(
            "Previous Image", "previous", "Alt+Left",
            "Select the previous visible image.",
        )
        self.next_image = self._action(
            "Next Image", "next", "Alt+Right",
            "Select the next visible image.",
        )
        self.apply_class = self._action(
            "Apply Class", "reviewed", None,
            "Save the selected class.",
        )
        self.apply_box = self._action(
            "Apply Box", "draw", "Ctrl+Return",
            "Save the edited bounding box.",
        )
        self.resegment = self._action(
            "Re-segment", "sam3", "Ctrl+R",
            "Generate a new mask/polygon from the selected bounding box.",
        )
        self.reset_sam3 = self._action(
            "Reset to SAM3", "reset", None,
            "Restore the original SAM3 annotation.",
        )
        self.mark_reviewed = self._action(
            "Mark Image Reviewed", "reviewed", "R",
            "Mark the selected image as reviewed.",
        )
        self.save_preview = self._action(
            "Save Preview", "preview", None,
            "Save a preview image using the visible overlays.",
        )
        self.open_preview = self._action(
            "Open Preview", "preview", None,
            "Open the most recently saved preview image.",
        )
        self.open_output = self._action(
            "Open Output Folder", "folder", None,
            "Open the current project output folder.",
        )
        self.cancel_batch = self._action(
            "Cancel Batch", "warning", None,
            "Stop after the image currently being processed.", role="danger",
        )

    def _action(
        self,
        text,
        icon_key,
        shortcut,
        tooltip,
        *,
        role=None,
        checkable=False,
    ):
        color = {
            "primary": "#2563eb",
            "danger": "#dc2626",
            "export": "#2563eb",
        }.get(role, "#334155")
        action = QAction(icon(ICONS[icon_key], color, scale_factor=0.82), text, self)
        action.setCheckable(checkable)
        action.setToolTip(tooltip)
        action.setStatusTip(tooltip)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if role:
            action.setProperty("role", role)
        return action

    @property
    def project_actions(self):
        return (
            self.import_yolo,
            self.save,
            self.run_current,
            self.run_remaining,
            self.draw_box,
            self.delete_annotation,
            self.export,
            self.fit,
            self.previous_image,
            self.next_image,
            self.apply_class,
            self.apply_box,
            self.resegment,
            self.reset_sam3,
            self.mark_reviewed,
            self.save_preview,
            self.open_preview,
            self.open_output,
        )
