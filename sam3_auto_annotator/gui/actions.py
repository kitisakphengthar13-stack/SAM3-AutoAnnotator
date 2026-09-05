from __future__ import annotations

from PySide6.QtCore import QObject
from PySide6.QtGui import QAction, QActionGroup, QKeySequence

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
        self.project_settings = self._action(
            "Setup", "setup", "Ctrl+,",
            "Configure model, classes, precision, and project output.",
        )
        self.import_yolo = self._action(
            "Import YOLO", "state", "Ctrl+I",
            "Import existing YOLO detection labels into the current project.",
        )
        self.save = self._action(
            "Save Project", "save", QKeySequence.StandardKey.Save,
            "Save the editable project state.",
        )
        self.undo = self._action(
            "Undo", "undo", QKeySequence.StandardKey.Undo,
            "Undo the last annotation edit.",
        )
        self.redo = self._action(
            "Redo", "redo", QKeySequence.StandardKey.Redo,
            "Redo the last undone annotation edit.",
        )
        self.undo.setEnabled(False)
        self.redo.setEnabled(False)
        self.run_current = self._action(
            "Run SAM3", "sam3", "F5",
            "Run SAM3 on the selected image.", role="primary",
        )
        self.run_remaining = self._action(
            "Run Pending", "sam3", "Shift+F5",
            "Run SAM3 on every not-predicted or failed image.",
        )

        self.select_tool = self._action(
            "Select", "annotate", "Escape",
            "Select, move, or resize annotations.", checkable=True,
        )
        self.pan_tool = self._action(
            "Pan", "previous", "P",
            "Pan the image without changing annotations.", checkable=True,
        )
        self.draw_box = self._action(
            "Box", "draw", "B",
            "Draw a bounding box using the active class.", checkable=True,
        )
        self.canvas_tool_group = QActionGroup(self)
        self.canvas_tool_group.setExclusive(True)
        for action in (self.select_tool, self.pan_tool, self.draw_box):
            self.canvas_tool_group.addAction(action)
        self.select_tool.setChecked(True)

        self.delete_annotation = self._action(
            "Delete", "trash", QKeySequence.StandardKey.Delete,
            "Delete the selected annotation. Undo restores it.", role="danger",
        )
        self.export_dialog = self._action(
            "Export…", "export", "Ctrl+E",
            "Review export readiness before writing corrected labels.", role="export",
        )
        self.export = self._action(
            "Export Now", "export", None,
            "Write corrected labels using the reviewed export plan.", role="export",
        )
        # Neither export surface is valid before a project is ready. Keeping the
        # write action disabled from construction also makes the preflight mirror
        # deterministic: the first controller enable transition emits changed().
        self.export.setEnabled(False)
        self.export_dialog.setEnabled(False)
        self.export.changed.connect(
            lambda: self.export_dialog.setEnabled(self.export.isEnabled())
        )
        self.fit = self._action(
            "Fit", "fit", "F",
            "Fit the current image inside the canvas.",
        )
        self.zoom_in = self._action(
            "Zoom In", "zoom_in", "Ctrl++",
            "Zoom into the image.",
        )
        self.zoom_out = self._action(
            "Zoom Out", "zoom_out", "Ctrl+-",
            "Zoom out of the image.",
        )
        self.actual_size = self._action(
            "100%", "actual_size", "Ctrl+0",
            "Show one image pixel per screen pixel.",
        )
        self.focus_workspace = self._action(
            "Focus Workspace", "fullscreen", "Ctrl+Shift+F",
            "Hide or restore side panels so the canvas gets maximum space.",
            checkable=True,
        )
        self.fullscreen = self._action(
            "Fullscreen", "fullscreen", "F11",
            "Toggle fullscreen application mode.", checkable=True,
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
            "Review & Next", "reviewed", "R",
            "Mark the selected image reviewed and advance when possible.",
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
        action = QAction(icon(ICONS[icon_key]), text, self)
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
            self.select_tool,
            self.pan_tool,
            self.draw_box,
            self.delete_annotation,
            self.export,
            self.fit,
            self.zoom_in,
            self.zoom_out,
            self.actual_size,
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
