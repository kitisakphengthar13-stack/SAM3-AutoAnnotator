from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QTimer
from PySide6.QtGui import QUndoStack

from gui.undo import AnnotationSnapshotCommand


@dataclass(frozen=True)
class _EditCapture:
    project: object
    image: object
    before: dict
    text: str
    selected_annotation_id: str | None


class AnnotationHistoryCoordinator:
    """Own undo/redo transactions and saved-state tracking for annotation edits."""

    def __init__(self, window):
        self.window = window
        self.stack = QUndoStack(window)
        self._project = None
        self._external_dirty = False
        self._connect()

    def _connect(self):
        actions = self.window.actions
        actions.undo.triggered.connect(self.stack.undo)
        actions.redo.triggered.connect(self.stack.redo)
        self.stack.canUndoChanged.connect(actions.undo.setEnabled)
        self.stack.canRedoChanged.connect(actions.redo.setEnabled)
        self.stack.cleanChanged.connect(self._clean_changed)

    def sync_project(self):
        controller = self.window.controller
        project = controller.project if controller is not None else None
        if project is self._project:
            return
        self._project = project
        self._external_dirty = False
        self.stack.clear()
        self.stack.setClean()

    def mark_clean(self):
        self._external_dirty = False
        self.stack.setClean()
        self._sync_dirty_state()

    def mark_external_dirty(self):
        """Mark a non-undoable mutation and invalidate incompatible edit commands."""
        self.stack.clear()
        self._external_dirty = True
        self._sync_dirty_state()

    def capture_edit(self, text):
        controller = self.window.controller
        image = controller.current_image if controller is not None else None
        if image is None:
            return None
        self.sync_project()
        return _EditCapture(
            project=controller.project,
            image=image,
            before=image.to_dict(),
            text=str(text),
            selected_annotation_id=controller.selected_annotation_id,
        )

    def commit_edit(self, capture, selected_annotation_id=None):
        if capture is None:
            return False
        controller = self.window.controller
        if controller is None or controller.project is not capture.project:
            self.sync_project()
            return False
        after = capture.image.to_dict()
        if capture.before == after:
            return False
        self.stack.push(
            AnnotationSnapshotCommand(
                capture.image,
                capture.before,
                after,
                self._apply_snapshot,
                text=capture.text,
                before_selected_annotation_id=capture.selected_annotation_id,
                after_selected_annotation_id=selected_annotation_id,
            )
        )
        return True

    def clear_for_inference_boundary(self):
        """Drop incompatible edit commands when a model task starts."""
        history_was_dirty = self._external_dirty or not self.stack.isClean()
        self.stack.clear()
        if history_was_dirty:
            self._external_dirty = True
        self._sync_dirty_state()

    def _apply_snapshot(self, image_index, selected_annotation_id):
        controller = self.window.controller
        if controller is None or controller.project is not self._project:
            return
        self.window.dataset.refresh(image_index)
        if controller.current_image_index == image_index:
            controller.annotations.render_current_annotations(selected_annotation_id)
        else:
            controller.presentation.update_actions()
            controller.presentation.update_context()
        # QUndoStack updates its index after the command callback returns.
        QTimer.singleShot(0, self._sync_dirty_state)

    def _clean_changed(self, _clean):
        QTimer.singleShot(0, self._sync_dirty_state)

    def _sync_dirty_state(self):
        controller = self.window.controller
        if controller is None or controller.project is not self._project:
            return
        controller.dirty = self._external_dirty or not self.stack.isClean()
        controller.presentation.update_actions()
        controller.presentation.update_context()
