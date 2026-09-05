from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtGui import QUndoStack

from gui.undo import ImageSnapshotCommand


class AnnotationHistoryCoordinator:
    """Own undo/redo capture and saved-state tracking for annotation edits."""

    def __init__(self, window):
        self.window = window
        self.stack = QUndoStack(window)
        self._project = None
        self._pending_capture = None
        self._external_dirty = False
        self._connect()

    def _connect(self):
        actions = self.window.actions
        actions.undo.triggered.connect(self.stack.undo)
        actions.redo.triggered.connect(self.stack.redo)
        self.stack.canUndoChanged.connect(actions.undo.setEnabled)
        self.stack.canRedoChanged.connect(actions.redo.setEnabled)
        self.stack.cleanChanged.connect(self._clean_changed)

        for action, text in (
            (actions.apply_class, "Change class"),
            (actions.apply_box, "Edit box"),
            (actions.reset_sam3, "Reset annotation"),
            (actions.delete_annotation, "Delete annotation"),
        ):
            action.triggered.connect(
                lambda _checked=False, label=text: self.begin_edit(label)
            )

        self.window.canvas.box_drawn.connect(
            lambda _box: self.begin_edit("Add annotation")
        )
        self.window.canvas.annotation_changed.connect(
            lambda _annotation_id, _box: self.begin_edit("Edit box")
        )
        for action in (
            actions.run_current,
            actions.run_remaining,
            actions.resegment,
        ):
            action.triggered.connect(
                lambda _checked=False: QTimer.singleShot(
                    0, self.clear_if_inference_started
                )
            )

    def sync_project(self):
        controller = self.window.controller
        project = controller.project if controller is not None else None
        if project is self._project:
            return
        self._project = project
        self._pending_capture = None
        self._external_dirty = False
        self.stack.clear()
        self.stack.setClean()

    def mark_clean(self):
        self._external_dirty = False
        self.stack.setClean()
        self._sync_dirty_state()

    def mark_external_dirty(self):
        self._external_dirty = True
        self._sync_dirty_state()

    def begin_edit(self, text):
        controller = self.window.controller
        image = controller.current_image if controller is not None else None
        if image is None or self._pending_capture is not None:
            return
        self.sync_project()
        self._pending_capture = (
            image,
            image.to_dict(),
            str(text),
            controller.selected_annotation_id,
        )
        QTimer.singleShot(0, self._finish_edit)

    def _finish_edit(self):
        capture = self._pending_capture
        self._pending_capture = None
        controller = self.window.controller
        if capture is None or controller is None:
            return
        image, before, text, selected_id = capture
        if controller.project is not self._project:
            self.sync_project()
            return
        after = image.to_dict()
        if before == after:
            return
        self.stack.push(
            ImageSnapshotCommand(
                image,
                before,
                after,
                self._apply_snapshot,
                text=text,
                selected_annotation_id=selected_id,
            )
        )

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

    def clear_if_inference_started(self):
        controller = self.window.controller
        mode = getattr(getattr(controller, "mode", None), "value", "")
        if mode in {"predicting", "batch", "resegmenting"}:
            self.stack.clear()
            self._external_dirty = True
            self._sync_dirty_state()
