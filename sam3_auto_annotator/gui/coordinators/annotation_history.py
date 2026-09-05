from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtGui import QUndoStack

from sam3_auto_annotator.gui.undo import ImageSnapshotCommand


class AnnotationHistoryCoordinator:
    """Own undo/redo capture around user-driven annotation mutations."""

    def __init__(self, window):
        self.window = window
        self.stack = QUndoStack(window)
        self._project = None
        self._pending_capture = None
        self._draw_class_restore = None
        self._connect()

    def _connect(self):
        actions = self.window.actions
        actions.undo.triggered.connect(self.stack.undo)
        actions.redo.triggered.connect(self.stack.redo)
        self.stack.canUndoChanged.connect(actions.undo.setEnabled)
        self.stack.canRedoChanged.connect(actions.redo.setEnabled)

        for action, text in (
            (actions.apply_class, "Change class"),
            (actions.apply_box, "Edit box"),
            (actions.reset_sam3, "Reset annotation"),
            (actions.delete_annotation, "Delete annotation"),
        ):
            action.triggered.connect(
                lambda _checked=False, label=text: self.begin_edit(label)
            )

        self.window.canvas.box_drawn.connect(self._prepare_active_class_for_draw)
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
        self.stack.clear()

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
        controller.dirty = True
        self.window.dataset.refresh(image_index)
        if controller.current_image_index == image_index:
            controller._render_current_annotations(selected_annotation_id)
        else:
            controller._update_actions()
            controller._update_context()

    def clear_if_inference_started(self):
        controller = self.window.controller
        mode = getattr(getattr(controller, "mode", None), "value", "")
        if mode in {"predicting", "batch", "resegmenting"}:
            self.stack.clear()

    def _prepare_active_class_for_draw(self, _box):
        """Bridge legacy controller class lookup until it reads active class directly."""
        controller = self.window.controller
        image = controller.current_image if controller is not None else None
        if image is None:
            return
        previous_index = self.window.annotation.class_combo.currentIndex()
        previous_count = len(image.annotations)
        self.window.annotation.class_combo.setCurrentIndex(
            self.window.canvas_area.active_class_combo.currentIndex()
        )
        self._draw_class_restore = (image, previous_count, previous_index)
        QTimer.singleShot(0, self._restore_draw_class_if_failed)

    def _restore_draw_class_if_failed(self):
        restore = self._draw_class_restore
        self._draw_class_restore = None
        controller = self.window.controller
        if restore is None or controller is None:
            return
        image, previous_count, previous_index = restore
        if controller.current_image is image and len(image.annotations) == previous_count:
            self.window.annotation.class_combo.setCurrentIndex(previous_index)
