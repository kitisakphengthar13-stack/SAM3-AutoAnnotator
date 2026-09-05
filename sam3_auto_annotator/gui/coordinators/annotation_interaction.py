from __future__ import annotations

from PySide6.QtCore import QTimer


class AnnotationInteractionCoordinator:
    """Coordinate cross-view annotation interactions that are not window behavior."""

    def __init__(self, window):
        self.window = window
        self._draw_class_restore = None
        window.canvas.box_drawn.connect(self._prepare_active_class_for_draw)
        window.actions.mark_reviewed.triggered.connect(
            lambda _checked=False: QTimer.singleShot(0, self._advance_after_review)
        )

    def _prepare_active_class_for_draw(self, _box):
        """Bridge legacy controller lookup until active class is controller-owned."""
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

    def _advance_after_review(self):
        controller = self.window.controller
        if controller is None:
            return
        image = controller.current_image
        if image is None or getattr(image.status, "value", None) != "reviewed":
            return
        if self.window.actions.next_image.isEnabled():
            self.window.dataset.select_relative(1)
