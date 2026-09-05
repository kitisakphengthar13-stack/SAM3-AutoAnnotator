from __future__ import annotations

from PySide6.QtCore import QTimer


class AnnotationInteractionCoordinator:
    """Coordinate cross-view annotation interactions that are not window behavior."""

    def __init__(self, window):
        self.window = window
        window.actions.mark_reviewed.triggered.connect(
            lambda _checked=False: QTimer.singleShot(0, self._advance_after_review)
        )

    def _advance_after_review(self):
        controller = self.window.controller
        if controller is None:
            return
        image = controller.current_image
        if image is None or getattr(image.status, "value", None) != "reviewed":
            return
        if self.window.actions.next_image.isEnabled():
            self.window.dataset.select_relative(1)
