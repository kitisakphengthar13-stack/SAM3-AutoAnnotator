from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSizePolicy


class ElidedLabel(QLabel):
    """Single-line label that keeps its full value in the API and tooltip."""

    def __init__(self, text="", *, mode=Qt.ElideRight, parent=None):
        super().__init__(parent)
        self._full_text = ""
        self._elide_mode = mode
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setText(text)

    def text(self):
        return self._full_text

    def setText(self, text):
        self._full_text = str(text)
        self.setToolTip(self._full_text)
        self._render_text()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_text()

    def _render_text(self):
        available = max(0, self.contentsRect().width())
        shown = self.fontMetrics().elidedText(
            self._full_text,
            self._elide_mode,
            available,
        )
        super().setText(shown)
