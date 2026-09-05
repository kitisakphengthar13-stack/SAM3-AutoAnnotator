from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QPlainTextEdit, QSizePolicy


class ClassPromptEditor(QPlainTextEdit):
    """Plain-text class editor with a bounded viewport and internal scrolling."""

    def __init__(self, parent=None, *, visible_rows=5):
        super().__init__(parent)
        self._visible_rows = max(2, int(visible_rows))
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def sizeHint(self):
        hint = super().sizeHint()
        line_height = self.fontMetrics().lineSpacing()
        document_padding = 12
        hint.setHeight(
            line_height * self._visible_rows
            + document_padding
            + 2 * self.frameWidth()
        )
        return QSize(hint)

    def minimumSizeHint(self):
        hint = self.sizeHint()
        hint.setWidth(80)
        return hint
