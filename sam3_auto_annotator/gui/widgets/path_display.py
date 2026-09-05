from __future__ import annotations

from PySide6.QtCore import Qt

from sam3_auto_annotator.gui.widgets.elided_label import ElidedLabel


class PathDisplay(ElidedLabel):
    """Compact read-only path with middle elision and the full value in a tooltip."""

    def __init__(self, text="-", parent=None):
        super().__init__(text, mode=Qt.ElideMiddle, parent=parent)
        self.setObjectName("pathDisplay")
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)

    def set_path(self, value):
        self.setText("-" if value in (None, "") else str(value))
