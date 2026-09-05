from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


class PreviewLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__("Preview appears after export or Save Preview.", parent)
        self.setObjectName("previewThumb")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(150)
        self.setMinimumWidth(0)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._source = QPixmap()

    def set_image(self, path):
        image = QImage(str(Path(path)))
        if image.isNull():
            return False
        self._source = QPixmap.fromImage(image)
        if self._source.isNull():
            return False
        self._render_scaled()
        return True

    def clear_preview(self):
        self._source = QPixmap()
        self.clear()
        self.setText("Preview appears after export or Save Preview.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_scaled()

    def _render_scaled(self):
        if self._source.isNull():
            return
        target = self.contentsRect().size()
        target.setWidth(max(target.width() - 16, 1))
        target.setHeight(max(target.height() - 16, 1))
        self.setPixmap(
            self._source.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
