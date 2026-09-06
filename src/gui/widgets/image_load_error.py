from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from gui.icons import ICONS, icon


class ImageLoadErrorWidget(QWidget):
    """Recoverable inline state shown instead of a stale canvas image."""

    retry_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("imageLoadError")
        self.setAttribute(Qt.WA_StyledBackground, True)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(10)

        glyph = QLabel()
        glyph.setPixmap(icon(ICONS["warning"]).pixmap(32, 32))
        glyph.setAlignment(Qt.AlignCenter)
        layout.addWidget(glyph)

        title = QLabel("Image could not be displayed")
        title.setObjectName("errorStateTitle")
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)
        layout.addWidget(title)

        self.detail_label = QLabel()
        self.detail_label.setObjectName("errorStateDetail")
        self.detail_label.setAlignment(Qt.AlignCenter)
        self.detail_label.setWordWrap(True)
        self.detail_label.setMaximumWidth(460)
        layout.addWidget(self.detail_label)

        retry = QPushButton(icon(ICONS["reset"]), "Retry")
        retry.setObjectName("emptyPrimary")
        retry.clicked.connect(self.retry_requested)
        layout.addWidget(retry, alignment=Qt.AlignCenter)

    def set_error(self, image_name, message):
        detail = (
            f"{image_name}\n{message}\n"
            "Retry after repairing the file, or select another image in Dataset."
        )
        self.detail_label.setText(detail)
        self.detail_label.setToolTip(str(message))
