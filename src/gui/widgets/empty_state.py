from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.icons import ICONS, icon


class EmptyStateWidget(QWidget):
    open_image_requested = Signal()
    open_folder_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("emptyState")
        self.setAttribute(Qt.WA_StyledBackground, True)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(12)
        layout.setContentsMargins(28, 20, 28, 20)

        glyph = QLabel()
        glyph.setPixmap(icon(ICONS["app"]).pixmap(52, 52))
        glyph.setAlignment(Qt.AlignCenter)
        layout.addWidget(glyph)

        title = QLabel("Ready to annotate.")
        title.setObjectName("emptyTitle")
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)
        layout.addWidget(title)

        subtitle = QLabel(
            "Create precise training data with SAM3 assistance.\nOpen images, correct objects, and export reviewed labels."
        )
        subtitle.setObjectName("emptySubtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setMaximumWidth(440)
        layout.addWidget(subtitle)

        buttons = QHBoxLayout()
        buttons.setAlignment(Qt.AlignCenter)
        buttons.setSpacing(8)
        open_folder = QPushButton("Open Folder")
        open_folder.setObjectName("emptyPrimary")
        open_folder.setIcon(icon(ICONS["folder"], "#09251e"))
        open_folder.clicked.connect(self.open_folder_requested)
        open_image = QPushButton("Open Image")
        open_image.setObjectName("emptySecondary")
        open_image.setIcon(icon(ICONS["image"], "#cbd5e1", scale_factor=0.8))
        open_image.clicked.connect(self.open_image_requested)
        buttons.addWidget(open_folder)
        buttons.addWidget(open_image)
        layout.addLayout(buttons)
        hint = QLabel(
            "Resume saved work from Open → Open Project\nKeyboard shortcuts: F1"
        )
        hint.setObjectName("mutedLabel")
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        layout.addWidget(hint)
