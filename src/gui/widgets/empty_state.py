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
        layout.setSpacing(10)

        glyph = QLabel()
        glyph.setPixmap(icon(ICONS["image"], "#93c5fd").pixmap(32, 32))
        glyph.setAlignment(Qt.AlignCenter)
        layout.addWidget(glyph)

        title = QLabel("Start an annotation project")
        title.setObjectName("emptyTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel(
            "Open a folder for a review workflow, or open one image for a quick annotation."
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
        open_folder.setIcon(icon(ICONS["folder"], "#dbeafe", scale_factor=0.8))
        open_folder.clicked.connect(self.open_folder_requested)
        open_image = QPushButton("Open Image")
        open_image.setObjectName("emptySecondary")
        open_image.setIcon(icon(ICONS["image"], "#cbd5e1", scale_factor=0.8))
        open_image.clicked.connect(self.open_image_requested)
        buttons.addWidget(open_folder)
        buttons.addWidget(open_image)
        layout.addLayout(buttons)
