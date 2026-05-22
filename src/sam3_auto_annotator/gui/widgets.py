from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from sam3_auto_annotator.gui.icons import ICONS, icon


class EmptyStateWidget(QWidget):
    open_image_requested = Signal()
    open_folder_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("emptyState")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(6)

        title = QLabel("No image loaded")
        title.setObjectName("emptyTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Open an image or folder to begin annotation.")
        subtitle.setObjectName("emptySubtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        buttons = QHBoxLayout()
        buttons.setAlignment(Qt.AlignCenter)
        buttons.setSpacing(6)
        open_image = QPushButton("Open Image")
        open_image.setObjectName("emptyPrimary")
        open_image.setIcon(icon(ICONS["image"], "#bfdbfe", scale_factor=0.75))
        open_image.clicked.connect(self.open_image_requested)
        open_folder = QPushButton("Open Folder")
        open_folder.setObjectName("emptySecondary")
        open_folder.setIcon(icon(ICONS["folder"], "#cbd5e1", scale_factor=0.75))
        open_folder.clicked.connect(self.open_folder_requested)
        buttons.addWidget(open_image)
        buttons.addWidget(open_folder)
        layout.addLayout(buttons)


class StatStrip(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.total = self._label("0 images")
        self.reviewed = self._label("0 reviewed")
        self.edited = self._label("0 edited")
        self.pending = self._label("0 pending")
        for label in (self.total, self.reviewed, self.edited, self.pending):
            layout.addWidget(label)
        layout.addStretch(1)

    def _label(self, text):
        label = QLabel(text)
        label.setStyleSheet("color:#64748b;font-size:8pt;")
        return label

    def update_counts(self, total, reviewed, edited, pending):
        self.total.setText(f"{total} images")
        self.reviewed.setText(f"{reviewed} reviewed")
        self.edited.setText(f"{edited} edited")
        self.pending.setText(f"{pending} pending")
