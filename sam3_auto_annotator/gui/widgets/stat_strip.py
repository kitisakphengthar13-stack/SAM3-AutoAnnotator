from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QLabel, QWidget


class StatStrip(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statStrip")
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        self.total = self._label("0 images")
        self.reviewed = self._label("0 reviewed")
        self.edited = self._label("0 edited")
        self.pending = self._label("0 pending")
        layout.addWidget(self.total, 0, 0)
        layout.addWidget(self.reviewed, 0, 1)
        layout.addWidget(self.edited, 1, 0)
        layout.addWidget(self.pending, 1, 1)

    @staticmethod
    def _label(text):
        label = QLabel(text)
        label.setObjectName("statLabel")
        return label

    def update_counts(self, total, reviewed, edited, pending):
        self.total.setText(f"{total} images")
        self.reviewed.setText(f"{reviewed} reviewed")
        self.edited.setText(f"{edited} edited")
        self.pending.setText(f"{pending} pending")
