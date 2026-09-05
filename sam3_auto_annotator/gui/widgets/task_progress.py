from __future__ import annotations

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QHBoxLayout, QProgressBar, QWidget

from sam3_auto_annotator.gui.widgets.action_button import action_button
from sam3_auto_annotator.gui.widgets.elided_label import ElidedLabel


class TaskProgressWidget(QWidget):
    def __init__(self, cancel_action: QAction, parent=None):
        super().__init__(parent)
        self.setObjectName("taskProgress")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(8)
        self.status_label = ElidedLabel("Idle")
        self.status_label.setObjectName("taskStatus")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.cancel_button = action_button(cancel_action)
        layout.addWidget(self.status_label, 2)
        layout.addWidget(self.progress, 3)
        layout.addWidget(self.cancel_button)
        self.setVisible(False)

    def show_running(self, message, maximum=0, *, cancellable=False):
        self.status_label.setText(message)
        self.progress.setRange(0, maximum if maximum > 0 else 0)
        self.progress.setValue(0)
        self.cancel_button.setVisible(bool(cancellable))
        self.setVisible(True)

    def show_result(self, message):
        self.status_label.setText(message)
        # Stop the indeterminate animation used by single-image operations.
        # A completed/error message must not continue to look busy forever.
        if self.progress.minimum() == 0 and self.progress.maximum() == 0:
            self.progress.setRange(0, 1)
            self.progress.setValue(1)
        self.cancel_button.setVisible(False)
        self.setVisible(True)

    def update_progress(self, current, total, message=None):
        self.progress.setRange(0, max(int(total), 1))
        self.progress.setValue(max(0, min(int(current), int(total))))
        if message is not None:
            self.status_label.setText(str(message))

    def hide_when_idle(self):
        self.cancel_button.setVisible(False)
        self.setVisible(False)
