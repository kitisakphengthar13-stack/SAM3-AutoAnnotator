from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QWidget
from gui.icons import icon


class DockTitle(QWidget):
    """Native dock dragging with explicit, usable close and float hit targets."""

    def __init__(self, dock):
        super().__init__(dock)
        self.setObjectName("dockTitle")
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 4, 6, 4)
        row.setSpacing(2)
        title = QLabel(dock.windowTitle())
        title.setObjectName("sectionTitle")
        title.setAttribute(Qt.WA_TransparentForMouseEvents)
        row.addWidget(title)
        row.addStretch()
        self.float_button = self._button(
            "fullscreen", "Float or dock " + dock.windowTitle()
        )
        self.float_button.clicked.connect(
            lambda: dock.setFloating(not dock.isFloating())
        )
        row.addWidget(self.float_button)
        self.close_button = self._button(
            "close", "Hide " + dock.windowTitle() + " panel"
        )
        self.close_button.clicked.connect(dock.close)
        row.addWidget(self.close_button)

    @staticmethod
    def _button(name, tooltip):
        button = QToolButton()
        button.setIcon(icon(name))
        button.setObjectName("dockButton")
        button.setFixedSize(28, 28)
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        return button
