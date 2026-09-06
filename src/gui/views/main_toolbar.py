from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMenu, QSizePolicy, QToolBar, QWidget

from gui.widgets.action_button import action_button, menu_button
from gui.widgets.elided_label import ElidedLabel


class CommandBar(QToolBar):
    """Project commands only. Editing and review live with their work surface."""

    def __init__(self, actions, parent=None):
        super().__init__("Project commands", parent)
        self.setObjectName("commandBar")
        self.setMovable(False)
        self.setFloatable(False)
        self.setIconSize(QSize(18, 18))
        self._buttons = {}
        # One layout lets the project title surrender space before Qt's toolbar
        # layout hides a whole command in its native overflow menu.
        body = QWidget(self)
        body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._layout = QHBoxLayout(body)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        super().addWidget(body)
        brand = QLabel("SAM3")
        brand.setObjectName("brand")
        self._layout.addWidget(brand)
        self._add_separator()
        self.project_label = ElidedLabel("AutoAnnotator")
        self.project_label.setObjectName("projectSubtitle")
        self.project_label.setMinimumWidth(0)
        self.project_label.setMaximumWidth(240)
        self._layout.addWidget(self.project_label, 1)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._layout.addWidget(spacer)

        self.open_menu = QMenu(self)
        for action in (actions.open_folder, actions.open_image, actions.open_state):
            self.open_menu.addAction(action)
        self.open_button = menu_button("Open", "folder", self.open_menu)
        self.open_button.setToolTip("Open images, a folder, or a saved project")
        self._layout.addWidget(self.open_button)
        self.add_command(actions.save, icon_only=True)
        self._add_separator()
        self.run_current_button = self.add_command(actions.run_current)
        self.run_menu = QMenu(self)
        for action in (actions.run_current, actions.run_remaining):
            self.run_menu.addAction(action)
        self.run_menu.addSeparator()
        self.run_menu.addAction(actions.import_yolo)
        self.run_menu.addSeparator()
        self.run_menu.addAction(actions.project_settings)
        self.run_menu.setToolTipsVisible(True)
        self.run_button = menu_button("", "more", self.run_menu)
        self.run_button.setObjectName("assistMenu")
        self.run_button.setProperty("popup", False)
        self.run_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.run_button.setFixedSize(36, 36)
        self.run_button.setAccessibleName("More assistance commands")
        self.run_button.setToolTip(
            "Run SAM3 on this image or pending images, or import YOLO labels"
        )
        self._layout.addWidget(self.run_button)
        self.add_command(actions.project_settings)
        self.add_command(actions.export_dialog)

    def _add_separator(self):
        separator = QFrame()
        separator.setObjectName("commandSeparator")
        separator.setFixedSize(1, 24)
        self._layout.addSpacing(6)
        self._layout.addWidget(separator)
        self._layout.addSpacing(6)

    def add_command(self, action, icon_only=False):
        button = action_button(action, icon_only=icon_only)
        self._buttons[action] = button
        self._layout.addWidget(button)
        return button

    def tool_button(self, action):
        return self._buttons.get(action)


def build_menus(window, actions):
    menu_bar = window.menuBar()
    file_menu = menu_bar.addMenu("&File")
    for action in (
        actions.open_image,
        actions.open_folder,
        actions.open_state,
        actions.project_settings,
        actions.import_yolo,
        actions.save,
        actions.export_dialog,
    ):
        file_menu.addAction(action)
    file_menu.addSeparator()
    exit_action = QAction("Exit", window)
    exit_action.setShortcut(QKeySequence.StandardKey.Quit)
    exit_action.triggered.connect(window.close)
    file_menu.addAction(exit_action)

    edit_menu = menu_bar.addMenu("&Edit")
    edit_menu.addAction(actions.undo)
    edit_menu.addAction(actions.redo)
    edit_menu.addSeparator()
    edit_menu.addAction(actions.delete_annotation)

    annotation_menu = menu_bar.addMenu("&Annotation")
    for action in (
        actions.run_current,
        actions.run_remaining,
        actions.select_tool,
        actions.pan_tool,
        actions.draw_box,
        actions.apply_box,
        actions.apply_class,
        actions.resegment,
        actions.reset_sam3,
        actions.mark_reviewed,
    ):
        annotation_menu.addAction(action)

    navigate_menu = menu_bar.addMenu("&Navigate")
    for action in (
        actions.previous_image,
        actions.next_image,
        actions.zoom_out,
        actions.actual_size,
        actions.zoom_in,
        actions.fit,
    ):
        navigate_menu.addAction(action)

    view_menu = menu_bar.addMenu("&View")
    window.view_menu = view_menu
    view_menu.addAction(actions.focus_workspace)
    view_menu.addAction(actions.fullscreen)

    return exit_action
