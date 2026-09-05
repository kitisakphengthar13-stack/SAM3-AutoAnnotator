from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QSizePolicy, QToolBar

from gui.widgets.elided_label import ElidedLabel


class CommandBar(QToolBar):
    """Compact global workflow bar; dense tool controls belong beside the canvas."""

    def __init__(self, actions, parent=None):
        super().__init__("Main Commands", parent)
        self.setObjectName("commandBar")
        self.setMovable(False)
        self.setFloatable(False)
        self.setIconSize(QSize(18, 18))
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        self.project_label = ElidedLabel("No project loaded")
        self.project_label.setObjectName("projectSubtitle")
        self.project_label.setMinimumWidth(100)
        self.project_label.setMaximumWidth(160)
        self.project_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.addWidget(self.project_label)
        self.addSeparator()

        self.addAction(actions.open_image)
        self.addAction(actions.save)
        self.addSeparator()
        self._add_compact(actions.previous_image)
        self._add_compact(actions.next_image)
        self.addAction(actions.mark_reviewed)
        self.addSeparator()
        self._add_compact(actions.undo)
        self._add_compact(actions.redo)
        self.addSeparator()
        self.addAction(actions.run_current)
        self.addAction(actions.project_settings)
        self.addAction(actions.export_dialog)

    def _add_compact(self, action):
        self.addAction(action)
        button = self.widgetForAction(action)
        if button is not None:
            button.setToolButtonStyle(Qt.ToolButtonIconOnly)

    def tool_button(self, action):
        return self.widgetForAction(action)


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
