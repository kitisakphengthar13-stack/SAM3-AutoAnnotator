from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QSizePolicy, QToolBar

from sam3_auto_annotator.gui.widgets.elided_label import ElidedLabel


class CommandBar(QToolBar):
    def __init__(self, actions, parent=None):
        super().__init__("Main Commands", parent)
        self.setObjectName("commandBar")
        self.setMovable(False)
        self.setFloatable(False)
        self.setIconSize(QSize(16, 16))
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        self.project_label = ElidedLabel("No project loaded")
        self.project_label.setObjectName("projectSubtitle")
        self.project_label.setMinimumWidth(120)
        self.project_label.setMaximumWidth(220)
        self.project_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.addWidget(self.project_label)
        self.addSeparator()

        for action in (
            actions.open_image,
            actions.open_folder,
            actions.open_state,
            actions.save,
        ):
            self.addAction(action)
        self.addSeparator()
        self.addAction(actions.project_settings)
        self.addAction(actions.export)

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
        actions.export,
    ):
        file_menu.addAction(action)
    file_menu.addSeparator()
    exit_action = QAction("Exit", window)
    exit_action.setShortcut(QKeySequence.StandardKey.Quit)
    exit_action.triggered.connect(window.close)
    file_menu.addAction(exit_action)

    annotation_menu = menu_bar.addMenu("&Annotation")
    for action in (
        actions.run_current,
        actions.run_remaining,
        actions.draw_box,
        actions.apply_box,
        actions.apply_class,
        actions.resegment,
        actions.reset_sam3,
        actions.delete_annotation,
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
    view_menu.addAction(actions.focus_workspace)
    view_menu.addAction(actions.fullscreen)

    return exit_action
