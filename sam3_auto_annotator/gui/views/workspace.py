from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sam3_auto_annotator.gui.icons import ICONS, icon
from sam3_auto_annotator.gui.views.annotation_panel import AnnotationPanel
from sam3_auto_annotator.gui.views.dataset_panel import DatasetPanel
from sam3_auto_annotator.gui.views.results_panel import ResultsPanel
from sam3_auto_annotator.gui.views.setup_panel import SetupPanel
from sam3_auto_annotator.gui.widgets.action_button import action_button
from sam3_auto_annotator.gui.widgets.elided_label import ElidedLabel
from sam3_auto_annotator.gui.widgets.empty_state import EmptyStateWidget
from sam3_auto_annotator.gui.widgets.image_canvas import ImageCanvas
from sam3_auto_annotator.gui.widgets.image_load_error import ImageLoadErrorWidget
from sam3_auto_annotator.gui.widgets.task_progress import TaskProgressWidget


class CanvasWorkspace(QWidget):
    def __init__(self, actions, parent=None):
        super().__init__(parent)
        self.setObjectName("canvasWorkspace")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        canvas_bar = QWidget()
        canvas_bar.setObjectName("canvasBar")
        bar_layout = QVBoxLayout(canvas_bar)
        bar_layout.setContentsMargins(10, 6, 8, 6)
        bar_layout.setSpacing(4)

        context_row = QHBoxLayout()
        self.canvas_hint = ElidedLabel(
            "Open an image or folder to start reviewing annotations."
        )
        self.canvas_hint.setObjectName("canvasHint")
        self.canvas_hint.setTextInteractionFlags(Qt.TextSelectableByMouse)
        context_row.addWidget(self.canvas_hint, 1)
        context_row.addWidget(action_button(actions.draw_box, icon_only=True))
        context_row.addWidget(action_button(actions.fit, icon_only=True))
        bar_layout.addLayout(context_row)

        overlay_row = QHBoxLayout()
        overlay_row.addWidget(QLabel("Overlays"))
        self.show_boxes_check = _overlay_checkbox("Boxes", True)
        self.show_masks_check = _overlay_checkbox("Masks", True)
        self.show_polygons_check = _overlay_checkbox("Polygons", False)
        for checkbox in (
            self.show_boxes_check,
            self.show_masks_check,
            self.show_polygons_check,
        ):
            overlay_row.addWidget(checkbox)
        overlay_row.addStretch(1)
        bar_layout.addLayout(overlay_row)
        layout.addWidget(canvas_bar)

        self.workspace_stack = QStackedWidget()
        self.workspace_stack.setObjectName("canvasStack")
        self.empty_state = EmptyStateWidget()
        self.image_load_error = ImageLoadErrorWidget()
        self.canvas = ImageCanvas()
        self.workspace_stack.addWidget(self.empty_state)
        self.workspace_stack.addWidget(self.image_load_error)
        self.workspace_stack.addWidget(self.canvas)
        layout.addWidget(self.workspace_stack, 1)

        self.task_progress = TaskProgressWidget(actions.cancel_batch)
        layout.addWidget(self.task_progress)


class InspectorPanel(QTabWidget):
    def __init__(self, actions, parent=None):
        super().__init__(parent)
        self.setObjectName("inspectorTabs")
        self.setDocumentMode(True)
        self.setMinimumWidth(260)
        self.setup = SetupPanel(actions)
        self.annotation = AnnotationPanel(actions)
        self.results = ResultsPanel(actions)
        self.addTab(self.setup, icon(ICONS["setup"], "#475569", 0.85), "Setup")
        self.addTab(self.annotation, icon(ICONS["annotate"], "#475569", 0.85), "Review")
        self.addTab(self.results, icon(ICONS["results"], "#475569", 0.85), "Export")


class AnnotationWorkspace(QSplitter):
    def __init__(self, actions, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.setObjectName("workspaceRoot")
        self.setChildrenCollapsible(False)
        self.setHandleWidth(5)
        self.dataset = DatasetPanel(actions)
        self.canvas_area = CanvasWorkspace(actions)
        self.inspector = InspectorPanel(actions)
        self.addWidget(self.dataset)
        self.addWidget(self.canvas_area)
        self.addWidget(self.inspector)
        self.setStretchFactor(0, 0)
        self.setStretchFactor(1, 1)
        self.setStretchFactor(2, 0)
        self.setSizes([220, 760, 320])


def _overlay_checkbox(text, checked):
    checkbox = QCheckBox(text)
    checkbox.setChecked(checked)
    checkbox.setToolTip(f"Show or hide {text.lower()} on the image and saved preview.")
    return checkbox
