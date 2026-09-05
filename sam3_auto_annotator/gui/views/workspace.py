from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from sam3_auto_annotator.gui.widgets.action_button import action_button
from sam3_auto_annotator.gui.widgets.elided_label import ElidedLabel
from sam3_auto_annotator.gui.widgets.empty_state import EmptyStateWidget
from sam3_auto_annotator.gui.widgets.image_canvas import ImageCanvas
from sam3_auto_annotator.gui.widgets.image_load_error import ImageLoadErrorWidget
from sam3_auto_annotator.gui.widgets.task_progress import TaskProgressWidget


class CanvasWorkspace(QWidget):
    """Primary work surface. Project/configuration UI must not steal canvas space."""

    def __init__(self, actions, parent=None):
        super().__init__(parent)
        self.setObjectName("canvasWorkspace")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QWidget()
        toolbar.setObjectName("canvasBar")
        toolbar_layout = QVBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 7, 10, 7)
        toolbar_layout.setSpacing(6)

        command_row = QHBoxLayout()
        command_row.setSpacing(5)
        self.canvas_hint = ElidedLabel(
            "Open an image or folder to start reviewing annotations."
        )
        self.canvas_hint.setObjectName("canvasHint")
        self.canvas_hint.setTextInteractionFlags(Qt.TextSelectableByMouse)
        command_row.addWidget(self.canvas_hint, 1)

        command_row.addWidget(QLabel("Class"))
        self.active_class_combo = QComboBox()
        self.active_class_combo.setObjectName("activeClassCombo")
        self.active_class_combo.setMinimumContentsLength(9)
        self.active_class_combo.setToolTip("Class assigned to the next box you draw.")
        command_row.addWidget(self.active_class_combo)
        command_row.addWidget(action_button(actions.draw_box))
        command_row.addSpacing(8)
        command_row.addWidget(action_button(actions.zoom_out, icon_only=True))
        command_row.addWidget(action_button(actions.actual_size))
        command_row.addWidget(action_button(actions.zoom_in, icon_only=True))
        command_row.addWidget(action_button(actions.fit))
        toolbar_layout.addLayout(command_row)

        overlay_row = QHBoxLayout()
        overlay_row.setSpacing(8)
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
        overlay_row.addWidget(action_button(actions.focus_workspace))
        toolbar_layout.addLayout(overlay_row)
        layout.addWidget(toolbar)

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

    def set_classes(self, prompts):
        current = self.active_class_combo.currentText()
        self.active_class_combo.blockSignals(True)
        try:
            self.active_class_combo.clear()
            self.active_class_combo.addItems(list(prompts))
            index = self.active_class_combo.findText(current)
            if index >= 0:
                self.active_class_combo.setCurrentIndex(index)
        finally:
            self.active_class_combo.blockSignals(False)
        self.active_class_combo.setToolTip(
            "Class assigned to the next box you draw: "
            + (self.active_class_combo.currentText() or "none")
        )


def _overlay_checkbox(text, checked):
    checkbox = QCheckBox(text)
    checkbox.setChecked(checked)
    checkbox.setToolTip(f"Show or hide {text.lower()} on the image and saved preview.")
    return checkbox
