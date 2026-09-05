from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGraphicsItem,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.action_button import action_button
from gui.widgets.elided_label import ElidedLabel
from gui.widgets.empty_state import EmptyStateWidget
from gui.widgets.image_canvas import ImageCanvas
from gui.widgets.image_load_error import ImageLoadErrorWidget
from gui.widgets.task_progress import TaskProgressWidget


class WorkCanvas(ImageCanvas):
    """Annotation canvas with explicit workstation tools and navigation."""

    def __init__(self, actions, parent=None):
        super().__init__(parent)
        self._actions = actions

    def zoom_in(self):
        self._zoom_by(1.2)

    def zoom_out(self):
        self._zoom_by(1 / 1.2)

    def actual_size(self):
        if self._pixmap_item is None:
            return
        self._auto_fit = False
        self.resetTransform()

    def _zoom_by(self, factor):
        if self._pixmap_item is None:
            return
        current = self.transform().m11()
        target = current * factor
        if not 0.05 <= target <= 20.0:
            return
        self._auto_fit = False
        self.scale(factor, factor)

    def set_select_mode(self, enabled):
        if not enabled:
            return
        super().set_draw_mode(False)
        self.setDragMode(QGraphicsView.NoDrag)
        self.viewport().setCursor(Qt.ArrowCursor)

    def set_pan_mode(self, enabled):
        if enabled:
            super().set_draw_mode(False)
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.OpenHandCursor)
        elif not self._draw_mode:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.ArrowCursor)

    def set_draw_mode(self, enabled):
        super().set_draw_mode(enabled)
        if enabled:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.CrossCursor)

    def set_annotations(self, annotations):
        super().set_annotations(annotations)
        for annotation in annotations:
            if annotation.deleted:
                continue
            box_item = self._items_by_id.get(annotation.id)
            if box_item is None:
                continue
            confidence = (
                "" if annotation.confidence is None else f"  {annotation.confidence:.2f}"
            )
            label = QGraphicsSimpleTextItem(
                f"{annotation.class_name}{confidence}", box_item
            )
            label.setBrush(QBrush(QColor("#ffffff")))
            label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            label.setAcceptedMouseButtons(Qt.NoButton)
            label.setPos(2, -19)
            label.setZValue(30)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and not event.isAutoRepeat():
            self._actions.select_tool.trigger()
            event.accept()
            return
        if (
            event.key() == Qt.Key_Space
            and not event.isAutoRepeat()
            and not self._actions.draw_box.isChecked()
            and not self._actions.pan_tool.isChecked()
        ):
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            if self._actions.pan_tool.isChecked():
                self.setDragMode(QGraphicsView.ScrollHandDrag)
                self.viewport().setCursor(Qt.OpenHandCursor)
            else:
                self.setDragMode(QGraphicsView.NoDrag)
                self.viewport().setCursor(
                    Qt.CrossCursor if self._actions.draw_box.isChecked() else Qt.ArrowCursor
                )
            event.accept()
            return
        super().keyReleaseEvent(event)


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
        command_row.addWidget(action_button(actions.select_tool))
        command_row.addWidget(action_button(actions.pan_tool))
        command_row.addWidget(action_button(actions.draw_box))
        command_row.addSpacing(8)

        command_row.addWidget(QLabel("Class"))
        self.active_class_combo = QComboBox()
        self.active_class_combo.setObjectName("activeClassCombo")
        self.active_class_combo.setMinimumContentsLength(9)
        self.active_class_combo.setToolTip("Class assigned to the next box you draw.")
        command_row.addWidget(self.active_class_combo)
        command_row.addStretch(1)

        command_row.addWidget(action_button(actions.zoom_out, icon_only=True))
        command_row.addWidget(action_button(actions.actual_size))
        command_row.addWidget(action_button(actions.zoom_in, icon_only=True))
        command_row.addWidget(action_button(actions.fit))
        toolbar_layout.addLayout(command_row)

        info_row = QHBoxLayout()
        info_row.setSpacing(8)
        self.canvas_hint = ElidedLabel(
            "Open an image or folder to start reviewing annotations."
        )
        self.canvas_hint.setObjectName("canvasHint")
        self.canvas_hint.setTextInteractionFlags(Qt.TextSelectableByMouse)
        info_row.addWidget(self.canvas_hint, 1)
        info_row.addWidget(QLabel("Overlays"))
        self.show_boxes_check = _overlay_checkbox("Boxes", True)
        self.show_masks_check = _overlay_checkbox("Masks", True)
        self.show_polygons_check = _overlay_checkbox("Polygons", False)
        for checkbox in (
            self.show_boxes_check,
            self.show_masks_check,
            self.show_polygons_check,
        ):
            info_row.addWidget(checkbox)
        info_row.addWidget(action_button(actions.focus_workspace))
        toolbar_layout.addLayout(info_row)
        layout.addWidget(toolbar)

        self.workspace_stack = QStackedWidget()
        self.workspace_stack.setObjectName("canvasStack")
        self.empty_state = EmptyStateWidget()
        self.image_load_error = ImageLoadErrorWidget()
        self.canvas = WorkCanvas(actions)
        self.workspace_stack.addWidget(self.empty_state)
        self.workspace_stack.addWidget(self.image_load_error)
        self.workspace_stack.addWidget(self.canvas)
        layout.addWidget(self.workspace_stack, 1)

        actions.select_tool.toggled.connect(self.canvas.set_select_mode)
        actions.pan_tool.toggled.connect(self.canvas.set_pan_mode)
        actions.draw_box.toggled.connect(self.canvas.set_draw_mode)
        self.canvas.set_select_mode(True)

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
