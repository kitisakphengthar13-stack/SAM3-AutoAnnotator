from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGraphicsItem,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMenu,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)
from shiboken6 import isValid

from gui.widgets.action_button import action_button, menu_button
from gui.widgets.elided_label import ElidedLabel
from gui.widgets.empty_state import EmptyStateWidget
from gui.widgets.image_canvas import ImageCanvas
from gui.widgets.image_load_error import ImageLoadErrorWidget
from gui.widgets.task_progress import TaskProgressWidget


NARROW_WORKSPACE_BREAKPOINT = 1120
MIN_ADAPTIVE_WIDE_WIDTH = 1024


class AnnotationLabel(QGraphicsSimpleTextItem):
    """Constant-size, click-through label readable on light and dark images."""

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#101b24"))
        painter.drawRoundedRect(self.boundingRect(), 3, 3)
        super().paint(painter, option, widget)

    def boundingRect(self):
        return super().boundingRect().adjusted(-3, -2, 3, 2)


class WorkCanvas(ImageCanvas):
    zoom_changed = Signal(float)

    def __init__(self, actions, parent=None):
        super().__init__(parent)
        self._actions = actions
        self._temporary_pan = False
        self._labels_by_id = {}
        self.setBackgroundBrush(QColor("#0b0f14"))
        self.setFocusPolicy(Qt.StrongFocus)

    def zoom_in(self):
        self._zoom_by(1.2)

    def zoom_out(self):
        self._zoom_by(1 / 1.2)

    def actual_size(self):
        if self._pixmap_item is not None:
            self._auto_fit = False
            self.resetTransform()
            self.zoom_changed.emit(1.0)

    def _zoom_by(self, factor):
        if self._pixmap_item is None:
            return
        current = self.transform().m11()
        target = max(0.05, min(20.0, current * factor))
        self._auto_fit = False
        self.scale(target / current, target / current)
        self.zoom_changed.emit(target)

    def fit_to_window(self):
        super().fit_to_window()
        self.zoom_changed.emit(self.transform().m11())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.zoom_changed.emit(self.transform().m11())

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta:
            self._zoom_by(1.15 if delta > 0 else 1 / 1.15)
        event.accept()

    def _sync_mode(self):
        pan = self._temporary_pan or self._actions.pan_tool.isChecked()
        draw = self._actions.draw_box.isChecked() and not pan
        if not draw:
            self.cancel_drawing()
        super().set_draw_mode(draw)
        self.setInteractive(not pan and not draw)
        self.setDragMode(QGraphicsView.ScrollHandDrag if pan else QGraphicsView.NoDrag)
        self.viewport().setCursor(
            Qt.OpenHandCursor if pan else Qt.CrossCursor if draw else Qt.ArrowCursor
        )

    def set_select_mode(self, enabled):
        if enabled:
            self._temporary_pan = False
        self._sync_mode()

    def set_pan_mode(self, enabled):
        self._sync_mode()

    def set_draw_mode(self, enabled):
        self._sync_mode()

    def cancel_drawing(self):
        if self._draft_item is not None:
            self._scene.removeItem(self._draft_item)
            self._draft_item = None
        self._drawing = False

    def clear_image(self):
        self._labels_by_id.clear()
        super().clear_image()

    def remove_annotation(self, annotation_id):
        self._labels_by_id.pop(annotation_id, None)
        super().remove_annotation(annotation_id)

    def set_annotations(self, annotations):
        annotations = list(annotations)
        super().set_annotations(annotations)
        active_by_id = {
            annotation.id: annotation
            for annotation in annotations
            if not annotation.deleted and annotation.id in self._items_by_id
        }
        for annotation_id in list(self._labels_by_id):
            label = self._labels_by_id.get(annotation_id)
            box = self._items_by_id.get(annotation_id)
            if (
                annotation_id not in active_by_id
                or not isValid(label)
                or label.parentItem() is not box
            ):
                self._labels_by_id.pop(annotation_id, None)

        for annotation_id, annotation in active_by_id.items():
            box = self._items_by_id[annotation_id]
            confidence = (
                ""
                if annotation.confidence is None
                else f"  {annotation.confidence:.2f}"
            )
            text = f"{annotation.class_name}{confidence}"
            label = self._labels_by_id.get(annotation_id)
            if label is None:
                label = AnnotationLabel(text, box)
                label.setBrush(QBrush(QColor("#ffffff")))
                label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                label.setAcceptedMouseButtons(Qt.NoButton)
                label.setPos(4, 3)
                label.setZValue(30)
                self._labels_by_id[annotation_id] = label
            elif label.text() != text:
                label.setText(text)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and not event.isAutoRepeat():
            self.cancel_drawing()
            self._actions.select_tool.trigger()
            event.accept()
            return
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._temporary_pan = True
            self._sync_mode()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._temporary_pan = False
            self._sync_mode()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self._temporary_pan = False
        self.cancel_drawing()
        self._sync_mode()
        super().focusOutEvent(event)


class CanvasWorkspace(QWidget):
    def __init__(self, actions, parent=None):
        super().__init__(parent)
        self.setObjectName("canvasWorkspace")
        self.setMinimumWidth(360)
        self._actions = actions
        self._responsive_dataset_auto_hidden = False
        self._responsive_dataset_override = None
        self._responsive_dataset_connected = False
        self._responsive_guard = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QWidget()
        toolbar.setObjectName("canvasBar")
        row = QHBoxLayout(toolbar)
        row.setContentsMargins(12, 8, 10, 8)
        row.setSpacing(8)
        label = QLabel("New box class")
        label.setObjectName("mutedLabel")
        row.addWidget(label)
        self.active_class_combo = QComboBox()
        self.active_class_combo.setObjectName("activeClassCombo")
        self.active_class_combo.setMinimumWidth(120)
        self.active_class_combo.setMaximumWidth(180)
        self.active_class_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.active_class_combo.setMinimumContentsLength(6)
        self.active_class_combo.setAccessibleName("Class for the next new box")
        self.active_class_combo.setToolTip(
            "Class assigned to the next box you draw. Selected objects are unchanged."
        )
        row.addWidget(self.active_class_combo, 1)
        row.addStretch(1)

        self.overlay_menu = QMenu(self)
        self.show_boxes_check = self._overlay("Boxes", True)
        self.show_masks_check = self._overlay("Masks", True)
        self.show_polygons_check = self._overlay("Polygons", False)
        self.overlay_button = menu_button("Layers", "preview", self.overlay_menu)
        row.addWidget(self.overlay_button)
        self.focus_button = action_button(actions.focus_workspace, icon_only=True)
        row.addWidget(self.focus_button)
        self.fullscreen_button = action_button(actions.fullscreen, icon_only=True)
        row.addWidget(self.fullscreen_button)
        layout.addWidget(toolbar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        rail = QWidget()
        rail.setObjectName("toolRail")
        rail.setFixedWidth(52)
        tools = QVBoxLayout(rail)
        tools.setContentsMargins(6, 10, 6, 10)
        tools.setSpacing(6)
        self.tool_buttons = {}
        for action in (actions.select_tool, actions.pan_tool, actions.draw_box):
            button = action_button(action, "railButton", icon_only=True)
            button.setFixedSize(34, 34)
            self.tool_buttons[action] = button
            tools.addWidget(button)
        tools.addSpacing(10)
        for action in (actions.undo, actions.redo):
            button = action_button(action, "railButton", icon_only=True)
            button.setFixedSize(34, 34)
            self.tool_buttons[action] = button
            tools.addWidget(button)
        tools.addStretch(1)
        for action in (
            actions.zoom_in,
            actions.zoom_out,
            actions.actual_size,
            actions.fit,
        ):
            button = action_button(action, "railButton", icon_only=True)
            button.setFixedSize(34, 34)
            self.tool_buttons[action] = button
            tools.addWidget(button)
        body.addWidget(rail)

        self.workspace_stack = QStackedWidget()
        self.workspace_stack.setObjectName("canvasStack")
        self.empty_state = EmptyStateWidget()
        self.image_load_error = ImageLoadErrorWidget()
        self.canvas = WorkCanvas(actions)
        for widget in (self.empty_state, self.image_load_error, self.canvas):
            self.workspace_stack.addWidget(widget)
        body.addWidget(self.workspace_stack, 1)
        layout.addLayout(body, 1)

        footer = QWidget()
        footer.setObjectName("reviewBar")
        bottom = QVBoxLayout(footer)
        bottom.setContentsMargins(10, 6, 10, 8)
        bottom.setSpacing(6)
        info = QHBoxLayout()
        self.canvas_hint = ElidedLabel("Open images to begin")
        self.canvas_hint.setObjectName("canvasHint")
        info.addWidget(self.canvas_hint, 1)
        self.zoom_label = QLabel("—")
        self.zoom_label.setFixedWidth(52)
        self.zoom_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.zoom_label.setObjectName("mutedLabel")
        self.zoom_label.setToolTip("Current image zoom · Ctrl+0 for 100%")
        info.addWidget(self.zoom_label)
        bottom.addLayout(info)
        nav = QHBoxLayout()
        nav.setSpacing(6)
        self.previous_button = action_button(actions.previous_image, icon_only=True)
        self.next_button = action_button(actions.next_image, icon_only=True)
        nav.addWidget(self.previous_button)
        self.position_label = QLabel("No image")
        self.position_label.setObjectName("mutedLabel")
        nav.addWidget(self.position_label)
        nav.addWidget(self.next_button)
        nav.addStretch(1)
        self.review_button = action_button(actions.mark_reviewed, "primaryButton")
        nav.addWidget(self.review_button)
        bottom.addLayout(nav)
        layout.addWidget(footer)
        self.task_progress = TaskProgressWidget(actions.cancel_batch)
        layout.addWidget(self.task_progress)

        self.canvas.zoom_changed.connect(
            lambda scale: self.zoom_label.setText(f"{scale * 100:.0f}%")
        )
        actions.select_tool.toggled.connect(self.canvas.set_select_mode)
        actions.pan_tool.toggled.connect(self.canvas.set_pan_mode)
        actions.draw_box.toggled.connect(self.canvas.set_draw_mode)
        self.canvas.set_select_mode(True)

    def showEvent(self, event):
        super().showEvent(event)
        self._install_responsive_workspace()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._responsive_guard:
            self._apply_responsive_workspace()

    def _responsive_breakpoint(self):
        window = self.window()
        screen = window.screen()
        if screen is None:
            return NARROW_WORKSPACE_BREAKPOINT
        available_width = screen.availableGeometry().width()
        adaptive_ceiling = max(MIN_ADAPTIVE_WIDE_WIDTH, available_width)
        return min(NARROW_WORKSPACE_BREAKPOINT, adaptive_ceiling)

    def _install_responsive_workspace(self):
        if not isValid(self):
            return
        window = self.window()
        dataset_dock = getattr(window, "dataset_dock", None)
        if dataset_dock is None:
            return
        if not self._responsive_dataset_connected:
            dataset_dock.visibilityChanged.connect(
                self._dataset_visibility_changed
            )
            self._responsive_dataset_connected = True
        self._apply_responsive_workspace()

    def _dataset_visibility_changed(self, visible):
        if (
            not isValid(self)
            or self._responsive_guard
            or self._actions.focus_workspace.isChecked()
        ):
            return
        window = self.window()
        if window.width() < self._responsive_breakpoint():
            self._responsive_dataset_override = bool(visible)
            if visible:
                self._responsive_dataset_auto_hidden = False
        else:
            self._responsive_dataset_override = None
            self._responsive_dataset_auto_hidden = False

    def _apply_responsive_workspace(self):
        if not isValid(self) or self._responsive_guard:
            return
        window = self.window()
        dataset_dock = getattr(window, "dataset_dock", None)
        if dataset_dock is None or self._actions.focus_workspace.isChecked():
            return
        narrow = window.width() < self._responsive_breakpoint()
        if narrow:
            if (
                self._responsive_dataset_override is None
                and dataset_dock.isVisible()
            ):
                self._responsive_guard = True
                try:
                    dataset_dock.hide()
                finally:
                    self._responsive_guard = False
                self._responsive_dataset_auto_hidden = True
            return

        self._responsive_dataset_override = None
        if self._responsive_dataset_auto_hidden:
            self._responsive_guard = True
            try:
                dataset_dock.show()
            finally:
                self._responsive_guard = False
        self._responsive_dataset_auto_hidden = False

    def _overlay(self, text, checked):
        checkbox = QCheckBox(text)
        checkbox.setChecked(checked)
        checkbox.setContentsMargins(10, 8, 20, 8)
        checkbox.setToolTip(f"Show {text.lower()} on the canvas and in saved previews")
        action = QWidgetAction(self.overlay_menu)
        action.setDefaultWidget(checkbox)
        self.overlay_menu.addAction(action)
        return checkbox

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
