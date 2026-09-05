from __future__ import annotations

from PySide6.QtCore import QItemSelection, QModelIndex, QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from sam3_auto_annotator.core.segmentation import segmentation_status_text
from sam3_auto_annotator.gui.models.annotation_table_model import (
    ANNOTATION_ID_ROLE,
    AnnotationTableModel,
)
from sam3_auto_annotator.gui.widgets.action_button import action_button
from sam3_auto_annotator.gui.widgets.numeric_field import NumericLineEdit


class AnnotationPanel(QWidget):
    annotation_selected = Signal(str)
    editing_changed = Signal()

    def __init__(self, actions, parent=None):
        super().__init__(parent)
        self.setObjectName("annotationPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        image_group = QGroupBox("Current Image")
        image_layout = QHBoxLayout(image_group)
        self.reviewed_button = action_button(actions.mark_reviewed)
        image_layout.addWidget(self.reviewed_button)
        image_layout.addStretch(1)
        layout.addWidget(image_group)

        selected_group = QGroupBox("Selected Annotation")
        selected_layout = QVBoxLayout(selected_group)
        selected_layout.setSpacing(7)

        self.selection_label = QLabel("No annotation selected")
        self.selection_label.setObjectName("selectionTitle")
        self.selection_label.setWordWrap(True)
        selected_layout.addWidget(self.selection_label)

        metadata = QGridLayout()
        metadata.setColumnStretch(1, 1)
        metadata.addWidget(_field_label("Source"), 0, 0)
        self.source_label = QLabel("-")
        metadata.addWidget(self.source_label, 0, 1)
        metadata.addWidget(_field_label("Confidence"), 1, 0)
        self.confidence_label = QLabel("-")
        metadata.addWidget(self.confidence_label, 1, 1)
        metadata.addWidget(_field_label("Segmentation"), 2, 0)
        self.segmentation_label = QLabel("none")
        self.segmentation_label.setWordWrap(True)
        metadata.addWidget(self.segmentation_label, 2, 1)
        selected_layout.addLayout(metadata)

        selected_layout.addWidget(_field_label("Class"))
        self.class_combo = QComboBox()
        self.class_combo.setAccessibleName("Selected annotation class")
        self.class_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.class_combo.setMinimumContentsLength(10)
        self.apply_class_button = action_button(actions.apply_class, stretch=True)
        selected_layout.addWidget(self.class_combo)
        selected_layout.addWidget(self.apply_class_button)

        coordinates = QGridLayout()
        coordinates.setHorizontalSpacing(6)
        coordinates.setVerticalSpacing(5)
        self.x1_edit = _coord_edit("Left x coordinate")
        self.y1_edit = _coord_edit("Top y coordinate")
        self.x2_edit = _coord_edit("Right x coordinate")
        self.y2_edit = _coord_edit("Bottom y coordinate")
        coordinates.addWidget(_field_label("x1"), 0, 0)
        coordinates.addWidget(self.x1_edit, 0, 1)
        coordinates.addWidget(_field_label("y1"), 0, 2)
        coordinates.addWidget(self.y1_edit, 0, 3)
        coordinates.addWidget(_field_label("x2"), 1, 0)
        coordinates.addWidget(self.x2_edit, 1, 1)
        coordinates.addWidget(_field_label("y2"), 1, 2)
        coordinates.addWidget(self.y2_edit, 1, 3)
        coordinates.setColumnStretch(1, 1)
        coordinates.setColumnStretch(3, 1)
        selected_layout.addLayout(coordinates)

        self.apply_box_button = action_button(actions.apply_box, stretch=True)
        self.delete_button = action_button(
            actions.delete_annotation, "dangerButton", stretch=True
        )
        self.resegment_button = action_button(actions.resegment, stretch=True)
        self.reset_sam3_button = action_button(actions.reset_sam3, stretch=True)
        for button in (
            self.apply_box_button,
            self.resegment_button,
            self.reset_sam3_button,
            self.delete_button,
        ):
            selected_layout.addWidget(button)
        selected_layout.addStretch(1)
        self.editor_scroll = QScrollArea()
        self.editor_scroll.setObjectName("annotationEditorScroll")
        self.editor_scroll.setFrameShape(QFrame.NoFrame)
        self.editor_scroll.setWidgetResizable(True)
        self.editor_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.editor_scroll.setWidget(selected_group)
        layout.addWidget(self.editor_scroll)

        table_group = QGroupBox("Annotations on Current Image")
        table_layout = QVBoxLayout(table_group)
        self.annotation_model = AnnotationTableModel(parent=self)
        self.annotation_table = QTableView()
        self.annotation_table.setObjectName("annotationTable")
        self.annotation_table.setModel(self.annotation_model)
        self.annotation_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.annotation_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.annotation_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.annotation_table.setAlternatingRowColors(True)
        self.annotation_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.annotation_table.verticalHeader().setVisible(False)
        header = self.annotation_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.annotation_table.setMinimumHeight(
            self.annotation_table.verticalHeader().defaultSectionSize() * 2
            + header.sizeHint().height()
            + self.annotation_table.frameWidth() * 2
            + self.fontMetrics().lineSpacing()
        )
        table_layout.addWidget(self.annotation_table)
        layout.addWidget(table_group, 1)

        self.annotation_table.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        self.class_combo.currentIndexChanged.connect(
            lambda _index: self.editing_changed.emit()
        )
        self.class_combo.currentTextChanged.connect(self.class_combo.setToolTip)
        for field in (self.x1_edit, self.y1_edit, self.x2_edit, self.y2_edit):
            field.textChanged.connect(lambda _text: self.editing_changed.emit())
        self.clear_details()
        self._update_editor_height()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_editor_height()

    def _update_editor_height(self):
        line_height = self.fontMetrics().lineSpacing()
        target = max(
            line_height * 14,
            min(line_height * 30, self.height() * 56 // 100),
        )
        self.editor_scroll.setFixedHeight(target)

    def set_classes(self, prompts):
        current = self.class_combo.currentText()
        blocker = QSignalBlocker(self.class_combo)
        try:
            self.class_combo.clear()
            self.class_combo.addItems(prompts)
            index = self.class_combo.findText(current)
            if index >= 0:
                self.class_combo.setCurrentIndex(index)
        finally:
            del blocker
        self.class_combo.setToolTip(self.class_combo.currentText())

    def set_annotations(self, annotations):
        self.annotation_model.set_items(annotations)

    def refresh_annotation(self, annotation_id=None):
        self.annotation_model.refresh(annotation_id)

    def selected_annotation_id(self):
        index = self.annotation_table.currentIndex()
        return str(index.data(ANNOTATION_ID_ROLE)) if index.isValid() else None

    def select_annotation(self, annotation_id):
        index = self.annotation_model.index_for_id(annotation_id)
        if not index.isValid():
            selection_model = self.annotation_table.selectionModel()
            selection_model.clearSelection()
            self.annotation_table.setCurrentIndex(QModelIndex())
            return False
        self.annotation_table.setCurrentIndex(index)
        self.annotation_table.selectRow(index.row())
        self.annotation_table.scrollTo(index)
        return True

    def show_details(self, annotation):
        blockers = self._detail_signal_blockers()
        self.selection_label.setText("Editing selected box")
        self.source_label.setText(annotation.source.value)
        self.confidence_label.setText(
            "-" if annotation.confidence is None else f"{annotation.confidence:.4f}"
        )
        self.segmentation_label.setText(segmentation_status_text(annotation))
        for field, value in zip(
            (self.x1_edit, self.y1_edit, self.x2_edit, self.y2_edit),
            annotation.box_xyxy,
        ):
            field.set_value(value)
        class_index = self.class_combo.findText(annotation.class_name)
        if class_index >= 0:
            self.class_combo.setCurrentIndex(class_index)
        self.class_combo.setToolTip(self.class_combo.currentText())
        del blockers

    def clear_details(self):
        blockers = self._detail_signal_blockers()
        self.selection_label.setText("No annotation selected")
        self.source_label.setText("-")
        self.confidence_label.setText("-")
        self.segmentation_label.setText("none")
        for field in (self.x1_edit, self.y1_edit, self.x2_edit, self.y2_edit):
            field.set_value(0)
        self.class_combo.setToolTip(self.class_combo.currentText())
        del blockers

    def box_values(self):
        return (
            self.x1_edit.value(),
            self.y1_edit.value(),
            self.x2_edit.value(),
            self.y2_edit.value(),
        )

    def _detail_signal_blockers(self):
        return [
            QSignalBlocker(widget)
            for widget in (
                self.class_combo,
                self.x1_edit,
                self.y1_edit,
                self.x2_edit,
                self.y2_edit,
            )
        ]

    def _on_selection_changed(
        self,
        _selected: QItemSelection,
        _deselected: QItemSelection,
    ):
        annotation_id = self.selected_annotation_id()
        if annotation_id:
            self.annotation_selected.emit(annotation_id)


def _field_label(text):
    label = QLabel(text)
    label.setObjectName("formLabel")
    label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    return label


def _coord_edit(accessible_name):
    edit = NumericLineEdit(value=0.0, decimals=2, minimum=0.0, maximum=999999.0)
    edit.setMinimumWidth(0)
    edit.setPlaceholderText("0.00")
    edit.setAccessibleName(accessible_name)
    return edit
