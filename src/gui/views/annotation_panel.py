from __future__ import annotations
from PySide6.QtCore import QItemSelection, QModelIndex, QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QPushButton,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from domain.segmentation import segmentation_status_text
from gui.icons import icon
from gui.models.annotation_table_model import ANNOTATION_ID_ROLE, AnnotationTableModel
from gui.widgets.action_button import action_button
from gui.widgets.list_delegates import ObjectDelegate
from gui.widgets.numeric_field import NumericLineEdit


class AnnotationPanel(QWidget):
    annotation_selected = Signal(str)
    editing_changed = Signal()

    def __init__(self, actions, parent=None):
        super().__init__(parent)
        self.setObjectName("annotationPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 10)
        layout.setSpacing(8)
        header = QHBoxLayout()
        label = QLabel("IN THIS IMAGE")
        label.setObjectName("mutedLabel")
        header.addWidget(label)
        header.addStretch()
        self.count_label = QLabel("0")
        self.count_label.setObjectName("countBadge")
        header.addWidget(self.count_label)
        layout.addLayout(header)

        self.annotation_model = AnnotationTableModel(parent=self)
        self.annotation_table = QTableView()
        self.annotation_table.setObjectName("annotationTable")
        self.annotation_table.setAccessibleName("Objects in this image")
        self.annotation_table.setModel(self.annotation_model)
        self.annotation_table.setItemDelegate(ObjectDelegate(self.annotation_table))
        self.annotation_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.annotation_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.annotation_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.annotation_table.setMouseTracking(True)
        self.annotation_table.setShowGrid(False)
        self.annotation_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.annotation_table.verticalHeader().hide()
        self.annotation_table.verticalHeader().setDefaultSectionSize(61)
        self.annotation_table.horizontalHeader().hide()
        self.annotation_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        for column in range(1, 4):
            self.annotation_table.setColumnHidden(column, True)
        layout.addWidget(self.annotation_table, 1)
        self.empty_label = QLabel(
            "No objects yet.\nDraw a box or use Assist to get started."
        )
        self.empty_label.setObjectName("selectionEmpty")
        self.empty_label.setWordWrap(True)
        layout.addWidget(self.empty_label)
        self.selection_hint = QLabel(
            "Select a box on the image or an object above to edit it."
        )
        self.selection_hint.setObjectName("selectionEmpty")
        self.selection_hint.setWordWrap(True)
        layout.addWidget(self.selection_hint)

        self.details_scroll = QScrollArea()
        self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setMaximumHeight(320)
        self.details_scroll.setMinimumHeight(250)
        self.details_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.details = QWidget()
        details = QVBoxLayout(self.details)
        details.setContentsMargins(0, 10, 0, 0)
        details.setSpacing(10)
        self.selection_label = QLabel("No annotation selected")
        self.selection_label.setObjectName("selectionTitle")
        self.selection_label.setWordWrap(True)
        details.addWidget(self.selection_label)
        metadata = QHBoxLayout()
        self.source_label = QLabel("-")
        self.source_label.setObjectName("mutedLabel")
        self.confidence_label = QLabel("-")
        self.confidence_label.setObjectName("mutedLabel")
        metadata.addWidget(self.source_label)
        metadata.addStretch()
        metadata.addWidget(self.confidence_label)
        details.addLayout(metadata)
        self.segmentation_label = QLabel("none")
        self.segmentation_label.setObjectName("mutedLabel")
        self.segmentation_label.setWordWrap(True)
        details.addWidget(self.segmentation_label)

        row = QHBoxLayout()
        self.class_combo = QComboBox()
        self.class_combo.setAccessibleName("Selected annotation class")
        self.class_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.class_combo.setMinimumContentsLength(6)
        row.addWidget(self.class_combo, 1)
        self.apply_class_button = action_button(actions.apply_class, icon_only=True)
        row.addWidget(self.apply_class_button)
        details.addLayout(row)

        self.coordinates_button = QToolButton()
        self.coordinates_button.setText("Edit coordinates…")
        self.coordinates_button.setObjectName("quietButton")
        self.coordinates_button.setIcon(icon("next"))
        self.coordinates_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.coordinates_button.setAccessibleName("Edit exact box coordinates")
        details.addWidget(self.coordinates_button)
        self.coordinates_dialog = QDialog(self)
        self.coordinates_dialog.setWindowTitle("Box coordinates")
        self.coordinates_dialog.setWindowModality(Qt.WindowModal)
        self.coordinates_dialog.resize(420, 300)
        dialog_layout = QVBoxLayout(self.coordinates_dialog)
        dialog_layout.setContentsMargins(24, 20, 24, 20)
        heading = QLabel("Edit box coordinates")
        heading.setObjectName("dialogTitle")
        dialog_layout.addWidget(heading)
        hint = QLabel(
            "Pixel positions in the source image. Apply updates the box; Cancel discards your changes."
        )
        hint.setWordWrap(True)
        hint.setObjectName("mutedLabel")
        dialog_layout.addWidget(hint)
        self.coordinates_widget = QWidget()
        coordinates = QGridLayout(self.coordinates_widget)
        coordinates.setContentsMargins(0, 0, 0, 0)
        coordinates.setSpacing(6)
        self.x1_edit = _coord_edit("Left x coordinate")
        self.y1_edit = _coord_edit("Top y coordinate")
        self.x2_edit = _coord_edit("Right x coordinate")
        self.y2_edit = _coord_edit("Bottom y coordinate")
        for row_index, fields in enumerate(
            ((self.x1_edit, self.y1_edit), (self.x2_edit, self.y2_edit))
        ):
            for column, field in enumerate(fields):
                coordinates.addWidget(
                    _field_label(("x" if column == 0 else "y") + str(row_index + 1)),
                    row_index * 2,
                    column,
                )
                coordinates.addWidget(field, row_index * 2 + 1, column)
        dialog_layout.addWidget(self.coordinates_widget)
        footer = QHBoxLayout()
        footer.addStretch()
        self.cancel_coordinates_button = QPushButton("Cancel")
        self.cancel_coordinates_button.clicked.connect(self.coordinates_dialog.reject)
        footer.addWidget(self.cancel_coordinates_button)
        self.apply_box_button = action_button(actions.apply_box, "primaryButton")
        footer.addWidget(self.apply_box_button)
        dialog_layout.addLayout(footer)
        self._coordinate_snapshot = None
        self.coordinates_dialog.rejected.connect(self._restore_coordinates)
        self.coordinates_dialog.accepted.connect(self._accept_coordinates)
        self.coordinates_button.clicked.connect(self.show_coordinates)

        row = QHBoxLayout()
        self.resegment_button = action_button(actions.resegment, stretch=True)
        self.delete_button = action_button(
            actions.delete_annotation, "dangerButton", icon_only=True
        )
        self.reset_sam3_button = action_button(actions.reset_sam3, icon_only=True)
        row.addWidget(self.resegment_button, 1)
        row.addWidget(self.reset_sam3_button)
        row.addWidget(self.delete_button)
        details.addLayout(row)
        self.details_scroll.setWidget(self.details)
        layout.addWidget(self.details_scroll)
        self.annotation_table.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        self.class_combo.currentIndexChanged.connect(
            lambda _: self.editing_changed.emit()
        )
        self.class_combo.currentTextChanged.connect(self.class_combo.setToolTip)
        for field in (self.x1_edit, self.y1_edit, self.x2_edit, self.y2_edit):
            field.textChanged.connect(lambda _: self.editing_changed.emit())
        self.clear_details()

    def show_coordinates(self):
        fields = (self.x1_edit, self.y1_edit, self.x2_edit, self.y2_edit)
        if not self.coordinates_dialog.isVisible():
            self._coordinate_snapshot = [field.text() for field in fields]
        self.coordinates_dialog.show()
        self.coordinates_dialog.raise_()
        self.x1_edit.setFocus()
        self.x1_edit.selectAll()

    def _restore_coordinates(self):
        if self._coordinate_snapshot is not None:
            blockers = self._detail_signal_blockers()
            for field, value in zip(
                (self.x1_edit, self.y1_edit, self.x2_edit, self.y2_edit),
                self._coordinate_snapshot,
            ):
                field.setText(value)
            del blockers
        self._coordinate_snapshot = None
        self.editing_changed.emit()

    def _accept_coordinates(self):
        self._coordinate_snapshot = None

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
        annotations = list(annotations)
        self.annotation_model.set_items(annotations)
        self.count_label.setText(str(len(annotations)))
        self.empty_label.setVisible(not annotations)

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
        self.details_scroll.show()
        self.selection_hint.hide()
        blockers = self._detail_signal_blockers()
        self.selection_label.setText(f"Selected · {annotation.class_name}")
        self.source_label.setText(annotation.source.value)
        self.confidence_label.setText(
            "-" if annotation.confidence is None else f"{annotation.confidence:.3f}"
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
        self.details_scroll.hide()
        self.selection_hint.setVisible(self.annotation_model.rowCount() > 0)
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
