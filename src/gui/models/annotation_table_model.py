from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QByteArray, QAbstractTableModel, QModelIndex, Qt

from domain import Annotation
from domain.segmentation import segmentation_status


ANNOTATION_ID_ROLE = int(Qt.ItemDataRole.UserRole)
ANNOTATION_ROLE = ANNOTATION_ID_ROLE + 1
CLASS_ID_ROLE = ANNOTATION_ID_ROLE + 2
CLASS_NAME_ROLE = ANNOTATION_ID_ROLE + 3
SOURCE_ROLE = ANNOTATION_ID_ROLE + 4
SEGMENTATION_STATUS_ROLE = ANNOTATION_ID_ROLE + 5
CONFIDENCE_ROLE = ANNOTATION_ID_ROLE + 6


_HEADERS = ("Class", "Source", "Segmentation", "Confidence")
_REFRESH_ROLES = [
    int(Qt.ItemDataRole.DisplayRole),
    int(Qt.ItemDataRole.ToolTipRole),
    int(Qt.ItemDataRole.AccessibleTextRole),
    int(Qt.ItemDataRole.TextAlignmentRole),
    ANNOTATION_ID_ROLE,
    ANNOTATION_ROLE,
    CLASS_ID_ROLE,
    CLASS_NAME_ROLE,
    SOURCE_ROLE,
    SEGMENTATION_STATUS_ROLE,
    CONFIDENCE_ROLE,
]


class AnnotationTableModel(QAbstractTableModel):
    """Read-only table projection for annotations on the current image."""

    def __init__(self, annotations: Iterable[Annotation] = (), parent=None):
        super().__init__(parent)
        self._annotations: list[Annotation] = []
        self.set_items(annotations)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._annotations)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(_HEADERS)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        annotation = self.annotation_at(index)
        if annotation is None:
            return None

        column = index.column()
        segment_status = segmentation_status(annotation)

        if role == Qt.ItemDataRole.DisplayRole:
            if column == 0:
                return annotation.class_name
            if column == 1:
                return annotation.source.value
            if column == 2:
                return segment_status
            if column == 3:
                return "-" if annotation.confidence is None else f"{annotation.confidence:.3f}"
        if role == Qt.ItemDataRole.ToolTipRole:
            x1, y1, x2, y2 = annotation.box_xyxy
            return (
                f"{annotation.class_name} ({annotation.source.value})\n"
                f"Box: {x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}\n"
                f"Segmentation: {segment_status}\n"
                f"Annotation ID: {annotation.id}"
            )
        if role == Qt.ItemDataRole.AccessibleTextRole:
            confidence = "not available" if annotation.confidence is None else f"{annotation.confidence:.3f}"
            return (
                f"{annotation.class_name}, source {annotation.source.value}, "
                f"segmentation {segment_status}, confidence {confidence}"
            )
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if column == 3:
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if role == ANNOTATION_ID_ROLE:
            return annotation.id
        if role == ANNOTATION_ROLE:
            return annotation
        if role == CLASS_ID_ROLE:
            return annotation.class_id
        if role == CLASS_NAME_ROLE:
            return annotation.class_name
        if role == SOURCE_ROLE:
            return annotation.source.value
        if role == SEGMENTATION_STATUS_ROLE:
            return segment_status
        if role == CONFIDENCE_ROLE:
            return annotation.confidence
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal and 0 <= section < len(_HEADERS):
            return _HEADERS[section]
        if orientation == Qt.Orientation.Vertical and 0 <= section < len(self._annotations):
            return section + 1
        return None

    def roleNames(self):
        roles = super().roleNames()
        roles.update(
            {
                ANNOTATION_ID_ROLE: QByteArray(b"annotationId"),
                ANNOTATION_ROLE: QByteArray(b"annotation"),
                CLASS_ID_ROLE: QByteArray(b"classId"),
                CLASS_NAME_ROLE: QByteArray(b"className"),
                SOURCE_ROLE: QByteArray(b"source"),
                SEGMENTATION_STATUS_ROLE: QByteArray(b"segmentationStatus"),
                CONFIDENCE_ROLE: QByteArray(b"confidence"),
            }
        )
        return roles

    def set_items(self, annotations: Iterable[Annotation]):
        items = list(annotations)
        if not all(isinstance(item, Annotation) for item in items):
            raise TypeError("AnnotationTableModel accepts Annotation instances only.")
        self.beginResetModel()
        self._annotations = items
        self.endResetModel()

    def annotation_at(self, index: QModelIndex | int):
        row = index.row() if isinstance(index, QModelIndex) else int(index)
        if 0 <= row < len(self._annotations):
            return self._annotations[row]
        return None

    def row_for_id(self, annotation_id: str):
        return next(
            (
                row
                for row, annotation in enumerate(self._annotations)
                if annotation.id == annotation_id
            ),
            -1,
        )

    def index_for_id(self, annotation_id: str, column: int = 0):
        row = self.row_for_id(annotation_id)
        if row < 0 or not 0 <= column < len(_HEADERS):
            return QModelIndex()
        return self.index(row, column)

    def annotation_by_id(self, annotation_id: str):
        row = self.row_for_id(annotation_id)
        return self._annotations[row] if row >= 0 else None

    def refresh(self, annotation_id: str | None = None):
        """Publish in-place annotation changes while preserving view selection."""
        if not self._annotations:
            return annotation_id is None
        if annotation_id is None:
            first_row, last_row = 0, len(self._annotations) - 1
        else:
            first_row = self.row_for_id(annotation_id)
            if first_row < 0:
                return False
            last_row = first_row
        self.dataChanged.emit(
            self.index(first_row, 0),
            self.index(last_row, len(_HEADERS) - 1),
            _REFRESH_ROLES,
        )
        return True
