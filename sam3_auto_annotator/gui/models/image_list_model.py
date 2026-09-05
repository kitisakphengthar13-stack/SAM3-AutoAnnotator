from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QByteArray, QAbstractListModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtGui import QBrush, QColor

from sam3_auto_annotator.core import ImageRecord, ImageStatus


IMAGE_INDEX_ROLE = int(Qt.ItemDataRole.UserRole)
IMAGE_PATH_ROLE = IMAGE_INDEX_ROLE + 1
IMAGE_NAME_ROLE = IMAGE_INDEX_ROLE + 2
IMAGE_STATUS_ROLE = IMAGE_INDEX_ROLE + 3
ANNOTATION_COUNT_ROLE = IMAGE_INDEX_ROLE + 4
IMAGE_RECORD_ROLE = IMAGE_INDEX_ROLE + 5
ERROR_MESSAGE_ROLE = IMAGE_INDEX_ROLE + 6


STATUS_LABELS = {
    ImageStatus.NOT_PREDICTED: "not predicted",
    ImageStatus.PREDICTED: "predicted",
    ImageStatus.EDITED: "edited",
    ImageStatus.REVIEWED: "reviewed",
    ImageStatus.NO_DETECTION: "no detection",
    ImageStatus.ERROR: "error",
}

STATUS_FOREGROUND_COLORS = {
    ImageStatus.NOT_PREDICTED: "#475569",
    ImageStatus.PREDICTED: "#1d4ed8",
    ImageStatus.EDITED: "#b45309",
    ImageStatus.REVIEWED: "#15803d",
    ImageStatus.NO_DETECTION: "#6d28d9",
    ImageStatus.ERROR: "#b91c1c",
}

STATUS_BACKGROUND_COLORS = {
    ImageStatus.NOT_PREDICTED: "#f8fafc",
    ImageStatus.PREDICTED: "#eff6ff",
    ImageStatus.EDITED: "#fffbeb",
    ImageStatus.REVIEWED: "#f0fdf4",
    ImageStatus.NO_DETECTION: "#f5f3ff",
    ImageStatus.ERROR: "#fef2f2",
}


_DISPLAY_ROLES = [
    int(Qt.ItemDataRole.DisplayRole),
    int(Qt.ItemDataRole.ToolTipRole),
    int(Qt.ItemDataRole.AccessibleTextRole),
    int(Qt.ItemDataRole.ForegroundRole),
    int(Qt.ItemDataRole.BackgroundRole),
    IMAGE_INDEX_ROLE,
    IMAGE_PATH_ROLE,
    IMAGE_NAME_ROLE,
    IMAGE_STATUS_ROLE,
    ANNOTATION_COUNT_ROLE,
    IMAGE_RECORD_ROLE,
    ERROR_MESSAGE_ROLE,
]


class ImageListModel(QAbstractListModel):
    """Qt list model over project images.

    ``ImageRecord`` instances remain owned by the project. The model keeps a shallow
    list copy so domain mutations can be published cheaply with :meth:`refresh`.
    """

    def __init__(self, images: Iterable[ImageRecord] = (), parent=None):
        super().__init__(parent)
        self._images: list[ImageRecord] = []
        self.set_images(images)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._images)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        image = self.image_at(index)
        if image is None:
            return None

        annotation_count = len(image.active_annotations)
        status_label = STATUS_LABELS.get(image.status, image.status.value.replace("_", " "))

        if role == Qt.ItemDataRole.DisplayRole:
            return (
                f"{image.image_index + 1:03d}  {image.image_name}\n"
                f"{status_label}    {annotation_count} annotations"
            )
        if role == Qt.ItemDataRole.ToolTipRole:
            lines = [
                image.image_path,
                f"Status: {status_label}",
                f"Annotations: {annotation_count}",
            ]
            if image.width is not None and image.height is not None:
                lines.append(f"Size: {image.width} x {image.height}")
            if image.error_message:
                lines.append(f"Error: {image.error_message}")
            return "\n".join(lines)
        if role == Qt.ItemDataRole.AccessibleTextRole:
            return f"{image.image_name}, {status_label}, {annotation_count} annotations"
        if role == Qt.ItemDataRole.ForegroundRole:
            return QBrush(QColor(STATUS_FOREGROUND_COLORS[image.status]))
        if role == Qt.ItemDataRole.BackgroundRole:
            return QBrush(QColor(STATUS_BACKGROUND_COLORS[image.status]))
        if role == IMAGE_INDEX_ROLE:
            return image.image_index
        if role == IMAGE_PATH_ROLE:
            return image.image_path
        if role == IMAGE_NAME_ROLE:
            return image.image_name
        if role == IMAGE_STATUS_ROLE:
            return image.status.value
        if role == ANNOTATION_COUNT_ROLE:
            return annotation_count
        if role == IMAGE_RECORD_ROLE:
            return image
        if role == ERROR_MESSAGE_ROLE:
            return image.error_message
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if (
            role == Qt.ItemDataRole.DisplayRole
            and orientation == Qt.Orientation.Horizontal
            and section == 0
        ):
            return "Images"
        return None

    def roleNames(self):
        roles = super().roleNames()
        roles.update(
            {
                IMAGE_INDEX_ROLE: QByteArray(b"imageIndex"),
                IMAGE_PATH_ROLE: QByteArray(b"imagePath"),
                IMAGE_NAME_ROLE: QByteArray(b"imageName"),
                IMAGE_STATUS_ROLE: QByteArray(b"imageStatus"),
                ANNOTATION_COUNT_ROLE: QByteArray(b"annotationCount"),
                IMAGE_RECORD_ROLE: QByteArray(b"imageRecord"),
                ERROR_MESSAGE_ROLE: QByteArray(b"errorMessage"),
            }
        )
        return roles

    def set_images(self, images: Iterable[ImageRecord]):
        items = list(images)
        if not all(isinstance(item, ImageRecord) for item in items):
            raise TypeError("ImageListModel accepts ImageRecord instances only.")
        self.beginResetModel()
        self._images = items
        self.endResetModel()

    def set_items(self, images: Iterable[ImageRecord]):
        """Alias matching other collection models in the GUI layer."""
        self.set_images(images)

    @property
    def images(self):
        return tuple(self._images)

    def image_at(self, index: QModelIndex | int):
        row = index.row() if isinstance(index, QModelIndex) else int(index)
        if 0 <= row < len(self._images):
            return self._images[row]
        return None

    def row_for_image_index(self, image_index: int):
        return next(
            (
                row
                for row, image in enumerate(self._images)
                if image.image_index == image_index
            ),
            -1,
        )

    def index_for_image_index(self, image_index: int):
        row = self.row_for_image_index(image_index)
        return self.index(row, 0) if row >= 0 else QModelIndex()

    def refresh(self, image_index: int | None = None):
        """Publish in-place ``ImageRecord`` changes to attached views.

        Returns ``False`` when a requested domain image index is not in the model.
        """
        if not self._images:
            return image_index is None
        if image_index is None:
            first_row, last_row = 0, len(self._images) - 1
        else:
            first_row = self.row_for_image_index(image_index)
            if first_row < 0:
                return False
            last_row = first_row
        self.dataChanged.emit(
            self.index(first_row, 0),
            self.index(last_row, 0),
            _DISPLAY_ROLES,
        )
        return True


class ImageFilterProxyModel(QSortFilterProxyModel):
    """Case-insensitive image-name/path search plus review-status filtering."""

    _GROUP_FILTERS = frozenset({"all", "needs_review", "reviewed", "error"})

    def __init__(self, parent=None):
        super().__init__(parent)
        self._search_text = ""
        self._status_filter = "all"
        self.setDynamicSortFilter(True)

    @property
    def search_text(self):
        return self._search_text

    @property
    def status_filter(self):
        return self._status_filter

    def set_search_text(self, text: str):
        normalized = str(text).strip().casefold()
        if normalized == self._search_text:
            return
        self._search_text = normalized
        self._refilter_rows()

    def set_status_filter(self, status: str | ImageStatus):
        normalized = status.value if isinstance(status, ImageStatus) else str(status).strip().casefold()
        valid_filters = self._GROUP_FILTERS | {item.value for item in ImageStatus}
        if normalized not in valid_filters:
            raise ValueError(f"Unsupported image status filter: {status!r}.")
        if normalized == self._status_filter:
            return
        self._status_filter = normalized
        self._refilter_rows()

    def _refilter_rows(self):
        # Qt 6.10 replaced invalidateFilter() with a directional change API.
        # Keep the fallback because the application supports PySide6 6.8+.
        if hasattr(self, "beginFilterChange") and hasattr(self, "endFilterChange"):
            self.beginFilterChange()
            self.endFilterChange(QSortFilterProxyModel.Direction.Rows)
            return
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        source = self.sourceModel()
        if source is None:
            return False
        index = source.index(source_row, 0, source_parent)
        if not index.isValid():
            return False

        if self._search_text:
            name = str(index.data(IMAGE_NAME_ROLE) or "").casefold()
            path = str(index.data(IMAGE_PATH_ROLE) or "").casefold()
            if self._search_text not in name and self._search_text not in path:
                return False

        status = str(index.data(IMAGE_STATUS_ROLE) or "")
        if self._status_filter == "all":
            return True
        if self._status_filter == "needs_review":
            return status != ImageStatus.REVIEWED.value
        return status == self._status_filter

    def image_at(self, index: QModelIndex | int):
        if isinstance(index, int):
            index = self.index(index, 0)
        if not index.isValid():
            return None
        source_index = self.mapToSource(index)
        source = self.sourceModel()
        if isinstance(source, ImageListModel):
            return source.image_at(source_index)
        return source_index.data(IMAGE_RECORD_ROLE)

    def index_for_image_index(self, image_index: int):
        source = self.sourceModel()
        if not isinstance(source, ImageListModel):
            return QModelIndex()
        return self.mapFromSource(source.index_for_image_index(image_index))
