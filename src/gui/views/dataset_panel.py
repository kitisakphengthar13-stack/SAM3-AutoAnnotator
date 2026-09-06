from __future__ import annotations

from collections import Counter

from PySide6.QtCore import QModelIndex, QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QVBoxLayout,
    QWidget,
)

from domain import ImageStatus
from gui.models.image_list_model import (
    IMAGE_INDEX_ROLE,
    ImageFilterProxyModel,
    ImageListModel,
)
from gui.widgets.stat_strip import StatStrip
from gui.widgets.list_delegates import DatasetDelegate


class DatasetPanel(QWidget):
    image_selected = Signal(int)
    filter_changed = Signal()

    def __init__(self, actions, parent=None):
        super().__init__(parent)
        self.setObjectName("datasetPanel")
        self.setMinimumWidth(200)
        self._status_counts = Counter()
        self._status_by_image_index = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 8, 10)
        layout.setSpacing(8)

        self.image_summary_label = QLabel("No project loaded")
        self.image_summary_label.setObjectName("mutedLabel")
        self.image_summary_label.setWordWrap(True)
        layout.addWidget(self.image_summary_label)

        self.stat_strip = StatStrip()
        layout.addWidget(self.stat_strip)

        self.search_edit = QLineEdit()
        self.search_edit.setObjectName("datasetSearch")
        self.search_edit.setPlaceholderText("Search images…")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setAccessibleName("Search dataset images")
        layout.addWidget(self.search_edit)

        self.status_filter = QComboBox()
        self.status_filter.setAccessibleName("Filter images by review status")
        self.status_filter.addItem("All images", "all")
        self.status_filter.addItem("Needs review", "needs_review")
        self.status_filter.addItem("Reviewed", "reviewed")
        self.status_filter.addItem("Errors", "error")
        layout.addWidget(self.status_filter)

        self.image_model = ImageListModel(parent=self)
        self.filter_model = ImageFilterProxyModel(self)
        self.filter_model.setSourceModel(self.image_model)

        self.image_list = QListView()
        self.image_list.setObjectName("imageList")
        self.image_list.setAccessibleName("Project images")
        self.image_list.setModel(self.filter_model)
        self.image_list.setItemDelegate(DatasetDelegate(self.image_list))
        self.image_list.setMouseTracking(True)
        self.image_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.image_list.setUniformItemSizes(True)
        self.image_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.image_list.setTextElideMode(Qt.ElideMiddle)
        layout.addWidget(self.image_list, 1)
        self.no_matches = QLabel("No images match your filter.")
        self.no_matches.setObjectName("selectionEmpty")
        self.no_matches.setWordWrap(True)
        layout.addWidget(self.no_matches)
        self.no_matches.hide()
        self.filter_changed.connect(self._update_empty)
        self.image_model.modelReset.connect(self._update_empty)

        self.search_edit.textChanged.connect(self._apply_search_filter)
        self.status_filter.currentIndexChanged.connect(self._apply_status_filter)
        self.image_list.selectionModel().currentChanged.connect(
            self._on_current_changed
        )

    def _update_empty(self):
        self.no_matches.setVisible(
            bool(self.image_model.rowCount()) and self.filter_model.rowCount() == 0
        )

    def _apply_status_filter(self):
        self.filter_model.set_status_filter(self.status_filter.currentData())
        self.filter_changed.emit()

    def _apply_search_filter(self, text):
        self.filter_model.set_search_text(text)
        self.filter_changed.emit()

    def reset_filters(self, *, notify=True):
        blockers = [
            QSignalBlocker(self.search_edit),
            QSignalBlocker(self.status_filter),
        ]
        self.search_edit.clear()
        self.status_filter.setCurrentIndex(0)
        del blockers
        self.filter_model.set_search_text("")
        self.filter_model.set_status_filter("all")
        if notify:
            self.filter_changed.emit()

    def _on_current_changed(self, current: QModelIndex, _previous: QModelIndex):
        if current.isValid():
            self.image_selected.emit(int(current.data(IMAGE_INDEX_ROLE)))

    def _rebuild_status_cache(self, images):
        self._status_counts = Counter(item.status for item in images)
        self._status_by_image_index = {
            item.image_index: item.status for item in images
        }

    def _update_stats(self):
        counts = self._status_counts
        self.stat_strip.update_counts(
            self.image_model.rowCount(),
            counts[ImageStatus.REVIEWED],
            counts[ImageStatus.EDITED],
            counts[ImageStatus.NOT_PREDICTED] + counts[ImageStatus.ERROR],
        )

    def set_images(self, images, project_name=None):
        images = list(images)
        self.reset_filters(notify=False)
        self._rebuild_status_cache(images)
        self.image_model.set_images(images)
        self.image_summary_label.setText(project_name or "Current project")
        self._update_stats()
        self.filter_changed.emit()

    def clear(self):
        self.reset_filters(notify=False)
        self._status_counts.clear()
        self._status_by_image_index.clear()
        self.image_model.set_images([])
        self.image_summary_label.setText("No project loaded")
        self._update_stats()
        self.filter_changed.emit()

    def refresh(self, image_index=None):
        if image_index is None:
            images = self.image_model.images
            self._rebuild_status_cache(images)
            refreshed = self.image_model.refresh()
        else:
            row = self.image_model.row_for_image_index(image_index)
            if row < 0:
                return False
            image = self.image_model.image_at(row)
            old_status = self._status_by_image_index.get(image.image_index)
            new_status = image.status
            if old_status != new_status:
                if old_status is not None:
                    self._status_counts[old_status] -= 1
                    if self._status_counts[old_status] <= 0:
                        del self._status_counts[old_status]
                self._status_counts[new_status] += 1
                self._status_by_image_index[image.image_index] = new_status
            refreshed = self.image_model.refresh(image_index)
        self._update_stats()
        return refreshed

    def selected_image_index(self):
        index = self.image_list.currentIndex()
        return int(index.data(IMAGE_INDEX_ROLE)) if index.isValid() else None

    def selected_visible_row(self):
        index = self.image_list.currentIndex()
        return index.row() if index.isValid() else -1

    def select_image(self, image_index, *, notify=True):
        index = self.filter_model.index_for_image_index(image_index)
        if not index.isValid():
            return False
        blocker = None
        if not notify:
            blocker = QSignalBlocker(self.image_list.selectionModel())
        try:
            self.image_list.setCurrentIndex(index)
            self.image_list.scrollTo(index)
        finally:
            del blocker
        return True

    def select_first(self):
        if self.filter_model.rowCount() > 0:
            self.image_list.setCurrentIndex(self.filter_model.index(0, 0))

    def select_relative(self, offset):
        count = self.filter_model.rowCount()
        if count == 0:
            return
        current = self.image_list.currentIndex().row()
        if current < 0:
            current = 0 if offset >= 0 else count - 1
        target = max(0, min(count - 1, current + offset))
        self.image_list.setCurrentIndex(self.filter_model.index(target, 0))
