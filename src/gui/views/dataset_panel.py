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
from gui.widgets.action_button import action_button
from gui.widgets.stat_strip import StatStrip


class DatasetPanel(QWidget):
    image_selected = Signal(int)
    filter_changed = Signal()

    def __init__(self, actions, parent=None):
        super().__init__(parent)
        self.setObjectName("datasetPanel")
        self.setMinimumWidth(180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 8, 10)
        layout.setSpacing(8)

        title_row = QHBoxLayout()
        title = QLabel("Dataset")
        title.setObjectName("sectionTitle")
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(action_button(actions.previous_image, icon_only=True))
        title_row.addWidget(action_button(actions.next_image, icon_only=True))
        layout.addLayout(title_row)

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
        self.image_list.setUniformItemSizes(True)
        self.image_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.image_list.setTextElideMode(Qt.ElideMiddle)
        layout.addWidget(self.image_list, 1)

        self.search_edit.textChanged.connect(self._apply_search_filter)
        self.status_filter.currentIndexChanged.connect(self._apply_status_filter)
        self.image_list.selectionModel().currentChanged.connect(self._on_current_changed)

    def _apply_status_filter(self):
        self.filter_model.set_status_filter(self.status_filter.currentData())
        self.filter_changed.emit()

    def _apply_search_filter(self, text):
        self.filter_model.set_search_text(text)
        self.filter_changed.emit()

    def reset_filters(self, *, notify=True):
        blockers = [QSignalBlocker(self.search_edit), QSignalBlocker(self.status_filter)]
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

    def set_images(self, images, project_name=None):
        images = list(images)
        self.reset_filters(notify=False)
        self.image_model.set_images(images)
        self.image_summary_label.setText(project_name or "Current project")
        counts = Counter(item.status for item in images)
        self.stat_strip.update_counts(
            len(images),
            counts[ImageStatus.REVIEWED],
            counts[ImageStatus.EDITED],
            counts[ImageStatus.NOT_PREDICTED] + counts[ImageStatus.ERROR],
        )
        self.filter_changed.emit()

    def clear(self):
        self.reset_filters(notify=False)
        self.image_model.set_images([])
        self.image_summary_label.setText("No project loaded")
        self.stat_strip.update_counts(0, 0, 0, 0)
        self.filter_changed.emit()

    def refresh(self, image_index=None):
        self.image_model.refresh(image_index)
        images = self.image_model.images
        counts = Counter(item.status for item in images)
        self.stat_strip.update_counts(
            len(images),
            counts[ImageStatus.REVIEWED],
            counts[ImageStatus.EDITED],
            counts[ImageStatus.NOT_PREDICTED] + counts[ImageStatus.ERROR],
        )

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
