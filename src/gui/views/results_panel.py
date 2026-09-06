from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from gui.widgets.action_button import action_button
from gui.widgets.path_display import PathDisplay
from gui.widgets.preview_image import PreviewLabel


class ResultsPanel(QWidget):
    close_requested = Signal()

    def __init__(self, actions, parent=None):
        super().__init__(parent)
        self.setObjectName("resultsPanel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        heading = QWidget()
        head = QVBoxLayout(heading)
        head.setContentsMargins(24, 20, 24, 12)
        eyebrow = QLabel("EXPORT DATASET")
        eyebrow.setObjectName("eyebrow")
        head.addWidget(eyebrow)
        self.title_label = QLabel("Ready for the next step?")
        self.title_label.setObjectName("dialogTitle")
        head.addWidget(self.title_label)
        outer.addWidget(heading)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 16, 24, 20)
        layout.setSpacing(14)
        self.result_status_label = QLabel("No export yet")
        self.result_status_label.setWordWrap(True)
        layout.addWidget(self.result_status_label)
        self.metrics = QWidget()
        grid = QGridLayout(self.metrics)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)
        self.metric_values = []
        for i, caption in enumerate(
            ("Reviewed images", "Needs review", "Not predicted / failed", "Mask issues")
        ):
            card = QWidget()
            card.setObjectName("metricCard")
            box = QVBoxLayout(card)
            box.setContentsMargins(14, 12, 14, 12)
            value = QLabel("0")
            value.setObjectName("metricValue")
            box.addWidget(value)
            label = QLabel(caption)
            label.setObjectName("mutedLabel")
            label.setWordWrap(True)
            box.addWidget(label)
            grid.addWidget(card, 0, i)
            grid.setColumnStretch(i, 1)
            self.metric_values.append(value)
        layout.addWidget(self.metrics)
        self.result_counts_label = QLabel("-")
        self.result_counts_label.setObjectName("mutedLabel")
        self.result_counts_label.setWordWrap(True)
        layout.addWidget(self.result_counts_label)
        self.warning_label = QLabel()
        self.warning_label.setObjectName("exportWarning")
        self.warning_label.setWordWrap(True)
        layout.addWidget(self.warning_label)
        formats = QLabel("CSV boxes  ·  YOLO detection  ·  Valid YOLO segmentation")
        formats.setWordWrap(True)
        formats.setObjectName("mutedLabel")
        layout.addWidget(formats)
        layout.addStretch()
        self.scroll_area.setWidget(content)
        self.tabs.addTab(self.scroll_area, "Overview")

        files_scroll = QScrollArea()
        files_scroll.setWidgetResizable(True)
        files = QWidget()
        files_layout = QVBoxLayout(files)
        files_layout.setContentsMargins(24, 16, 24, 20)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.result_output_label = PathDisplay()
        self.result_csv_label = PathDisplay()
        self.result_detection_label = PathDisplay()
        self.result_segmentation_label = PathDisplay()
        self.result_skipped_label = PathDisplay()
        self.preview_label = PathDisplay()
        for name, widget in (
            ("Folder", self.result_output_label),
            ("Box CSV", self.result_csv_label),
            ("Detection", self.result_detection_label),
            ("Segmentation", self.result_segmentation_label),
            ("Skipped masks", self.result_skipped_label),
            ("Preview", self.preview_label),
        ):
            form.addRow(name, widget)
        files_layout.addLayout(form)
        self.skipped_note = QLabel("No skipped segmentation annotations.")
        self.skipped_note.setObjectName("mutedLabel")
        self.skipped_note.setWordWrap(True)
        files_layout.addWidget(self.skipped_note)
        self.preview_thumb = PreviewLabel()
        files_layout.addWidget(self.preview_thumb)
        row = QHBoxLayout()
        self.save_preview_button = action_button(actions.save_preview)
        self.open_preview_button = action_button(actions.open_preview)
        row.addWidget(self.save_preview_button)
        row.addWidget(self.open_preview_button)
        row.addStretch()
        files_layout.addLayout(row)
        files_scroll.setWidget(files)
        self.tabs.addTab(files_scroll, "Files and preview")

        footer = QWidget()
        footer.setObjectName("panelActionFooter")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(24, 12, 24, 16)
        footer_layout.setSpacing(10)
        destination_row = QHBoxLayout()
        destination = QLabel("Destination")
        destination.setObjectName("mutedLabel")
        destination_row.addWidget(destination)
        self.destination_label = PathDisplay()
        destination_row.addWidget(self.destination_label, 1)
        footer_layout.addLayout(destination_row)
        row = QHBoxLayout()
        footer_layout.addLayout(row)
        self.open_output_button = action_button(actions.open_output)
        row.addWidget(self.open_output_button)
        row.addStretch()
        self.close_button = QPushButton("Back to review")
        self.close_button.clicked.connect(self.close_requested.emit)
        row.addWidget(self.close_button)
        self.export_button = action_button(actions.export, "exportButton")
        row.addWidget(self.export_button)
        outer.addWidget(footer)
        self.set_phase("idle")

    def set_phase(self, phase):
        self.phase = phase
        preflight = phase == "preflight"
        self.metrics.setVisible(preflight)
        self.warning_label.setVisible(preflight and bool(self.warning_label.text()))
        self.result_counts_label.setVisible(not preflight)
        self.tabs.setTabEnabled(1, not preflight)
        self.export_button.setVisible(preflight)
        self.open_output_button.setVisible(not preflight)
        self.close_button.setText("Back to review" if preflight else "Done")
        if preflight:
            self.title_label.setText("Ready for the next step?")
            self.tabs.setCurrentIndex(0)
        elif phase == "complete":
            self.title_label.setText("Your labels are exported")
            self.tabs.setCurrentIndex(0)

    def set_preflight(self, readiness, output_dir=None):
        values = (
            f"{readiness.reviewed_images} / {readiness.total_images}",
            readiness.needs_review,
            readiness.incomplete_images,
            readiness.stale_segmentations,
        )
        for label, value in zip(self.metric_values, values):
            label.setText(str(value))
        warnings = []
        if readiness.needs_review:
            warnings.append(
                "Unreviewed images will be included. Check them before using the dataset for training."
            )
        if readiness.stale_segmentations:
            warnings.append(
                "Objects without valid masks export as boxes only. Their segmentation is skipped and reported."
            )
        self.warning_label.setText("\n\n".join(warnings))
        if output_dir:
            self.destination_label.set_path(output_dir)
        self.set_phase("preflight")

    def reset(self, output_dir=None):
        self.set_phase("idle")
        self.title_label.setText("Dataset output")
        self.set_status("Project ready. No export in this session yet.")
        self.set_output_paths(output_dir=output_dir)
        self.preview_label.set_path(None)
        self.preview_thumb.clear_preview()

    def set_preview(self, path):
        loaded = self.preview_thumb.set_image(path)
        if loaded:
            self.preview_label.set_path(path)
            return True
        self.preview_label.set_path(None)
        self.preview_thumb.clear_preview()
        return False

    def set_output_dir(self, output_dir):
        self.result_output_label.set_path(output_dir)
        self.destination_label.set_path(output_dir)

    def set_status(self, message, counts=None):
        self.result_status_label.setText(str(message))
        self.result_counts_label.setText("-" if counts is None else str(counts))

    def set_output_paths(
        self,
        *,
        output_dir=None,
        box_csv=None,
        detection_dir=None,
        segmentation_dir=None,
        skipped_report=None,
    ):
        self.set_output_dir(output_dir)
        self.result_csv_label.set_path(box_csv)
        self.result_detection_label.set_path(detection_dir)
        self.result_segmentation_label.set_path(segmentation_dir)
        self.result_skipped_label.set_path(skipped_report)
        self.skipped_note.setVisible(not bool(skipped_report))
