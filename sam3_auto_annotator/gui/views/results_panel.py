from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from sam3_auto_annotator.gui.widgets.action_button import action_button
from sam3_auto_annotator.gui.widgets.path_display import PathDisplay
from sam3_auto_annotator.gui.widgets.preview_image import PreviewLabel


class ResultsPanel(QWidget):
    def __init__(self, actions, parent=None):
        super().__init__(parent)
        self.setObjectName("resultsPanel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("resultsScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        status_group = QGroupBox("Export Status")
        status_layout = QVBoxLayout(status_group)
        self.result_status_label = QLabel("No export yet")
        self.result_status_label.setObjectName("resultStatus")
        self.result_status_label.setWordWrap(True)
        self.result_counts_label = QLabel("-")
        self.result_counts_label.setObjectName("mutedLabel")
        self.result_counts_label.setWordWrap(True)
        status_layout.addWidget(self.result_status_label)
        status_layout.addWidget(self.result_counts_label)
        layout.addWidget(status_group)

        files_group = QGroupBox("Output Files")
        files_layout = QFormLayout(files_group)
        files_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        files_layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.result_output_label = PathDisplay()
        self.result_csv_label = PathDisplay()
        self.result_detection_label = PathDisplay()
        self.result_segmentation_label = PathDisplay()
        self.result_skipped_label = PathDisplay()
        self.preview_label = PathDisplay()
        files_layout.addRow("Folder", self.result_output_label)
        files_layout.addRow("Box CSV", self.result_csv_label)
        files_layout.addRow("Detection", self.result_detection_label)
        files_layout.addRow("Segmentation", self.result_segmentation_label)
        files_layout.addRow("Skipped Seg", self.result_skipped_label)
        files_layout.addRow("Preview", self.preview_label)
        self.skipped_note = QLabel("No skipped segmentation annotations.")
        self.skipped_note.setObjectName("mutedLabel")
        self.skipped_note.setWordWrap(True)
        files_layout.addRow("", self.skipped_note)
        layout.addWidget(files_group)

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_thumb = PreviewLabel()
        preview_layout.addWidget(self.preview_thumb)
        layout.addWidget(preview_group)
        layout.addStretch(1)
        self.scroll_area.setWidget(content)
        outer.addWidget(self.scroll_area, 1)

        actions_footer = QWidget()
        actions_footer.setObjectName("panelActionFooter")
        footer = QVBoxLayout(actions_footer)
        footer.setContentsMargins(12, 9, 12, 10)
        footer.setSpacing(6)
        self.export_button = action_button(
            actions.export,
            "exportButton",
            stretch=True,
        )
        footer.addWidget(self.export_button)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.save_preview_button = action_button(actions.save_preview)
        self.open_preview_button = action_button(actions.open_preview)
        row.addWidget(self.save_preview_button)
        row.addWidget(self.open_preview_button)
        row.addStretch(1)
        footer.addLayout(row)
        self.open_output_button = action_button(actions.open_output)
        footer.addWidget(self.open_output_button)
        outer.addWidget(actions_footer)

    def reset(self, output_dir=None):
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
        self.result_output_label.set_path(output_dir)
        self.result_csv_label.set_path(box_csv)
        self.result_detection_label.set_path(detection_dir)
        self.result_segmentation_label.set_path(segmentation_dir)
        self.result_skipped_label.set_path(skipped_report)
        self.skipped_note.setVisible(not bool(skipped_report))
