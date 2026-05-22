from pathlib import Path

from PySide6.QtCore import QSize, QUrl, Qt
from PySide6.QtGui import QColor, QDesktopServices, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QFileDialog,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sam3_auto_annotator.annotation.models import ImageStatus
from sam3_auto_annotator.annotation.yolo_importer import import_yolo_detection_labels
from sam3_auto_annotator.gui.fields import NumericLineEdit, configure_c_locale
from sam3_auto_annotator.gui.icons import ICONS, icon
from sam3_auto_annotator.gui.image_canvas import BOX_COLORS, ImageCanvas
from sam3_auto_annotator.gui.project_ops import (
    create_project,
    default_output_dir,
    export_project,
    load_state,
    parse_prompts,
    remaining_prediction_targets,
    save_state_to_output,
)
from sam3_auto_annotator.gui.predictor_cache import PredictorCache
from sam3_auto_annotator.gui.theme import APP_STYLESHEET
from sam3_auto_annotator.gui.widgets import EmptyStateWidget, StatStrip
from sam3_auto_annotator.gui.workers import BatchPredictionWorker, PredictionWorker
from sam3_auto_annotator.paths import validate_model_path


STATUS_LABELS = {
    ImageStatus.NOT_PREDICTED: "not predicted",
    ImageStatus.PREDICTED: "predicted",
    ImageStatus.EDITED: "edited",
    ImageStatus.REVIEWED: "reviewed",
    ImageStatus.NO_DETECTION: "no detection",
    ImageStatus.ERROR: "error",
}

STATUS_COLORS = {
    ImageStatus.NOT_PREDICTED: "#64748b",
    ImageStatus.PREDICTED: "#2563eb",
    ImageStatus.EDITED: "#d97706",
    ImageStatus.REVIEWED: "#16a34a",
    ImageStatus.NO_DETECTION: "#7c3aed",
    ImageStatus.ERROR: "#dc2626",
}

STATUS_BACKGROUNDS = {
    ImageStatus.NOT_PREDICTED: "#ffffff",
    ImageStatus.PREDICTED: "#eff6ff",
    ImageStatus.EDITED: "#fffbeb",
    ImageStatus.REVIEWED: "#f0fdf4",
    ImageStatus.NO_DETECTION: "#f5f3ff",
    ImageStatus.ERROR: "#fef2f2",
}


class MainWindow(QMainWindow):
    def __init__(self):
        configure_c_locale()
        super().__init__()
        self.setWindowTitle("SAM3 AutoAnnotator")
        self.resize(1460, 880)

        self.project_state = None
        self.current_image_index = None
        self.current_state_path = None
        self.prediction_worker = None
        self.batch_worker = None
        self.unsaved = False
        self._updating_details = False
        self.last_export_result = None
        self.last_preview_path = None
        self.predictor_cache = PredictorCache()

        self._build_command_bar()
        self._build_ui()
        self.setStyleSheet(APP_STYLESHEET)
        self._set_project_enabled(False)
        self._set_annotation_detail_enabled(False)

    def _build_command_bar(self):
        toolbar = QToolBar("Command Bar")
        toolbar.setObjectName("commandBar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(14, 14))
        self.addToolBar(toolbar)

        brand_icon = QLabel()
        brand_icon.setPixmap(icon(ICONS["app"], "#334155").pixmap(14, 14))
        title = QLabel("SAM3 AutoAnnotator")
        title.setObjectName("appTitle")
        brand_separator = QLabel()
        brand_separator.setObjectName("brandSeparator")
        brand_separator.setFixedWidth(1)
        self.project_subtitle = QLabel("No project loaded")
        self.project_subtitle.setObjectName("projectSubtitle")
        toolbar.addWidget(brand_icon)
        toolbar.addWidget(title)
        toolbar.addWidget(brand_separator)
        toolbar.addWidget(self.project_subtitle)
        toolbar.addSeparator()

        self.open_image_button = self._button("Open Image", ICONS["image"], self.open_image)
        self.open_folder_button = self._button("Open Folder", ICONS["folder"], self.open_folder)
        self.open_state_button = self._button("Open State", ICONS["state"], self.open_state)
        self.import_yolo_button = self._button("Import YOLO", ICONS["state"], self.import_yolo_labels)
        self.save_button = self._button("Save", ICONS["save"], self.save_project)
        toolbar.addWidget(self.open_image_button)
        toolbar.addWidget(self.open_folder_button)
        toolbar.addWidget(self.open_state_button)
        toolbar.addWidget(self.import_yolo_button)
        toolbar.addWidget(self.save_button)
        toolbar.addSeparator()

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        self.run_current_button = self._button("Run SAM3", ICONS["sam3"], self.run_sam3_current, "#2563eb")
        self.run_current_button.setObjectName("primaryButton")
        self.run_current_button.setToolTip("Run SAM3 on the selected image and create editable draft boxes.")
        self.run_all_button = self._button("Run All Remaining", ICONS["sam3"], self.run_sam3_all_remaining, "#2563eb")
        self.run_all_button.setObjectName("primaryButton")
        self.run_all_button.setToolTip("Run SAM3 on not-predicted/error images only. Edited/reviewed images are skipped.")
        self.draw_button = QToolButton()
        self.draw_button.setObjectName("drawButton")
        self.draw_button.setText("Draw Box")
        self.draw_button.setIcon(icon(ICONS["draw"], "#334155", scale_factor=0.85))
        self.draw_button.setCheckable(True)
        self.draw_button.setToolTip("Toggle manual box drawing. Drag on the image to create a box.")
        self.draw_button.toggled.connect(self._toggle_draw_mode)
        self.delete_toolbar_button = self._button("Delete", ICONS["trash"], self.delete_selected_annotation, "#dc2626")
        self.delete_toolbar_button.setObjectName("dangerButton")
        self.export_button = self._button("Export", ICONS["export"], self.export_corrected, "#15803d")
        self.export_button.setObjectName("exportButton")
        self.fit_button = self._button("Fit", ICONS["fit"], lambda: self.canvas.fit_to_window())
        toolbar.addWidget(self.run_current_button)
        toolbar.addWidget(self.run_all_button)
        toolbar.addSeparator()
        toolbar.addWidget(self.draw_button)
        toolbar.addWidget(self.delete_toolbar_button)
        toolbar.addSeparator()
        toolbar.addWidget(self.export_button)
        toolbar.addWidget(self.fit_button)

    def _button(self, text, icon_name, callback, icon_color="#334155"):
        button = QPushButton(text)
        button.setIcon(icon(icon_name, icon_color, scale_factor=0.75))
        button.clicked.connect(callback)
        return button

    def _form_label(self, text):
        label = QLabel(text)
        label.setObjectName("formLabel")
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return label

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        splitter.addWidget(self._build_image_panel())
        splitter.addWidget(self._build_center_workspace())

        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(380)
        self.tabs.addTab(self._build_setup_tab(), icon(ICONS["setup"], "#475569", scale_factor=0.8), "Setup")
        self.tabs.addTab(self._build_annotation_tab(), icon(ICONS["annotate"], "#475569", scale_factor=0.8), "Annotation")
        self.tabs.addTab(self._build_results_tab(), icon(ICONS["results"], "#475569", scale_factor=0.8), "Results")
        splitter.addWidget(self.tabs)
        splitter.setSizes([260, 850, 380])

        self.setStatusBar(QStatusBar())
        self.status_context_label = QLabel("No image | 0 objects | saved")
        self.status_context_label.setObjectName("mutedLabel")
        self.statusBar().addPermanentWidget(self.status_context_label)
        self._set_message("Open an image or folder to begin.")

    def _build_image_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(235)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(7, 7, 7, 7)
        layout.setSpacing(4)
        title_row = QHBoxLayout()
        title = QLabel("Dataset")
        title.setObjectName("sectionTitle")
        title_row.addWidget(title)
        title_row.addStretch(1)
        layout.addLayout(title_row)

        self.image_summary_label = QLabel("No project loaded")
        self.image_summary_label.setObjectName("mutedLabel")
        layout.addWidget(self.image_summary_label)

        self.stat_strip = StatStrip()
        layout.addWidget(self.stat_strip)

        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self._on_image_selected)
        layout.addWidget(self.image_list, stretch=1)
        return panel

    def _build_center_workspace(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_hint = QLabel("Open an image or folder to start reviewing annotations.")
        self.canvas_hint.setObjectName("canvasHint")
        layout.addWidget(self.canvas_hint)

        self.workspace_stack = QStackedWidget()
        self.empty_state = EmptyStateWidget()
        self.empty_state.open_image_requested.connect(self.open_image)
        self.empty_state.open_folder_requested.connect(self.open_folder)
        self.canvas = ImageCanvas()
        self.canvas.box_drawn.connect(self._add_manual_box)
        self.canvas.annotation_selected.connect(self._select_annotation)
        self.canvas.annotation_changed.connect(self._annotation_box_changed)
        self.workspace_stack.addWidget(self.empty_state)
        self.workspace_stack.addWidget(self.canvas)
        layout.addWidget(self.workspace_stack, stretch=1)
        return panel

    def _build_setup_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(7, 7, 7, 7)
        layout.setSpacing(5)

        help_card = QGroupBox("Workflow")
        help_layout = QVBoxLayout(help_card)
        help_layout.setContentsMargins(6, 8, 6, 6)
        help = QLabel("Open input -> set SAM3 -> run or draw boxes -> review -> export.")
        help.setObjectName("mutedLabel")
        help.setWordWrap(True)
        help_layout.addWidget(help)
        layout.addWidget(help_card)

        output_group = QGroupBox("Input / Output")
        output_layout = QFormLayout(output_group)
        output_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Auto-generated after opening input")
        browse_output = self._button("Browse", ICONS["folder"], self._browse_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit)
        output_row.addWidget(browse_output)
        output_layout.addRow(self._form_label("Folder"), output_row)
        self.setup_import_yolo_button = self._button("Import YOLO Labels", ICONS["state"], self.import_yolo_labels)
        output_layout.addRow(self.setup_import_yolo_button)
        layout.addWidget(output_group)

        sam_group = QGroupBox("SAM3")
        sam_layout = QFormLayout(sam_group)
        sam_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Select SAM3 model...")
        browse_model = self._button("Browse", ICONS["folder"], self._browse_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_path_edit)
        model_row.addWidget(browse_model)
        sam_layout.addRow(self._form_label("Model"), model_row)

        self.prompts_edit = QTextEdit()
        self.prompts_edit.setPlaceholderText("One class per line, or comma-separated")
        self.prompts_edit.setFixedHeight(76)
        self.prompts_edit.setPlainText("object")
        self.prompts_edit.textChanged.connect(self._refresh_classes)
        sam_layout.addRow(self._form_label("Classes"), self.prompts_edit)

        self.conf_edit = NumericLineEdit(value=0.50, decimals=2, minimum=0.01, maximum=1.0)
        self.conf_edit.setPlaceholderText("0.50")
        sam_layout.addRow(self._form_label("Confidence"), self.conf_edit)

        fp16_box = QWidget()
        fp16_box.setObjectName("fp16Box")
        fp16_layout = QHBoxLayout(fp16_box)
        fp16_layout.setContentsMargins(6, 4, 6, 4)
        self.half_check = QCheckBox("Use fp16 when supported")
        self.half_check.setChecked(True)
        fp16_layout.addWidget(self.half_check)
        sam_layout.addRow(self._form_label("Precision"), fp16_box)

        setup_run = self._button("Run SAM3 Current Image", ICONS["sam3"], self.run_sam3_current, "#2563eb")
        setup_run.setObjectName("primaryButton")
        self.setup_run_button = setup_run
        sam_layout.addRow(setup_run)
        setup_run_all = self._button("Run SAM3 All Remaining", ICONS["sam3"], self.run_sam3_all_remaining, "#2563eb")
        setup_run_all.setObjectName("primaryButton")
        self.setup_run_all_button = setup_run_all
        sam_layout.addRow(setup_run_all)
        layout.addWidget(sam_group)

        batch_group = QGroupBox("Batch Progress")
        batch_layout = QVBoxLayout(batch_group)
        batch_layout.setContentsMargins(6, 8, 6, 6)
        self.batch_status_label = QLabel("Idle")
        self.batch_status_label.setObjectName("mutedLabel")
        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 1)
        self.batch_progress.setValue(0)
        self.cancel_batch_button = self._button("Cancel Batch", ICONS["warning"], self.cancel_batch)
        self.cancel_batch_button.setEnabled(False)
        batch_layout.addWidget(self.batch_status_label)
        batch_layout.addWidget(self.batch_progress)
        batch_layout.addWidget(self.cancel_batch_button)
        layout.addWidget(batch_group)
        layout.addStretch(1)
        return tab

    def _build_annotation_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(7, 7, 7, 7)
        layout.setSpacing(5)

        selected_group = QGroupBox("Selected Annotation")
        selected_layout = QFormLayout(selected_group)
        selected_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.selection_label = QLabel("No annotation selected")
        self.selection_label.setObjectName("mutedLabel")
        selected_layout.addRow(self.selection_label)

        self.class_combo = QComboBox()
        selected_layout.addRow(self._form_label("Class"), self.class_combo)
        self.apply_class_button = self._button("Apply Class", ICONS["reviewed"], self._apply_selected_class)
        selected_layout.addRow(self.apply_class_button)
        self.source_label = QLabel("-")
        self.confidence_label = QLabel("-")
        selected_layout.addRow(self._form_label("Source"), self.source_label)
        selected_layout.addRow(self._form_label("Confidence"), self.confidence_label)

        self.x1_edit = self._coord_edit()
        self.y1_edit = self._coord_edit()
        self.x2_edit = self._coord_edit()
        self.y2_edit = self._coord_edit()
        selected_layout.addRow(self._form_label("x1"), self.x1_edit)
        selected_layout.addRow(self._form_label("y1"), self.y1_edit)
        selected_layout.addRow(self._form_label("x2"), self.x2_edit)
        selected_layout.addRow(self._form_label("y2"), self.y2_edit)

        action_row = QHBoxLayout()
        self.apply_box_button = self._button("Apply Box", ICONS["draw"], self._apply_box_details)
        self.delete_button = self._button("Delete", ICONS["trash"], self.delete_selected_annotation, "#dc2626")
        self.delete_button.setObjectName("dangerButton")
        action_row.addWidget(self.apply_box_button)
        action_row.addWidget(self.delete_button)
        selected_layout.addRow(action_row)

        self.reviewed_button = self._button("Mark Image Reviewed", ICONS["reviewed"], self.mark_current_reviewed)
        selected_layout.addRow(self.reviewed_button)
        layout.addWidget(selected_group)

        table_group = QGroupBox("Current Image Annotations")
        table_layout = QVBoxLayout(table_group)
        self.annotation_table = QTableWidget(0, 5)
        self.annotation_table.setHorizontalHeaderLabels(["Class", "Source", "Conf", "Top-left", "Bottom-right"])
        self.annotation_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.annotation_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.annotation_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.annotation_table.verticalHeader().setVisible(False)
        header = self.annotation_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        self.annotation_table.itemSelectionChanged.connect(self._on_table_selection)
        table_layout.addWidget(self.annotation_table)
        layout.addWidget(table_group, stretch=1)
        return tab

    def _build_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(7, 7, 7, 7)
        layout.setSpacing(5)

        summary_group = QGroupBox("Export Summary")
        summary_layout = QFormLayout(summary_group)
        summary_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.result_status_label = QLabel("No export yet")
        self.result_status_label.setWordWrap(True)
        self.result_output_label = QLabel("-")
        self.result_output_label.setWordWrap(True)
        self.result_csv_label = QLabel("-")
        self.result_csv_label.setWordWrap(True)
        self.result_yolo_label = QLabel("-")
        self.result_yolo_label.setWordWrap(True)
        self.preview_label = QLabel("-")
        self.preview_label.setWordWrap(True)
        summary_layout.addRow(self._form_label("Status"), self.result_status_label)
        summary_layout.addRow(self._form_label("Output"), self.result_output_label)
        summary_layout.addRow(self._form_label("Box CSV"), self.result_csv_label)
        summary_layout.addRow(self._form_label("YOLO labels"), self.result_yolo_label)
        summary_layout.addRow(self._form_label("Preview"), self.preview_label)
        layout.addWidget(summary_group)

        self.preview_thumb = QLabel("Preview thumbnail appears after export or Save Preview.")
        self.preview_thumb.setObjectName("mutedLabel")
        self.preview_thumb.setAlignment(Qt.AlignCenter)
        self.preview_thumb.setMinimumHeight(120)
        self.preview_thumb.setStyleSheet("background:#f8fafc;border:1px solid #cbd5e1;padding:6px;")
        layout.addWidget(self.preview_thumb)

        self.export_results_button = self._button("Export Results", ICONS["export"], self.export_corrected, "#166534")
        self.export_results_button.setObjectName("exportButton")
        self.save_preview_button = self._button("Save Preview", ICONS["preview"], lambda: self.save_current_preview(silent=False))
        self.open_preview_button = self._button("Open Preview", ICONS["preview"], self.open_preview_image)
        self.open_output_button = self._button("Open Output Folder", ICONS["folder"], self.open_output_folder)
        layout.addWidget(self.export_results_button)
        row = QHBoxLayout()
        row.addWidget(self.save_preview_button)
        row.addWidget(self.open_preview_button)
        layout.addLayout(row)
        layout.addWidget(self.open_output_button)
        layout.addStretch(1)
        return tab

    def _coord_edit(self):
        edit = NumericLineEdit(value=0.0, decimals=2, minimum=0.0, maximum=999999.0)
        edit.setPlaceholderText("0.00")
        return edit

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
        )
        if path:
            self._create_project_from_path(path)

    def open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Open Folder")
        if path:
            self._create_project_from_path(path)

    def open_state(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Annotation State",
            "",
            "Annotation State (annotation_state.json);;JSON (*.json)",
        )
        if not path:
            return
        try:
            self.project_state = load_state(path)
            self.current_state_path = Path(path)
            self.output_dir_edit.setText(str(Path(path).parent))
            self.model_path_edit.setText(self.project_state.model_path or "")
            self.prompts_edit.setPlainText("\n".join(self.project_state.prompts))
            self._after_project_loaded()
            self._update_results_panel(status=f"Loaded project state from {path}")
            self._set_message("State loaded. Continue reviewing annotations or export results.")
        except Exception as exc:
            self._show_error("Could not load project state", exc)

    def import_yolo_labels(self):
        if self.project_state is None:
            self._show_error("No project loaded", "Open an image or folder before importing YOLO labels.")
            return
        label_dir = QFileDialog.getExistingDirectory(self, "Select YOLO Detection Label Folder")
        if not label_dir:
            return
        try:
            self._ensure_project_image_sizes()
            summary = import_yolo_detection_labels(self.project_state, label_dir)
            self._refresh_image_list_keep_current()
            image = self.current_image()
            if image is not None:
                self.canvas.set_annotations(image.active_annotations)
                self._refresh_annotation_table()
                self._clear_annotation_details()
            self._mark_dirty()
            message = summary.to_message()
            self._update_results_panel(status=message)
            self._set_message(message)
            QMessageBox.information(self, "YOLO Import Complete", message)
        except Exception as exc:
            self._show_error("Could not import YOLO labels", exc)

    def _ensure_project_image_sizes(self):
        for image in self.project_state.images:
            if image.width is not None and image.height is not None:
                continue
            qimage = QImage(str(image.image_path))
            if qimage.isNull():
                raise ValueError(f"Could not load image dimensions: {image.image_path}")
            image.width = qimage.width()
            image.height = qimage.height()

    def _create_project_from_path(self, path):
        try:
            prompts = parse_prompts(self.prompts_edit.toPlainText())
            self.project_state = create_project(
                input_path=path,
                prompts=prompts,
                model_path=self.model_path_edit.text().strip() or None,
            )
            self.output_dir_edit.setText(str(default_output_dir(self.project_state)))
            self.current_state_path = None
            self.last_export_result = None
            self.last_preview_path = None
            self._after_project_loaded()
            self._update_results_panel(status="Project opened. Save or export when ready.")
            self._set_message("Set model/classes, then run SAM3 or draw boxes manually.")
        except Exception as exc:
            self._show_error("Could not open input", exc)

    def _after_project_loaded(self):
        self.workspace_stack.setCurrentWidget(self.canvas)
        self._set_project_enabled(True)
        self.unsaved = False
        self._refresh_classes()
        self._refresh_image_list()
        if self.project_state.images:
            self.image_list.setCurrentRow(0)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select SAM3 Model", "", "Model Files (*.pt);;All Files (*)")
        if path:
            self.model_path_edit.setText(path)
            if self.project_state:
                self.project_state.model_path = path
                self._mark_dirty()

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_dir_edit.setText(path)

    def _refresh_classes(self):
        prompts = parse_prompts(self.prompts_edit.toPlainText())
        current_text = self.class_combo.currentText() if hasattr(self, "class_combo") else ""
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItems(prompts or ["object"])
        if current_text:
            index = self.class_combo.findText(current_text)
            if index >= 0:
                self.class_combo.setCurrentIndex(index)
        self.class_combo.blockSignals(False)
        if self.project_state and prompts:
            self.project_state.prompts = prompts

    def _refresh_image_list(self):
        self.image_list.blockSignals(True)
        self.image_list.clear()
        if self.project_state:
            total = len(self.project_state.images)
            reviewed = sum(1 for image in self.project_state.images if image.status == ImageStatus.REVIEWED)
            edited = sum(1 for image in self.project_state.images if image.status == ImageStatus.EDITED)
            pending = sum(1 for image in self.project_state.images if image.status == ImageStatus.NOT_PREDICTED)
            self.stat_strip.update_counts(total, reviewed, edited, pending)
            self.image_summary_label.setText(self.project_state.project_name or "Current project")
            self.project_subtitle.setText(self.project_state.project_name or "Current project")
            for image in self.project_state.images:
                item = QListWidgetItem(self._image_item_text(image))
                item.setData(Qt.UserRole, image.image_index)
                item.setToolTip(f"{image.image_path}\nStatus: {STATUS_LABELS.get(image.status)}")
                item.setBackground(QColor(STATUS_BACKGROUNDS.get(image.status, "#ffffff")))
                item.setForeground(QColor(STATUS_COLORS.get(image.status, "#111827")))
                self.image_list.addItem(item)
        else:
            self.image_summary_label.setText("No project yet")
            self.project_subtitle.setText("No project loaded")
            self.stat_strip.update_counts(0, 0, 0, 0)
        self.image_list.blockSignals(False)

    def _image_item_text(self, image):
        status = STATUS_LABELS.get(image.status, str(image.status))
        count = len(image.active_annotations)
        return f"{image.image_index + 1:03d}  {image.image_name}\n{status}    {count} objects"

    def _on_image_selected(self, current, _previous):
        if current is None or self.project_state is None:
            return
        self.current_image_index = current.data(Qt.UserRole)
        self._load_current_image()

    def _load_current_image(self):
        image = self.current_image()
        if image is None:
            return
        try:
            width, height = self.canvas.load_image(image.image_path)
            if image.width is None or image.height is None:
                image.width = width
                image.height = height
            self.canvas.set_annotations(image.active_annotations)
            self._refresh_annotation_table()
            self._clear_annotation_details()
            self.canvas_hint.setText(
                f"{image.image_name} | {image.width}x{image.height} | "
                f"{len(image.active_annotations)} active box(es)"
            )
            self._update_status_context()
            self._set_message("Run SAM3, draw a box, or select an existing box to edit.")
        except Exception as exc:
            image.mark_error(exc)
            self._refresh_image_list()
            self._show_error("Could not load image", exc)

    def current_image(self):
        if self.project_state is None or self.current_image_index is None:
            return None
        return self.project_state.get_image(self.current_image_index)

    def selected_annotation(self):
        image = self.current_image()
        selected_id = self.canvas.selected_annotation_id()
        if image is None or selected_id is None:
            return None
        for annotation in image.annotations:
            if annotation.id == selected_id:
                return annotation
        return None

    def _toggle_draw_mode(self, checked):
        self.canvas.set_draw_mode(checked)
        self.canvas_hint.setText(
            "Draw Box mode ON. Drag on the image to create a manual box."
            if checked
            else "Draw Box mode OFF. Select and drag boxes to move them."
        )
        self._set_message("Draw mode enabled." if checked else "Draw mode disabled.")
        self._update_status_context()

    def _add_manual_box(self, box_xyxy):
        image = self.current_image()
        if image is None:
            return
        prompts = parse_prompts(self.prompts_edit.toPlainText()) or ["object"]
        class_id = max(0, self.class_combo.currentIndex())
        class_name = prompts[class_id] if class_id < len(prompts) else prompts[0]
        try:
            annotation = image.add_manual_annotation(class_id, class_name, box_xyxy)
            self.canvas.set_annotations(image.active_annotations)
            self.canvas.select_annotation(annotation.id)
            self._refresh_annotation_table()
            self._refresh_image_list_keep_current()
            self._mark_dirty()
            self._set_message("Manual box added. Adjust it on the canvas or in the inspector.")
        except Exception as exc:
            self._show_error("Could not add annotation", exc)

    def _annotation_box_changed(self, annotation_id, box_xyxy):
        image = self.current_image()
        if image is None:
            return
        for annotation in image.annotations:
            if annotation.id == annotation_id:
                if all(abs(old - new) < 0.5 for old, new in zip(annotation.box_xyxy, box_xyxy)):
                    return
                try:
                    annotation.edit_box(box_xyxy, image.width, image.height)
                    if image.status != ImageStatus.REVIEWED:
                        image.status = ImageStatus.EDITED
                    self.canvas.update_annotation_box(annotation)
                    self._refresh_annotation_table()
                    self._refresh_image_list_keep_current()
                    self._show_annotation_details(annotation)
                    self._mark_dirty()
                except Exception as exc:
                    self._show_error("Could not update annotation", exc)
                return

    def _select_annotation(self, annotation_id):
        image = self.current_image()
        if image is None:
            self._clear_annotation_details()
            return
        annotation = next((item for item in image.active_annotations if item.id == annotation_id), None)
        if annotation is None:
            self._clear_annotation_details()
            return
        self._show_annotation_details(annotation)
        for row in range(self.annotation_table.rowCount()):
            if self.annotation_table.item(row, 0).data(Qt.UserRole) == annotation.id:
                self.annotation_table.blockSignals(True)
                self.annotation_table.selectRow(row)
                self.annotation_table.blockSignals(False)
                break

    def _show_annotation_details(self, annotation):
        self._updating_details = True
        self._set_annotation_detail_enabled(True)
        x1, y1, x2, y2 = annotation.box_xyxy
        self.selection_label.setText("Editing selected box")
        self.source_label.setText(annotation.source.value)
        self.confidence_label.setText("-" if annotation.confidence is None else f"{annotation.confidence:.4f}")
        self.x1_edit.set_value(x1)
        self.y1_edit.set_value(y1)
        self.x2_edit.set_value(x2)
        self.y2_edit.set_value(y2)
        class_index = self.class_combo.findText(annotation.class_name)
        if class_index >= 0:
            self.class_combo.setCurrentIndex(class_index)
        self._updating_details = False

    def _clear_annotation_details(self):
        self.selection_label.setText("No annotation selected")
        self.source_label.setText("-")
        self.confidence_label.setText("-")
        for edit in (self.x1_edit, self.y1_edit, self.x2_edit, self.y2_edit):
            edit.set_value(0)
        self._set_annotation_detail_enabled(False)

    def _set_annotation_detail_enabled(self, enabled):
        for widget in (
            self.x1_edit,
            self.y1_edit,
            self.x2_edit,
            self.y2_edit,
            self.apply_box_button,
            self.apply_class_button,
            self.delete_button,
        ):
            widget.setEnabled(enabled)
        self.class_combo.setEnabled(self.project_state is not None)
        self.delete_toolbar_button.setEnabled(enabled)

    def _refresh_annotation_table(self):
        image = self.current_image()
        self.annotation_table.setRowCount(0)
        if image is None:
            return
        for annotation in image.active_annotations:
            row = self.annotation_table.rowCount()
            self.annotation_table.insertRow(row)
            class_item = QTableWidgetItem(annotation.class_name)
            class_item.setData(Qt.UserRole, annotation.id)
            self.annotation_table.setItem(row, 0, class_item)
            self.annotation_table.setItem(row, 1, QTableWidgetItem(annotation.source.value))
            conf = "-" if annotation.confidence is None else f"{annotation.confidence:.3f}"
            self.annotation_table.setItem(row, 2, QTableWidgetItem(conf))
            x1, y1, x2, y2 = annotation.box_xyxy
            self.annotation_table.setItem(row, 3, QTableWidgetItem(f"{x1:.1f}, {y1:.1f}"))
            self.annotation_table.setItem(row, 4, QTableWidgetItem(f"{x2:.1f}, {y2:.1f}"))

    def _on_table_selection(self):
        selected = self.annotation_table.selectedItems()
        if not selected:
            return
        annotation_id = self.annotation_table.item(selected[0].row(), 0).data(Qt.UserRole)
        self.canvas.select_annotation(annotation_id)
        self.tabs.setCurrentIndex(1)

    def _apply_selected_class(self):
        annotation = self.selected_annotation()
        image = self.current_image()
        if annotation is None or image is None:
            return
        class_id = max(0, self.class_combo.currentIndex())
        class_name = self.class_combo.currentText() or "object"
        annotation.change_class(class_id, class_name)
        if image.status != ImageStatus.REVIEWED:
            image.status = ImageStatus.EDITED
        self.canvas.set_annotations(image.active_annotations)
        self.canvas.select_annotation(annotation.id)
        self._refresh_annotation_table()
        self._refresh_image_list_keep_current()
        self._mark_dirty()

    def _apply_box_details(self):
        if self._updating_details:
            return
        annotation = self.selected_annotation()
        image = self.current_image()
        if annotation is None or image is None:
            return
        try:
            annotation.edit_box(
                (
                    self.x1_edit.value(),
                    self.y1_edit.value(),
                    self.x2_edit.value(),
                    self.y2_edit.value(),
                ),
                image.width,
                image.height,
            )
            if image.status != ImageStatus.REVIEWED:
                image.status = ImageStatus.EDITED
            self.canvas.update_annotation_box(annotation)
            self._refresh_annotation_table()
            self._refresh_image_list_keep_current()
            self._mark_dirty()
            self._set_message("Box coordinates updated.")
        except Exception as exc:
            self._show_error("Invalid box coordinates", exc)

    def delete_selected_annotation(self):
        annotation = self.selected_annotation()
        image = self.current_image()
        if annotation is None or image is None:
            self._set_message("Select a box before deleting.")
            return
        annotation.mark_deleted()
        if image.status != ImageStatus.REVIEWED:
            image.status = ImageStatus.EDITED
        self.canvas.remove_annotation(annotation.id)
        self._refresh_annotation_table()
        self._refresh_image_list_keep_current()
        self._clear_annotation_details()
        self._mark_dirty()
        self._set_message("Box deleted. Deleted boxes are not exported.")

    def mark_current_reviewed(self):
        image = self.current_image()
        if image is None:
            return
        image.mark_reviewed()
        self._refresh_image_list_keep_current()
        self._mark_dirty()
        self._set_message(f"Marked reviewed: {image.image_name}")

    def run_sam3_current(self):
        image = self.current_image()
        if image is None or self.project_state is None:
            self._show_error("No image selected", "Open an image or folder, then select an image first.")
            return
        prompts = parse_prompts(self.prompts_edit.toPlainText())
        if not prompts:
            self._show_error("Missing classes", "Enter at least one class prompt, such as 'car' or 'person'.")
            self.tabs.setCurrentIndex(0)
            return
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            self._show_error("Missing model path", "Select a local SAM3 model file before running prediction.")
            self.tabs.setCurrentIndex(0)
            return
        try:
            validate_model_path(model_path)
        except Exception as exc:
            self._show_error("Invalid model path", exc)
            self.tabs.setCurrentIndex(0)
            return

        if image.annotations:
            answer = QMessageBox.question(
                self,
                "Replace Current Annotations?",
                "Running SAM3 on this image will replace its current draft/manual annotations.",
            )
            if answer != QMessageBox.Yes:
                return

        self.project_state.prompts = prompts
        self.project_state.model_path = model_path
        self._set_busy(True)
        cached = self.predictor_cache.has_predictor(model_path, self.conf_edit.value(), self.half_check.isChecked())
        initial_status = (
            f"Running SAM3 on {image.image_name} with cached model..."
            if cached
            else f"Loading model, then running SAM3 on {image.image_name}..."
        )
        self.result_status_label.setText(initial_status)
        self._set_message(initial_status)
        self.prediction_worker = PredictionWorker(
            image_index=image.image_index,
            image_path=image.image_path,
            model_path=model_path,
            prompts=prompts,
            conf=self.conf_edit.value(),
            half=self.half_check.isChecked(),
            predictor_cache=self.predictor_cache,
            parent=self,
        )
        self.prediction_worker.status.connect(self._prediction_status)
        self.prediction_worker.finished_prediction.connect(self._prediction_finished)
        self.prediction_worker.failed.connect(self._prediction_failed)
        self.prediction_worker.finished.connect(lambda: self._set_busy(False))
        self.prediction_worker.start()

    def _prediction_status(self, message):
        self.result_status_label.setText(message)
        self._set_message(message)

    def _prediction_finished(self, image_index, annotations, width, height):
        image = self.project_state.get_image(image_index)
        if width is not None and height is not None:
            image.width = width
            image.height = height
        image.replace_sam3_drafts(annotations)
        if self.current_image_index == image_index:
            self.canvas.set_annotations(image.active_annotations)
            self._refresh_annotation_table()
            self._clear_annotation_details()
        self._refresh_image_list_keep_current()
        self._mark_dirty()
        self._update_results_panel(status=f"SAM3 finished for {image.image_name}: {len(annotations)} box(es).")
        self._set_message("Review/edit boxes, then save or export.")

    def _prediction_failed(self, image_index, message):
        image = self.project_state.get_image(image_index)
        image.mark_error(message)
        self._refresh_image_list_keep_current()
        self._mark_dirty()
        self._update_results_panel(status=f"SAM3 failed for {image.image_name}: {message}")
        self._show_error("SAM3 prediction failed", message)

    def run_sam3_all_remaining(self):
        if self.project_state is None:
            self._show_error("No project loaded", "Open an image or folder before running batch prediction.")
            return
        prompts = parse_prompts(self.prompts_edit.toPlainText())
        if not prompts:
            self._show_error("Missing classes", "Enter at least one class prompt before running prediction.")
            self.tabs.setCurrentIndex(0)
            return
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            self._show_error("Missing model path", "Select a local SAM3 model file before running prediction.")
            self.tabs.setCurrentIndex(0)
            return
        try:
            validate_model_path(model_path)
        except Exception as exc:
            self._show_error("Invalid model path", exc)
            self.tabs.setCurrentIndex(0)
            return

        targets = remaining_prediction_targets(self.project_state)
        if not targets:
            self._set_message("No remaining images to predict. Edited/reviewed/predicted images are skipped.")
            self.batch_status_label.setText("No remaining images to predict.")
            return

        risky_targets = [image for image in targets if image.annotations]
        skipped_count = len(self.project_state.images) - len(targets)
        if risky_targets:
            answer = QMessageBox.question(
                self,
                "Replace Target Annotations?",
                f"{len(risky_targets)} remaining target image(s) already have annotations. "
                "Batch prediction will replace annotations only on target images. "
                f"{skipped_count} predicted/edited/reviewed image(s) will be skipped.\n\n"
                "Continue?",
            )
            if answer != QMessageBox.Yes:
                return
        elif skipped_count:
            answer = QMessageBox.question(
                self,
                "Run Remaining Images?",
                f"SAM3 will run on {len(targets)} not-predicted/error image(s). "
                f"{skipped_count} predicted/edited/reviewed image(s) will be skipped.\n\n"
                "Continue?",
            )
            if answer != QMessageBox.Yes:
                return

        self.project_state.prompts = prompts
        self.project_state.model_path = model_path
        self.batch_progress.setRange(0, len(targets))
        self.batch_progress.setValue(0)
        self.batch_status_label.setText(f"Starting batch: {len(targets)} image(s)")
        self._set_batch_busy(True)
        cached = self.predictor_cache.has_predictor(model_path, self.conf_edit.value(), self.half_check.isChecked())
        self._set_message(
            "Running SAM3 batch with cached model..."
            if cached
            else "Loading model for SAM3 batch..."
        )
        self.batch_worker = BatchPredictionWorker(
            image_records=targets,
            model_path=model_path,
            prompts=prompts,
            conf=self.conf_edit.value(),
            half=self.half_check.isChecked(),
            predictor_cache=self.predictor_cache,
            parent=self,
        )
        self.batch_worker.status.connect(self._batch_status)
        self.batch_worker.progress.connect(self._batch_progress)
        self.batch_worker.image_finished.connect(self._batch_image_finished)
        self.batch_worker.image_failed.connect(self._batch_image_failed)
        self.batch_worker.failed.connect(self._batch_failed)
        self.batch_worker.completed.connect(self._batch_completed)
        self.batch_worker.cancelled.connect(self._batch_cancelled)
        self.batch_worker.finished.connect(lambda: self._set_batch_busy(False))
        self.batch_worker.start()

    def cancel_batch(self):
        if self.batch_worker is not None and self.batch_worker.isRunning():
            self.batch_worker.request_cancel()
            self.cancel_batch_button.setEnabled(False)
            self.batch_status_label.setText("Cancel requested. Waiting for current image to finish...")
            self._set_message("Cancel requested. Batch will stop after the current image.")

    def _batch_status(self, message):
        self.batch_status_label.setText(message)
        self.result_status_label.setText(message)
        self._set_message(message)

    def _batch_progress(self, current, total, image_path):
        name = Path(image_path).name
        self.batch_progress.setRange(0, total)
        self.batch_progress.setValue(current - 1)
        message = f"Running SAM3 {current}/{total}: {name}"
        self.batch_status_label.setText(message)
        self._set_message(message)

    def _batch_image_finished(self, image_index, annotations, width, height):
        image = self.project_state.get_image(image_index)
        if width is not None and height is not None:
            image.width = width
            image.height = height
        image.replace_sam3_drafts(annotations)
        if self.current_image_index == image_index:
            self.canvas.set_annotations(image.active_annotations)
            self._refresh_annotation_table()
            self._clear_annotation_details()
        self._refresh_image_list_keep_current()
        self._mark_dirty()
        self.batch_progress.setValue(min(self.batch_progress.value() + 1, self.batch_progress.maximum()))

    def _batch_image_failed(self, image_index, message):
        image = self.project_state.get_image(image_index)
        image.mark_error(message)
        self._refresh_image_list_keep_current()
        self._mark_dirty()
        self.batch_progress.setValue(min(self.batch_progress.value() + 1, self.batch_progress.maximum()))

    def _batch_failed(self, message):
        self.batch_status_label.setText(f"Batch failed before prediction: {message}")
        self._update_results_panel(status=f"SAM3 batch failed: {message}")
        self._show_error("SAM3 batch failed", message)

    def _batch_completed(self, summary):
        self.batch_progress.setValue(self.batch_progress.maximum())
        message = (
            f"Batch complete: {summary['processed']} processed, "
            f"{summary['predicted']} with detections, "
            f"{summary['no_detection']} no detections, "
            f"{summary['errors']} errors."
        )
        self.batch_status_label.setText(message)
        self._update_results_panel(status=message)
        self._set_message(message + " Save the project to persist batch results.")

    def _batch_cancelled(self, summary):
        message = (
            f"Batch cancelled: {summary['processed']} processed, "
            f"{summary['predicted']} with detections, "
            f"{summary['no_detection']} no detections, "
            f"{summary['errors']} errors."
        )
        self.batch_status_label.setText(message)
        self._update_results_panel(status=message)
        self._set_message(message + " Partial results remain in the project.")

    def save_project(self):
        if self.project_state is None:
            return
        self._sync_project_settings()
        output_dir = self._output_dir_or_default()
        try:
            path = save_state_to_output(self.project_state, output_dir)
            self.current_state_path = path
            self.unsaved = False
            self._update_status_context()
            self._update_results_panel(status="Project state saved.", output_dir=output_dir)
            self._set_message(f"Saved annotation_state.json to {path}")
        except Exception as exc:
            self._show_error("Could not save project", exc)

    def export_corrected(self):
        if self.project_state is None:
            return
        self._sync_project_settings()
        unpredicted = self.project_state.unpredicted_images
        if unpredicted:
            answer = QMessageBox.warning(
                self,
                "Unpredicted Images",
                f"{len(unpredicted)} image(s) have not been predicted or reviewed. "
                "Export will still create empty label files where there are no active boxes.\n\n"
                "Export anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        output_dir = self._output_dir_or_default()
        try:
            save_state_to_output(self.project_state, output_dir)
            result = export_project(self.project_state, output_dir)
            self.last_export_result = result
            preview_path = self.save_current_preview(silent=True)
            self.unsaved = False
            self._update_status_context()
            self._update_results_panel(
                status=f"Export complete: {len(result['rows'])} active box(es).",
                output_dir=output_dir,
                box_csv=result["box_csv"],
                yolo_detection_dir=result["yolo_detection_dir"],
                preview_path=preview_path,
            )
            self.tabs.setCurrentIndex(2)
            self._set_message(f"Exported corrected labels to {output_dir}")
            QMessageBox.information(self, "Export Complete", f"Corrected labels exported to:\n{output_dir}")
        except Exception as exc:
            self._show_error("Could not export corrected labels", exc)

    def save_current_preview(self, silent=False):
        image = self.current_image()
        if image is None:
            if not silent:
                self._show_error("No image selected", "Select an image before saving a preview.")
            return None
        output_dir = self._output_dir_or_default()
        preview_dir = output_dir / "preview_results"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"{Path(image.image_path).stem}_reviewed.png"
        qimage = QImage(str(image.image_path))
        if qimage.isNull():
            if not silent:
                self._show_error("Could not create preview", f"Could not load image: {image.image_path}")
            return None

        painter = QPainter(qimage)
        painter.setRenderHint(QPainter.Antialiasing)
        for annotation in image.active_annotations:
            color = BOX_COLORS[annotation.class_id % len(BOX_COLORS)]
            pen = QPen(color, max(2, int(qimage.width() / 640)))
            painter.setPen(pen)
            x1, y1, x2, y2 = annotation.box_xyxy
            painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            painter.drawText(int(x1) + 4, max(14, int(y1) - 4), annotation.class_name)
        painter.end()
        qimage.save(str(preview_path))
        self.last_preview_path = preview_path
        self._update_results_panel(preview_path=preview_path)
        if not silent:
            self.tabs.setCurrentIndex(2)
            self._set_message(f"Saved preview image: {preview_path}")
        return preview_path

    def open_preview_image(self):
        if not self.last_preview_path:
            self._set_message("No preview image has been saved yet.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.last_preview_path).resolve())))

    def open_output_folder(self):
        output_dir = self._output_dir_or_default() if self.project_state else self.output_dir_edit.text().strip()
        if not output_dir:
            self._show_error("No output folder", "Open a project or choose an output folder first.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(output_dir).resolve())))

    def _sync_project_settings(self):
        if self.project_state is None:
            return
        prompts = parse_prompts(self.prompts_edit.toPlainText())
        if prompts:
            self.project_state.prompts = prompts
        model_path = self.model_path_edit.text().strip()
        if model_path:
            self.project_state.model_path = model_path

    def _output_dir_or_default(self):
        output = self.output_dir_edit.text().strip()
        if output:
            return Path(output)
        output_dir = default_output_dir(self.project_state)
        self.output_dir_edit.setText(str(output_dir))
        return output_dir

    def _update_results_panel(self, status=None, output_dir=None, box_csv=None, yolo_detection_dir=None, preview_path=None):
        if status is not None:
            self.result_status_label.setText(status)
        if output_dir is None and self.project_state is not None:
            output_dir = self.output_dir_edit.text().strip() or default_output_dir(self.project_state)
        if output_dir is not None:
            self.result_output_label.setText(str(output_dir))
        if box_csv is not None:
            self.result_csv_label.setText(str(box_csv))
        elif self.last_export_result:
            self.result_csv_label.setText(str(self.last_export_result["box_csv"]))
        if yolo_detection_dir is not None:
            self.result_yolo_label.setText(str(yolo_detection_dir))
        elif self.last_export_result:
            self.result_yolo_label.setText(str(self.last_export_result["yolo_detection_dir"]))
        if preview_path is not None:
            self.preview_label.setText(str(preview_path))
            self._set_preview_thumbnail(preview_path)
        elif self.last_preview_path:
            self.preview_label.setText(str(self.last_preview_path))
            self._set_preview_thumbnail(self.last_preview_path)

    def _set_preview_thumbnail(self, preview_path):
        image = QImage(str(preview_path))
        if image.isNull():
            return
        pixmap = image.scaled(300, 170, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_thumb.setPixmap(pixmap)

    def _refresh_image_list_keep_current(self):
        current_index = self.current_image_index
        self._refresh_image_list()
        if current_index is not None:
            for row in range(self.image_list.count()):
                if self.image_list.item(row).data(Qt.UserRole) == current_index:
                    self.image_list.setCurrentRow(row)
                    break

    def _set_project_enabled(self, enabled):
        for widget in (
            self.save_button,
            self.import_yolo_button,
            self.setup_import_yolo_button,
            self.run_current_button,
            self.run_all_button,
            self.setup_run_button,
            self.setup_run_all_button,
            self.export_button,
            self.export_results_button,
            self.draw_button,
            self.fit_button,
            self.annotation_table,
            self.reviewed_button,
            self.save_preview_button,
            self.open_output_button,
            self.open_preview_button,
            self.delete_toolbar_button,
        ):
            widget.setEnabled(enabled)
        self.cancel_batch_button.setEnabled(False)
        self.class_combo.setEnabled(enabled)

    def _set_busy(self, busy):
        for widget in (
            self.run_current_button,
            self.run_all_button,
            self.setup_run_button,
            self.setup_run_all_button,
            self.open_image_button,
            self.open_folder_button,
            self.open_state_button,
            self.import_yolo_button,
            self.save_button,
            self.export_button,
            self.export_results_button,
            self.draw_button,
            self.delete_toolbar_button,
        ):
            widget.setEnabled(not busy and (self.project_state is not None or widget in (self.open_image_button, self.open_folder_button, self.open_state_button)))

    def _set_batch_busy(self, busy):
        self._set_busy(busy)
        for widget in (
            self.model_path_edit,
            self.prompts_edit,
            self.conf_edit,
            self.half_check,
            self.output_dir_edit,
            self.setup_import_yolo_button,
        ):
            widget.setEnabled(not busy)
        self.cancel_batch_button.setEnabled(busy)
        if not busy and self.project_state is not None:
            self._set_project_enabled(True)
            self._set_annotation_detail_enabled(self.selected_annotation() is not None)

    def _mark_dirty(self):
        self.unsaved = True
        self._update_status_context()

    def _update_status_context(self):
        image = self.current_image()
        if image is None:
            image_text = "No image"
            count = 0
        else:
            size = f"{image.width}x{image.height}" if image.width and image.height else "size unknown"
            image_text = f"{image.image_name} ({size})"
            count = len(image.active_annotations)
        state = "unsaved" if self.unsaved else "saved"
        self.status_context_label.setText(f"{image_text} | {count} objects | {state}")

    def _set_message(self, message):
        self.statusBar().showMessage(message)

    def _show_error(self, title, error):
        message = str(error)
        self._set_message(f"{title}: {message}")
        QMessageBox.critical(self, title, message)

    def closeEvent(self, event):
        if self.unsaved:
            answer = QMessageBox.question(
                self,
                "Unsaved Changes",
                "Project has unsaved changes. Close anyway?",
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        event.accept()
