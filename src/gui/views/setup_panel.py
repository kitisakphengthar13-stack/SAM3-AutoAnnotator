from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from gui.icons import icon
from gui.widgets.class_prompt_editor import ClassPromptEditor
from gui.widgets.path_display import PathDisplay


class SetupPanel(QWidget):
    browse_model_requested = Signal()
    browse_output_requested = Signal()
    settings_changed = Signal()
    apply_requested = Signal()
    cancel_requested = Signal()

    def __init__(self, actions, parent=None):
        super().__init__(parent)
        self.setObjectName("setupPanel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        heading = QWidget()
        head = QVBoxLayout(heading)
        head.setContentsMargins(24, 20, 24, 12)
        eyebrow = QLabel("PROJECT SETUP")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("Tell SAM3 what to find")
        title.setObjectName("dialogTitle")
        head.addWidget(eyebrow)
        head.addWidget(title)
        outer.addWidget(heading)
        self.tabs = QTabWidget()
        self.tabs.setAccessibleName("Setup sections")
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 16, 24, 20)
        layout.setSpacing(10)
        layout.addWidget(_label("SAM3 CHECKPOINT"))
        row = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText(
            "Choose a local SAM3 .pt file (optional for manual annotation)"
        )
        self.model_path_edit.setAccessibleName("SAM3 model checkpoint path")
        row.addWidget(self.model_path_edit, 1)
        self.browse_model_button = QPushButton(icon("folder"), "Browse")
        self.browse_model_button.clicked.connect(
            lambda: self.browse_model_requested.emit()
        )
        row.addWidget(self.browse_model_button)
        layout.addLayout(row)
        self.model_validation_label = _error()
        layout.addWidget(self.model_validation_label)
        layout.addWidget(_label("CLASSES / TEXT PROMPTS"))
        self.prompts_edit = ClassPromptEditor(visible_rows=5)
        self.prompts_edit.setPlaceholderText("One class per line, or comma-separated")
        self.prompts_edit.setAccessibleName("SAM3 class prompts")
        layout.addWidget(self.prompts_edit)
        hint = QLabel(
            "Use clear object names, for example: car, person, traffic light.\nTheir order determines the YOLO class IDs."
        )
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.prompt_validation_label = _error()
        layout.addWidget(self.prompt_validation_label)
        row = QHBoxLayout()
        row.addWidget(_label("Confidence"))
        self.conf_edit = QDoubleSpinBox()
        self.conf_edit.setAccessibleName("Minimum prediction confidence")
        self.conf_edit.setRange(0.01, 1.0)
        self.conf_edit.setDecimals(2)
        self.conf_edit.setSingleStep(0.05)
        self.conf_edit.setValue(0.5)
        self.conf_edit.setKeyboardTracking(False)
        self.conf_edit.setFixedWidth(100)
        self.conf_edit.setToolTip(
            "Keep predictions at or above this confidence. Lower values return more candidates."
        )
        row.addWidget(self.conf_edit)
        row.addStretch()
        self.half_check = QCheckBox("Use FP16")
        self.half_check.setChecked(True)
        self.half_check.setToolTip("Uses less GPU memory on compatible CUDA devices.")
        row.addWidget(self.half_check)
        layout.addLayout(row)
        layout.addStretch()
        self.scroll_area.setWidget(content)
        self.tabs.addTab(self.scroll_area, "Annotation")

        files_scroll = QScrollArea()
        files_scroll.setWidgetResizable(True)
        files = QWidget()
        form = QVBoxLayout(files)
        form.setContentsMargins(24, 20, 24, 20)
        form.setSpacing(12)
        form.addWidget(_label("SOURCE IMAGES"))
        self.input_path_display = PathDisplay()
        form.addWidget(self.input_path_display)
        form.addWidget(_label("PROJECT & EXPORT FOLDER"))
        row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Created automatically from the input")
        self.output_dir_edit.setAccessibleName("Project and export output folder")
        self.browse_output_button = QPushButton(icon("folder"), "Browse")
        self.browse_output_button.clicked.connect(
            lambda: self.browse_output_requested.emit()
        )
        row.addWidget(self.output_dir_edit, 1)
        row.addWidget(self.browse_output_button)
        form.addLayout(row)
        note = QLabel(
            "Save Project keeps your editable annotations here. Export writes CSV and YOLO labels to the same folder.\n\nSource images stay in their original location."
        )
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        form.addWidget(note)
        form.addStretch()
        files_scroll.setWidget(files)
        self.tabs.addTab(files_scroll, "Files and output")
        outer.addWidget(self.tabs, 1)
        footer = QWidget()
        footer.setObjectName("panelActionFooter")
        row = QHBoxLayout(footer)
        row.setContentsMargins(24, 12, 24, 16)
        row.addStretch()
        self.cancel_button = QPushButton("Cancel")
        self.apply_button = QPushButton("Apply")
        self.apply_button.setObjectName("primaryButton")
        self.apply_button.setDefault(True)
        self.cancel_button.clicked.connect(lambda: self.cancel_requested.emit())
        self.apply_button.clicked.connect(lambda: self.apply_requested.emit())
        row.addWidget(self.cancel_button)
        row.addWidget(self.apply_button)
        outer.addWidget(footer)
        self.model_path_edit.textChanged.connect(self.model_path_edit.setToolTip)
        self.output_dir_edit.textChanged.connect(self.output_dir_edit.setToolTip)

    def set_project(self, project, output_dir):
        self.input_path_display.set_path(project.input_path)
        self.output_dir_edit.setText(str(output_dir))
        self.model_path_edit.setText(
            "" if not project.model_path else str(project.model_path)
        )
        self.prompts_edit.setPlainText("\n".join(project.prompts))
        self.conf_edit.setValue(project.confidence)
        self.half_check.setChecked(project.half)

    def snapshot(self):
        return {
            "output_dir": self.output_dir_edit.text(),
            "model_path": self.model_path_edit.text(),
            "prompts": self.prompts_edit.toPlainText(),
            "confidence": self.conf_edit.value(),
            "half": self.half_check.isChecked(),
        }

    def restore_snapshot(self, snapshot):
        if not snapshot:
            return
        self.output_dir_edit.setText(snapshot["output_dir"])
        self.model_path_edit.setText(snapshot["model_path"])
        self.prompts_edit.setPlainText(snapshot["prompts"])
        self.conf_edit.setValue(snapshot["confidence"])
        self.half_check.setChecked(snapshot["half"])
        self.set_prompt_error(None)
        self.set_model_error(None)

    def prompts_text(self):
        return self.prompts_edit.toPlainText()

    def set_prompts(self, prompts):
        self.prompts_edit.setPlainText("\n".join(prompts))

    def set_settings_enabled(self, enabled, *, project_open=True):
        for widget in (
            self.model_path_edit,
            self.browse_model_button,
            self.prompts_edit,
            self.conf_edit,
            self.half_check,
            self.apply_button,
        ):
            widget.setEnabled(enabled)
        self.output_dir_edit.setEnabled(enabled and project_open)
        self.browse_output_button.setEnabled(enabled and project_open)

    def set_prompt_error(self, message=None):
        self.prompt_validation_label.setText("" if message is None else str(message))
        self.prompt_validation_label.setVisible(bool(message))

    def set_model_error(self, message=None):
        self.model_validation_label.setText("" if message is None else str(message))
        self.model_validation_label.setVisible(bool(message))


def _label(text):
    label = QLabel(text)
    label.setObjectName("mutedLabel")
    return label


def _error():
    label = QLabel()
    label.setObjectName("validationError")
    label.setWordWrap(True)
    label.hide()
    return label
