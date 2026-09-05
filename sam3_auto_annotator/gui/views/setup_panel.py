from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from sam3_auto_annotator.gui.icons import ICONS, icon
from sam3_auto_annotator.gui.widgets.class_prompt_editor import ClassPromptEditor
from sam3_auto_annotator.gui.widgets.path_display import PathDisplay


class SetupPanel(QWidget):
    browse_model_requested = Signal()
    browse_output_requested = Signal()
    settings_changed = Signal()
    apply_requested = Signal()
    cancel_requested = Signal()

    def __init__(self, actions, parent=None):
        super().__init__(parent)
        del actions
        self.setObjectName("setupPanel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("setupScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        guidance = QLabel(
            "Changes are staged here. Apply commits them to the project; Cancel discards them."
        )
        guidance.setObjectName("mutedLabel")
        guidance.setWordWrap(True)
        layout.addWidget(guidance)

        output_group = QGroupBox("Project")
        output_layout = _form_layout(output_group)
        self.input_path_display = PathDisplay()
        output_layout.addRow("Input", self.input_path_display)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Created automatically from the input")
        self.browse_output_button = QPushButton(icon(ICONS["folder"]), "Browse")
        self.browse_output_button.clicked.connect(
            lambda _checked=False: self.browse_output_requested.emit()
        )
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit, 1)
        output_row.addWidget(self.browse_output_button)
        output_layout.addRow("Output", output_row)
        layout.addWidget(output_group)

        sam_group = QGroupBox("SAM3 Settings")
        sam_layout = _form_layout(sam_group)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Select a local SAM3 model…")
        self.browse_model_button = QPushButton(icon(ICONS["folder"]), "Browse")
        self.browse_model_button.clicked.connect(
            lambda _checked=False: self.browse_model_requested.emit()
        )
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_path_edit, 1)
        model_row.addWidget(self.browse_model_button)
        sam_layout.addRow("Model", model_row)

        self.model_validation_label = QLabel()
        self.model_validation_label.setObjectName("validationError")
        self.model_validation_label.setWordWrap(True)
        self.model_validation_label.setVisible(False)
        sam_layout.addRow("", self.model_validation_label)

        self.prompts_edit = ClassPromptEditor(visible_rows=5)
        self.prompts_edit.setPlaceholderText("One class per line, or comma-separated")
        self.prompts_edit.setAccessibleName("SAM3 class prompts")
        sam_layout.addRow("Classes", self.prompts_edit)

        self.prompt_validation_label = QLabel()
        self.prompt_validation_label.setObjectName("validationError")
        self.prompt_validation_label.setWordWrap(True)
        self.prompt_validation_label.setVisible(False)
        sam_layout.addRow("", self.prompt_validation_label)

        self.conf_edit = QDoubleSpinBox()
        self.conf_edit.setRange(0.01, 1.0)
        self.conf_edit.setDecimals(2)
        self.conf_edit.setSingleStep(0.05)
        self.conf_edit.setValue(0.50)
        self.conf_edit.setKeyboardTracking(False)
        sam_layout.addRow("Confidence", self.conf_edit)

        self.half_check = QCheckBox("Use FP16")
        self.half_check.setChecked(True)
        self.half_check.setToolTip("Uses less GPU memory on compatible CUDA devices.")
        sam_layout.addRow("Precision", self.half_check)
        layout.addWidget(sam_group)

        note = QLabel(
            "Box or class edits invalidate segmentation. Re-segment those objects before "
            "segmentation export when masks/polygons are required."
        )
        note.setObjectName("mutedLabel")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        self.scroll_area.setWidget(content)
        outer.addWidget(self.scroll_area, 1)

        actions_footer = QWidget()
        actions_footer.setObjectName("panelActionFooter")
        footer_layout = QHBoxLayout(actions_footer)
        footer_layout.setContentsMargins(12, 9, 12, 10)
        footer_layout.setSpacing(6)
        footer_layout.addStretch(1)
        self.cancel_button = QPushButton("Cancel")
        self.apply_button = QPushButton("Apply")
        self.apply_button.setObjectName("primaryButton")
        footer_layout.addWidget(self.cancel_button)
        footer_layout.addWidget(self.apply_button)
        outer.addWidget(actions_footer)

        self.cancel_button.clicked.connect(
            lambda _checked=False: self.cancel_requested.emit()
        )
        self.apply_button.clicked.connect(
            lambda _checked=False: self.apply_requested.emit()
        )
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


def _form_layout(parent):
    layout = QFormLayout(parent)
    layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
    return layout
