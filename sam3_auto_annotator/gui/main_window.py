from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QStatusBar

from sam3_auto_annotator.gui.actions import AppActions
from sam3_auto_annotator.gui.theme import APP_STYLESHEET
from sam3_auto_annotator.gui.views.main_toolbar import CommandBar, build_menus
from sam3_auto_annotator.gui.views.workspace import AnnotationWorkspace
from sam3_auto_annotator.gui.widgets.elided_label import ElidedLabel


IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)"


class MainWindow(QMainWindow):
    """Qt shell and user interaction surface; workflow lives in AppController."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("mainWindow")
        self.setWindowTitle("SAM3 AutoAnnotator")
        self.setMinimumSize(960, 620)
        self.resize(1360, 840)
        self.controller = None
        self.ui_settings = None
        self.diagnostic_log_path = None

        self.actions = AppActions(self)
        self.exit_action = build_menus(self, self.actions)
        self.command_bar = CommandBar(self.actions, self)
        self.addToolBar(self.command_bar)

        self.workspace = AnnotationWorkspace(self.actions, self)
        self.setCentralWidget(self.workspace)

        self.setStatusBar(QStatusBar(self))
        self.status_context = ElidedLabel("No image | 0 annotations | saved")
        self.status_context.setObjectName("mutedLabel")
        self.status_context.setMinimumWidth(210)
        self.status_context.setMaximumWidth(340)
        self.statusBar().addPermanentWidget(self.status_context, 1)
        self.set_message("Open an image or folder to begin.")
        self.setStyleSheet(APP_STYLESHEET)

    @property
    def dataset(self):
        return self.workspace.dataset

    @property
    def canvas_area(self):
        return self.workspace.canvas_area

    @property
    def canvas(self):
        return self.workspace.canvas_area.canvas

    @property
    def setup(self):
        return self.workspace.inspector.setup

    @property
    def annotation(self):
        return self.workspace.inspector.annotation

    @property
    def results(self):
        return self.workspace.inspector.results

    @property
    def inspector(self):
        return self.workspace.inspector

    @property
    def task_progress(self):
        return self.workspace.canvas_area.task_progress

    def set_controller(self, controller):
        self.controller = controller

    def set_message(self, message, timeout=0):
        self.statusBar().showMessage(str(message), int(timeout))

    def set_status_context(self, text):
        self.status_context.setText(str(text))

    def set_project_title(self, text):
        self.command_bar.project_label.setText(str(text))

    def show_canvas(self, enabled):
        stack = self.canvas_area.workspace_stack
        stack.setCurrentWidget(self.canvas if enabled else self.canvas_area.empty_state)

    def show_canvas_error(self, image_name, message):
        state = self.canvas_area.image_load_error
        state.set_error(image_name, message)
        self.canvas_area.workspace_stack.setCurrentWidget(state)

    def choose_image(self, start_directory=""):
        return QFileDialog.getOpenFileName(
            self,
            "Open Image",
            start_directory,
            IMAGE_FILTER,
        )[0]

    def choose_folder(self, title, start_directory=""):
        return QFileDialog.getExistingDirectory(self, title, start_directory)

    def choose_project(self, start_directory=""):
        return QFileDialog.getOpenFileName(
            self,
            "Open Annotation Project",
            start_directory,
            "Annotation Project (annotation_state.json);;JSON (*.json)",
        )[0]

    def choose_model(self, start_directory=""):
        return QFileDialog.getOpenFileName(
            self,
            "Select SAM3 Model",
            start_directory,
            "PyTorch Model (*.pt);;All Files (*)",
        )[0]

    def show_error(self, title, message, *, next_action=None, details=None):
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Critical)
        dialog.setWindowTitle(str(title))
        dialog.setText(str(message))
        if next_action:
            dialog.setInformativeText(str(next_action))
        if details:
            dialog.setDetailedText(str(details))
        dialog.exec()

    def show_info(self, title, message):
        QMessageBox.information(self, str(title), str(message))

    def confirm(self, title, message, *, confirm_text=None):
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Question)
        dialog.setWindowTitle(str(title))
        dialog.setText(str(message))
        confirm_button = dialog.addButton(
            confirm_text or "Continue", QMessageBox.AcceptRole
        )
        dialog.addButton(QMessageBox.Cancel)
        dialog.exec()
        return dialog.clickedButton() is confirm_button

    def ask_unsaved_changes(self):
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Warning)
        dialog.setWindowTitle("Unsaved Changes")
        dialog.setText("This project has unsaved changes.")
        dialog.setInformativeText("Save before leaving the current project?")
        dialog.setStandardButtons(
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
        )
        dialog.setDefaultButton(QMessageBox.Save)
        result = dialog.exec()
        if result == QMessageBox.Save:
            return "save"
        if result == QMessageBox.Discard:
            return "discard"
        return "cancel"

    def open_local_path(self, path):
        if path is None:
            return False
        return QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(Path(path).resolve()))
        )

    def closeEvent(self, event):
        if self.controller is None:
            event.accept()
            return
        self.controller.handle_close_event(event)
