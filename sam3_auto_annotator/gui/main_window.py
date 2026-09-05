from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QVBoxLayout,
)

from sam3_auto_annotator.gui.actions import AppActions
from sam3_auto_annotator.gui.coordinators import (
    AnnotationHistoryCoordinator,
    AnnotationInteractionCoordinator,
    ExportDialogCoordinator,
    SetupDialogCoordinator,
)
from sam3_auto_annotator.gui.coordinators.surface_compat import ControllerSurfaceAdapter
from sam3_auto_annotator.gui.theme import APP_STYLESHEET
from sam3_auto_annotator.gui.views.annotation_panel import AnnotationPanel
from sam3_auto_annotator.gui.views.dataset_panel import DatasetPanel
from sam3_auto_annotator.gui.views.main_toolbar import CommandBar, build_menus
from sam3_auto_annotator.gui.views.results_panel import ResultsPanel
from sam3_auto_annotator.gui.views.setup_panel import SetupPanel
from sam3_auto_annotator.gui.views.workspace import CanvasWorkspace
from sam3_auto_annotator.gui.widgets.elided_label import ElidedLabel


IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)"


class MainWindow(QMainWindow):
    """Compose the canvas workstation; application workflows live elsewhere."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("mainWindow")
        self.setWindowTitle("SAM3 AutoAnnotator")
        self.setMinimumSize(960, 620)
        self.resize(1360, 840)

        self.controller = None
        self.ui_settings = None
        self.diagnostic_log_path = None
        self._focus_previous_visibility = (True, True)
        self._fullscreen_restore_maximized = False

        self.actions = AppActions(self)
        self.exit_action = build_menus(self, self.actions)
        self.command_bar = CommandBar(self.actions, self)
        self.addToolBar(self.command_bar)

        self.canvas_area = CanvasWorkspace(self.actions, self)
        self.workspace = self.canvas_area
        self.setCentralWidget(self.canvas_area)

        self.dataset = DatasetPanel(self.actions)
        self.dataset_dock = self._dock(
            "Dataset", "datasetDock", self.dataset, Qt.LeftDockWidgetArea
        )

        self.annotation = AnnotationPanel(self.actions)
        self.annotation_dock = self._dock(
            "Objects", "objectsDock", self.annotation, Qt.RightDockWidgetArea
        )
        self.annotation.setMinimumWidth(280)
        self.canvas_area.active_class_combo.setModel(self.annotation.class_combo.model())

        self.view_menu.addSeparator()
        dataset_toggle = self.dataset_dock.toggleViewAction()
        dataset_toggle.setText("Dataset Panel")
        objects_toggle = self.annotation_dock.toggleViewAction()
        objects_toggle.setText("Objects Panel")
        self.view_menu.addAction(dataset_toggle)
        self.view_menu.addAction(objects_toggle)

        self.setup = SetupPanel(self.actions)
        self.setup_dialog = self._dialog("Project Setup", self.setup, 430, 650)
        self.setup_dialog.setWindowModality(Qt.WindowModal)

        self.results = ResultsPanel(self.actions)
        self.results_dialog = self._dialog("Export", self.results, 500, 680)
        self.results_dialog.setWindowModality(Qt.WindowModal)

        # Cross-view review follow-up and edit-history policy stay outside
        # the window shell.
        self.annotation_interaction = AnnotationInteractionCoordinator(self)
        self.history = AnnotationHistoryCoordinator(self)
        self.undo_stack = self.history.stack
        self.setup_flow = SetupDialogCoordinator(self)
        self.export_flow = ExportDialogCoordinator(self)

        # Remove after AppController no longer targets the retired Inspector API.
        self.inspector = ControllerSurfaceAdapter(self)

        self.actions.project_settings.triggered.connect(self.show_setup)
        self.actions.export_dialog.triggered.connect(self.show_export_preflight)
        self.actions.fit.triggered.connect(self.canvas.fit_to_window)
        self.actions.zoom_in.triggered.connect(self.canvas.zoom_in)
        self.actions.zoom_out.triggered.connect(self.canvas.zoom_out)
        self.actions.actual_size.triggered.connect(self.canvas.actual_size)
        self.actions.focus_workspace.toggled.connect(self.set_focus_workspace)
        self.actions.fullscreen.toggled.connect(self.set_fullscreen)
        self._set_canvas_navigation_enabled(False)

        self.setStatusBar(QStatusBar(self))
        self.status_context = ElidedLabel("No image | 0 annotations | saved")
        self.status_context.setObjectName("mutedLabel")
        self.status_context.setMinimumWidth(210)
        self.status_context.setMaximumWidth(420)
        self.statusBar().addPermanentWidget(self.status_context, 1)
        self.set_message("Open an image or folder to begin.")
        self.setStyleSheet(APP_STYLESHEET)

    def _dock(self, title, object_name, widget, area):
        dock = QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        features = QDockWidget.DockWidgetFeature
        dock.setFeatures(
            features.DockWidgetClosable
            | features.DockWidgetMovable
            | features.DockWidgetFloatable
        )
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _dialog(self, title, widget, width, height):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(width, height)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        return dialog

    @property
    def canvas(self):
        return self.canvas_area.canvas

    @property
    def task_progress(self):
        return self.canvas_area.task_progress

    def set_controller(self, controller):
        self.controller = controller
        self.history.sync_project()

    def show_setup(self):
        self.setup_flow.show()

    def show_review(self):
        self.annotation_dock.show()
        self.annotation_dock.raise_()

    def show_export_preflight(self):
        self.export_flow.show_preflight()

    def show_results(self):
        self.export_flow.show_results()

    def set_focus_workspace(self, enabled):
        if enabled:
            self._focus_previous_visibility = (
                self.dataset_dock.isVisible(),
                self.annotation_dock.isVisible(),
            )
            self.dataset_dock.hide()
            self.annotation_dock.hide()
            return
        dataset_visible, annotation_visible = self._focus_previous_visibility
        self.dataset_dock.setVisible(dataset_visible)
        self.annotation_dock.setVisible(annotation_visible)

    def set_fullscreen(self, enabled):
        if enabled:
            self._fullscreen_restore_maximized = self.isMaximized()
            self.showFullScreen()
            return
        if self._fullscreen_restore_maximized:
            self.showMaximized()
        else:
            self.showNormal()

    def _set_canvas_navigation_enabled(self, enabled):
        for action in (
            self.actions.zoom_in,
            self.actions.zoom_out,
            self.actions.actual_size,
        ):
            action.setEnabled(bool(enabled))

    def set_message(self, message, timeout=0):
        self.statusBar().showMessage(str(message), int(timeout))

    def set_status_context(self, text):
        self.status_context.setText(str(text))

    def set_project_title(self, text):
        self.history.sync_project()
        self.command_bar.project_label.setText(str(text))

    def show_canvas(self, enabled):
        stack = self.canvas_area.workspace_stack
        stack.setCurrentWidget(self.canvas if enabled else self.canvas_area.empty_state)
        self._set_canvas_navigation_enabled(enabled)

    def show_canvas_error(self, image_name, message):
        state = self.canvas_area.image_load_error
        state.set_error(image_name, message)
        self.canvas_area.workspace_stack.setCurrentWidget(state)
        self._set_canvas_navigation_enabled(False)

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
        title = str(title)
        if title == "Delete Annotation":
            return True
        if self.export_flow.bypass_incomplete_confirmation(title):
            return True

        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Question)
        dialog.setWindowTitle(title)
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
