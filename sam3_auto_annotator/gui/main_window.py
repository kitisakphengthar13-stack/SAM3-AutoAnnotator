from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QUndoStack
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
from sam3_auto_annotator.gui.theme import APP_STYLESHEET
from sam3_auto_annotator.gui.undo import ImageSnapshotCommand
from sam3_auto_annotator.gui.views.annotation_panel import AnnotationPanel
from sam3_auto_annotator.gui.views.dataset_panel import DatasetPanel
from sam3_auto_annotator.gui.views.main_toolbar import CommandBar, build_menus
from sam3_auto_annotator.gui.views.results_panel import ResultsPanel
from sam3_auto_annotator.gui.views.setup_panel import SetupPanel
from sam3_auto_annotator.gui.views.workspace import CanvasWorkspace
from sam3_auto_annotator.gui.widgets.elided_label import ElidedLabel
from sam3_auto_annotator.services.project_service import parse_prompts


IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)"


class _SurfaceRouter:
    """Temporary controller boundary while surfaces are no longer tabs."""

    def __init__(self, window):
        self.window = window
        self._current = window.setup

    def setCurrentWidget(self, widget):
        self._current = widget
        if widget is self.window.setup:
            self.window.show_setup()
        elif widget is self.window.annotation:
            self.window.show_review()
        elif widget is self.window.results:
            self.window.show_results()

    def currentWidget(self):
        return self._current


class MainWindow(QMainWindow):
    """Canvas-first desktop shell; workflow coordination lives in AppController."""

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
        self._undo_project = None
        self._pending_undo_capture = None
        self._draw_class_restore = None
        self._setup_snapshot = None
        self._setup_snapshot_pending = False

        self.actions = AppActions(self)
        self.undo_stack = QUndoStack(self)
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
        self.inspector = _SurfaceRouter(self)

        self.actions.project_settings.triggered.connect(self.show_setup)
        self.actions.export_dialog.triggered.connect(self.show_export_preflight)
        self.actions.fit.triggered.connect(self.canvas.fit_to_window)
        self.actions.zoom_in.triggered.connect(self.canvas.zoom_in)
        self.actions.zoom_out.triggered.connect(self.canvas.zoom_out)
        self.actions.actual_size.triggered.connect(self.canvas.actual_size)
        self.actions.focus_workspace.toggled.connect(self.set_focus_workspace)
        self.actions.fullscreen.toggled.connect(self.set_fullscreen)
        self.actions.mark_reviewed.triggered.connect(
            lambda: QTimer.singleShot(0, self._advance_after_review)
        )
        self.setup.apply_requested.connect(self._apply_setup)
        self.setup.cancel_requested.connect(self.setup_dialog.reject)
        self.setup_dialog.rejected.connect(self._restore_setup_snapshot)
        self._set_canvas_navigation_enabled(False)
        self._setup_undo_tracking()

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

    def _setup_undo_tracking(self):
        self.actions.undo.triggered.connect(self.undo_stack.undo)
        self.actions.redo.triggered.connect(self.undo_stack.redo)
        self.undo_stack.canUndoChanged.connect(self.actions.undo.setEnabled)
        self.undo_stack.canRedoChanged.connect(self.actions.redo.setEnabled)

        for action, text in (
            (self.actions.apply_class, "Change class"),
            (self.actions.apply_box, "Edit box"),
            (self.actions.reset_sam3, "Reset annotation"),
            (self.actions.delete_annotation, "Delete annotation"),
        ):
            action.triggered.connect(
                lambda _checked=False, label=text: self._begin_undoable_edit(label)
            )

        self.canvas.box_drawn.connect(self._prepare_active_class_for_draw)
        self.canvas.box_drawn.connect(
            lambda _box: self._begin_undoable_edit("Add annotation")
        )
        self.canvas.annotation_changed.connect(
            lambda _annotation_id, _box: self._begin_undoable_edit("Edit box")
        )
        for action in (
            self.actions.run_current,
            self.actions.run_remaining,
            self.actions.resegment,
        ):
            action.triggered.connect(
                lambda _checked=False: QTimer.singleShot(
                    0, self._clear_undo_if_inference_started
                )
            )

    def _prepare_active_class_for_draw(self, _box):
        controller = self.controller
        image = controller.current_image if controller is not None else None
        if image is None:
            return
        previous_index = self.annotation.class_combo.currentIndex()
        previous_count = len(image.annotations)
        self.annotation.class_combo.setCurrentIndex(
            self.canvas_area.active_class_combo.currentIndex()
        )
        self._draw_class_restore = (image, previous_count, previous_index)
        QTimer.singleShot(0, self._restore_draw_class_if_failed)

    def _restore_draw_class_if_failed(self):
        restore = self._draw_class_restore
        self._draw_class_restore = None
        if restore is None or self.controller is None:
            return
        image, previous_count, previous_index = restore
        if self.controller.current_image is image and len(image.annotations) == previous_count:
            self.annotation.class_combo.setCurrentIndex(previous_index)

    def _begin_undoable_edit(self, text):
        controller = self.controller
        image = controller.current_image if controller is not None else None
        if image is None or self._pending_undo_capture is not None:
            return
        self._sync_undo_project()
        self._pending_undo_capture = (
            image,
            image.to_dict(),
            str(text),
            controller.selected_annotation_id,
        )
        QTimer.singleShot(0, self._finish_undoable_edit)

    def _finish_undoable_edit(self):
        capture = self._pending_undo_capture
        self._pending_undo_capture = None
        if capture is None or self.controller is None:
            return
        image, before, text, selected_id = capture
        if self.controller.project is not self._undo_project:
            self._sync_undo_project()
            return
        after = image.to_dict()
        if before == after:
            return
        self.undo_stack.push(
            ImageSnapshotCommand(
                image,
                before,
                after,
                self._apply_undo_snapshot,
                text=text,
                selected_annotation_id=selected_id,
            )
        )

    def _apply_undo_snapshot(self, image_index, selected_annotation_id):
        controller = self.controller
        if controller is None or controller.project is not self._undo_project:
            return
        controller.dirty = True
        self.dataset.refresh(image_index)
        if controller.current_image_index == image_index:
            controller._render_current_annotations(selected_annotation_id)
        else:
            controller._update_actions()
            controller._update_context()

    def _sync_undo_project(self):
        project = self.controller.project if self.controller is not None else None
        if project is self._undo_project:
            return
        self._undo_project = project
        self._pending_undo_capture = None
        self.undo_stack.clear()

    def _clear_undo_if_inference_started(self):
        controller = self.controller
        mode = getattr(getattr(controller, "mode", None), "value", "")
        if mode in {"predicting", "batch", "resegmenting"}:
            self.undo_stack.clear()

    def show_setup(self):
        if not self.setup_dialog.isVisible():
            self._setup_snapshot_pending = True
            QTimer.singleShot(0, self._capture_setup_snapshot)
        self.setup_dialog.show()
        self.setup_dialog.raise_()
        self.setup_dialog.activateWindow()

    def _capture_setup_snapshot(self):
        if not self._setup_snapshot_pending or not self.setup_dialog.isVisible():
            return
        self._setup_snapshot_pending = False
        self._setup_snapshot = self.setup.snapshot()

    def _apply_setup(self):
        controller = self.controller
        if controller is not None and controller.project is not None:
            prompts = parse_prompts(self.setup.prompts_text())
            prompt_error = controller._prompt_validation_error(prompts)
            if prompt_error:
                self.setup.set_prompt_error(prompt_error)
                controller._update_actions()
                controller._update_context()
                return
        self.setup.settings_changed.emit()
        if self.setup.prompt_validation_label.isVisible():
            return
        self._setup_snapshot = None
        self._setup_snapshot_pending = False
        self.setup_dialog.accept()

    def _restore_setup_snapshot(self):
        if self._setup_snapshot is not None:
            self.setup.restore_snapshot(self._setup_snapshot)
        self._setup_snapshot = None
        self._setup_snapshot_pending = False
        if self.controller is not None:
            self.controller._update_actions()
            self.controller._update_context()

    def show_review(self):
        self.annotation_dock.show()
        self.annotation_dock.raise_()

    def show_export_preflight(self):
        project = self.controller.project if self.controller is not None else None
        if project is None:
            return
        images = list(project.images)
        reviewed = sum(
            getattr(image.status, "value", image.status) == "reviewed" for image in images
        )
        incomplete = sum(
            getattr(image.status, "value", image.status) in {"not_predicted", "error"}
            for image in images
        )
        stale_segmentation = sum(
            1
            for image in images
            for annotation in image.active_annotations
            if not annotation.segmentation_valid
        )
        needs_review = len(images) - reviewed
        warning = needs_review > 0 or incomplete > 0 or stale_segmentation > 0
        self.results.set_status(
            "Review export warnings before writing files."
            if warning
            else "Project is ready to export.",
            "\n".join(
                (
                    f"Reviewed images: {reviewed}/{len(images)}",
                    f"Needs review: {needs_review}",
                    f"Unpredicted / failed: {incomplete}",
                    f"Stale / missing segmentation: {stale_segmentation}",
                )
            ),
        )
        self.actions.export.setText("Export Anyway" if warning else "Export Now")
        self.show_results()

    def show_results(self):
        self.results_dialog.show()
        self.results_dialog.raise_()
        self.results_dialog.activateWindow()

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

    def _advance_after_review(self):
        if self.controller is None:
            return
        image = self.controller.current_image
        if image is None or getattr(image.status, "value", None) != "reviewed":
            return
        if self.actions.next_image.isEnabled():
            self.dataset.select_relative(1)

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
        self._sync_undo_project()
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
        if title == "Incomplete Images" and self.results_dialog.isVisible():
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
