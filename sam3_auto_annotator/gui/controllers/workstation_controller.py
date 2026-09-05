from __future__ import annotations

import logging
from math import isclose
from pathlib import Path

from PySide6.QtCore import QObject, QTimer

from sam3_auto_annotator.app_paths import discover_default_model
from sam3_auto_annotator.core import ImageStatus
from sam3_auto_annotator.gui.controllers.annotation_controller import AnnotationController
from sam3_auto_annotator.gui.controllers.export_controller import ExportController
from sam3_auto_annotator.gui.controllers.inference_controller import InferenceController
from sam3_auto_annotator.gui.controllers.project_controller import ProjectController
from sam3_auto_annotator.gui.controllers.state import UiMode
from sam3_auto_annotator.gui.tasks.inference_task_manager import InferenceTaskManager
from sam3_auto_annotator.services.prediction_service import PredictionService
from sam3_auto_annotator.services.project_service import (
    parse_prompts,
    remaining_prediction_targets,
)
from sam3_auto_annotator.storage.image_catalog import validate_model_path


logger = logging.getLogger(__name__)


class WorkstationController(QObject):
    """Compose focused use-case controllers and shared workstation policy."""

    def __init__(
        self,
        view,
        settings,
        prediction_service=None,
        task_manager=None,
        parent=None,
    ):
        super().__init__(parent or view)
        self.view = view
        self.settings = settings
        self.project = None
        self.current_image_index = None
        self.selected_annotation_id = None
        self.current_state_path = None
        self._saved_output_dir = None
        self.last_export_result = None
        self.last_preview_path = None
        self.dirty = False
        self.mode = UiMode.EMPTY
        self._rendering = False
        self._selecting = False
        self._close_pending = False
        self._task_project = None
        self._current_image_loaded = False

        self.annotations = AnnotationController(self)
        self.projects = ProjectController(self)
        self.inference = InferenceController(self)
        self.exports = ExportController(self)

        prediction_service = prediction_service or PredictionService()
        self.tasks = task_manager or InferenceTaskManager(
            prediction_service,
            parent=self,
        )

        self.view.set_controller(self)
        self._connect_actions()
        self._connect_views()
        self._connect_tasks()
        self._show_initial_state()

    def _connect_actions(self):
        actions = self.view.actions
        connections = {
            actions.open_image: self.open_image,
            actions.open_folder: self.open_folder,
            actions.open_state: self.open_project,
            actions.import_yolo: self.import_yolo,
            actions.save: self.save_project,
            actions.run_current: self.run_current,
            actions.run_remaining: self.run_remaining,
            actions.delete_annotation: self.delete_selected,
            actions.export: self.export_labels,
            actions.previous_image: lambda: self.view.dataset.select_relative(-1),
            actions.next_image: lambda: self.view.dataset.select_relative(1),
            actions.apply_class: self.apply_selected_class,
            actions.apply_box: self.apply_box_fields,
            actions.resegment: self.resegment_selected,
            actions.reset_sam3: self.reset_selected,
            actions.mark_reviewed: self.mark_current_reviewed,
            actions.save_preview: self.save_preview,
            actions.open_preview: self.open_preview,
            actions.open_output: self.open_output,
            actions.cancel_batch: self.cancel_task,
        }
        for action, callback in connections.items():
            action.triggered.connect(lambda _checked=False, fn=callback: fn())
        actions.draw_box.toggled.connect(self.toggle_draw_mode)

    def _connect_views(self):
        self.view.dataset.image_selected.connect(self.select_image)
        self.view.dataset.filter_changed.connect(self.dataset_filter_changed)
        self.view.canvas_area.empty_state.open_image_requested.connect(self.open_image)
        self.view.canvas_area.empty_state.open_folder_requested.connect(self.open_folder)
        self.view.canvas_area.image_load_error.retry_requested.connect(
            self.load_current_image
        )
        self.view.canvas.box_drawn.connect(self.add_manual_box)
        self.view.canvas.annotation_selected.connect(self.select_annotation)
        self.view.canvas.annotation_changed.connect(self.canvas_box_changed)
        self.view.annotation.annotation_selected.connect(self.select_annotation)
        self.view.annotation.editing_changed.connect(self._update_actions)
        self.view.setup.browse_model_requested.connect(self.browse_model)
        self.view.setup.browse_output_requested.connect(self.browse_output)
        self.view.setup.settings_changed.connect(self.settings_changed)
        for checkbox in (
            self.view.canvas_area.show_boxes_check,
            self.view.canvas_area.show_masks_check,
            self.view.canvas_area.show_polygons_check,
        ):
            checkbox.toggled.connect(self.update_overlays)

    def _connect_tasks(self):
        self.tasks.task_started.connect(self.task_started)
        self.tasks.status.connect(self.task_status)
        self.tasks.progress.connect(self.batch_progress)
        self.tasks.prediction_ready.connect(self.prediction_ready)
        self.tasks.prediction_failed.connect(self.prediction_failed)
        self.tasks.segmentation_ready.connect(self.segmentation_ready)
        self.tasks.segmentation_failed.connect(self.segmentation_failed)
        self.tasks.batch_image_ready.connect(self.batch_image_ready)
        self.tasks.batch_image_failed.connect(self.batch_image_failed)
        self.tasks.batch_completed.connect(self.batch_completed)
        self.tasks.batch_cancelled.connect(self.batch_cancelled)
        self.tasks.task_failed.connect(self.task_failed)
        self.tasks.task_finished.connect(self.task_finished)

    def _show_initial_state(self):
        model_path = discover_default_model()
        if model_path is not None:
            self.view.setup.model_path_edit.setText(str(model_path))
        self.view.dataset.clear()
        self.view.canvas.clear_image()
        self.view.show_canvas(False)
        self.view.results.set_status("No export yet")
        self.view.set_project_title("No project loaded")
        self._update_actions()
        self._update_context()

    @property
    def current_image(self):
        if self.project is None or self.current_image_index is None:
            return None
        try:
            return self.project.get_image(self.current_image_index)
        except KeyError:
            return None

    @property
    def selected_annotation(self):
        image = self.current_image
        if image is None or self.selected_annotation_id is None:
            return None
        annotation = image.annotation_by_id(self.selected_annotation_id)
        return annotation if annotation is not None and annotation.is_active else None

    # Project workflow ----------------------------------------------------
    def _last_directory(self):
        return self.projects.last_directory()

    def _remember_path(self, path):
        return self.projects.remember_path(path)

    def _can_replace_project(self):
        return self.projects.can_replace_project()

    def open_image(self):
        return self.projects.open_image()

    def open_folder(self):
        return self.projects.open_folder()

    def _create_project(self, input_path):
        return self.projects.create_project(input_path)

    def open_project(self):
        return self.projects.open_project()

    def _load_project(self, project, state_path=None):
        return self.projects.load_project(project, state_path)

    def import_yolo(self):
        return self.projects.import_yolo()

    def browse_model(self):
        return self.projects.browse_model()

    def browse_output(self):
        return self.projects.browse_output()

    def settings_changed(self):
        return self.projects.settings_changed()

    def _prompt_validation_error(self, prompts):
        return self.projects.prompt_validation_error(prompts)

    def _apply_settings_if_valid(self, prompts, *, prompts_valid=None):
        return self.projects.apply_settings_if_valid(
            prompts,
            prompts_valid=prompts_valid,
        )

    def _sync_project_settings(self, require_prompts=False):
        return self.projects.sync_project_settings(require_prompts=require_prompts)

    def save_project(self):
        return self.projects.save_project()

    # Dataset and image presentation -------------------------------------
    def select_image(self, image_index):
        if self.project is None:
            return
        self.current_image_index = int(image_index)
        self.selected_annotation_id = None
        self.load_current_image()

    def dataset_filter_changed(self):
        image = self.current_image
        current_visible = False
        if image is not None:
            current_visible = self.view.dataset.filter_model.index_for_image_index(
                image.image_index
            ).isValid()
            if (
                current_visible
                and self.view.dataset.selected_image_index() != image.image_index
            ):
                self.view.dataset.select_image(image.image_index, notify=False)
        if image is not None and not current_visible:
            if self.view.dataset.filter_model.rowCount() > 0:
                self.view.dataset.select_first()
                return
            self.view.canvas_area.canvas_hint.setText(
                f"{image.image_name} is hidden by the current Dataset filter."
            )
        elif image is not None:
            self._update_canvas_hint()
        self._update_actions()
        self._update_context()

    def load_current_image(self):
        image = self.current_image
        if image is None:
            self._current_image_loaded = False
            self.view.canvas.clear_image()
            self.view.show_canvas(False)
            self._update_actions()
            return
        self._current_image_loaded = False
        try:
            width, height = self.view.canvas.load_image(image.image_path)
            self._current_image_loaded = True
            self.view.show_canvas(True)
            if image.width is None or image.height is None:
                image.width, image.height = width, height
                self._mark_dirty(refresh=False)
            self._render_current_annotations()
            self.view.set_message(
                "Run SAM3, draw a box, or select an annotation to edit."
            )
            return
        except Exception as exc:
            image.mark_error(exc)
            self._mark_dirty(refresh=False)
            self.view.actions.select_tool.setChecked(True)
            self.view.dataset.refresh(image.image_index)
            self.view.annotation.set_annotations([])
            self._clear_annotation_selection()
            self.view.canvas_area.canvas_hint.setText(
                f"Could not display {image.image_name}. Select another image or retry."
            )
            self.view.show_canvas_error(image.image_name, str(exc))
            self._report_error(
                "Could Not Load Image",
                "The selected image could not be displayed.",
                "Check the image file and select another image if it is damaged.",
                exc,
            )
        self._update_actions()
        self._update_context()

    # Annotation workflow -------------------------------------------------
    def _render_current_annotations(self, select_id=None):
        return self.annotations.render_current_annotations(select_id)

    def select_annotation(self, annotation_id):
        return self.annotations.select_annotation(annotation_id)

    def _clear_annotation_selection(self):
        return self.annotations.clear_selection()

    def toggle_draw_mode(self, checked):
        return self.annotations.toggle_draw_mode(checked)

    def _update_canvas_hint(self):
        return self.annotations.update_canvas_hint()

    def update_overlays(self):
        return self.annotations.update_overlays()

    def add_manual_box(self, box_xyxy):
        return self.annotations.add_manual_box(box_xyxy)

    def canvas_box_changed(self, annotation_id, box_xyxy):
        return self.annotations.canvas_box_changed(annotation_id, box_xyxy)

    def apply_box_fields(self):
        return self.annotations.apply_box_fields()

    def apply_selected_class(self):
        return self.annotations.apply_selected_class()

    def delete_selected(self):
        return self.annotations.delete_selected()

    def reset_selected(self):
        return self.annotations.reset_selected()

    def mark_current_reviewed(self):
        return self.annotations.mark_current_reviewed()

    def _after_annotation_change(self, select_id=None):
        return self.annotations.after_annotation_change(select_id)

    # Inference command and task lifecycle --------------------------------
    def _inference_settings(self):
        prompts = self._sync_project_settings(require_prompts=True)
        model_path = self.view.setup.model_path_edit.text().strip()
        validate_model_path(model_path)
        return {
            "model_path": model_path,
            "prompts": prompts,
            "confidence": self.view.setup.conf_edit.value(),
            "half": self.view.setup.half_check.isChecked(),
        }

    def run_current(self):
        image = self.current_image
        if image is None or not self._current_image_loaded:
            return
        try:
            settings = self._inference_settings()
        except Exception as exc:
            self._finish_start_failure(exc)
            return
        if image.active_annotations and not self.view.confirm(
            "Replace Current Annotations",
            "Running SAM3 will replace all annotations on the selected image.",
            confirm_text="Replace and Run",
        ):
            return
        try:
            self._start_task(UiMode.PREDICTING)
            self.tasks.start_prediction(
                image.image_index,
                image_path=image.image_path,
                **settings,
            )
            self.view.task_progress.show_running("Running SAM3…")
        except Exception as exc:
            self._finish_start_failure(exc)

    def run_remaining(self):
        if self.project is None:
            return
        targets = remaining_prediction_targets(self.project)
        if not targets:
            self.view.show_info(
                "Nothing to Run",
                "All images are already predicted, edited, reviewed, or marked as no detection.",
            )
            return
        try:
            settings = self._inference_settings()
            self._start_task(UiMode.BATCH)
            self.tasks.start_batch(targets, **settings)
            self.view.task_progress.show_running(
                f"Preparing {len(targets)} images…",
                len(targets),
                cancellable=True,
            )
        except Exception as exc:
            self._finish_start_failure(exc)

    def resegment_selected(self):
        annotation = self.selected_annotation
        image = self.current_image
        if annotation is None or image is None:
            return
        try:
            settings = self._inference_settings()
            settings.pop("prompts")
            self._start_task(UiMode.RESEGMENTING)
            self.tasks.start_segmentation(
                image.image_index,
                annotation.id,
                image_path=image.image_path,
                box_xyxy=annotation.box_xyxy,
                class_name=annotation.class_name,
                **settings,
            )
            self.view.task_progress.show_running("Re-segmenting selected box…")
        except Exception as exc:
            self._finish_start_failure(exc)

    def _start_task(self, mode):
        if self.tasks.is_running:
            raise RuntimeError("Another SAM3 task is already running.")
        self.mode = mode
        self._task_project = self.project
        self._update_actions()

    def _finish_start_failure(self, exc):
        self.mode = UiMode.READY if self.project is not None else UiMode.EMPTY
        self._task_project = None
        self._update_actions()
        self._report_error(
            "Could Not Start SAM3",
            "SAM3 could not be started with the current settings.",
            "Select a valid model, enter at least one class, and retry.",
            exc,
        )

    def cancel_task(self):
        if self.tasks.request_cancel():
            self.view.actions.cancel_batch.setEnabled(False)
            self.view.task_progress.status_label.setText(
                "Cancel requested; waiting for the current image…"
            )

    def task_status(self, message):
        self.view.task_progress.status_label.setText(message)
        self.view.set_message(message)

    def task_started(self, _kind):
        self._update_actions()

    def batch_progress(self, current, total, image_path):
        message = f"{current}/{total}  {Path(image_path).name}"
        self.view.task_progress.update_progress(current - 1, total, message)
        self.view.set_message(f"Running SAM3: {message}")

    # Inference result workflow ------------------------------------------
    def prediction_ready(self, image_index, prediction):
        return self.inference.prediction_ready(image_index, prediction)

    def prediction_failed(self, image_index, message):
        return self.inference.prediction_failed(image_index, message)

    def segmentation_ready(self, image_index, annotation_id, result):
        return self.inference.segmentation_ready(image_index, annotation_id, result)

    def segmentation_failed(self, image_index, annotation_id, message):
        return self.inference.segmentation_failed(image_index, annotation_id, message)

    def batch_image_ready(self, image_index, prediction):
        return self.inference.batch_image_ready(image_index, prediction)

    def batch_image_failed(self, image_index, message):
        return self.inference.batch_image_failed(image_index, message)

    def batch_completed(self, summary):
        return self.inference.batch_completed(summary)

    def batch_cancelled(self, summary):
        return self.inference.batch_cancelled(summary)

    def task_failed(self, message):
        return self.inference.task_failed(message)

    def _prediction_error(self, message, exc=None):
        return self.inference.prediction_error(message, exc)

    def task_finished(self, _kind):
        self.mode = UiMode.READY if self.project is not None else UiMode.EMPTY
        self._task_project = None
        self._update_actions()
        self._update_context()
        if self._close_pending:
            self._close_pending = False
            QTimer.singleShot(0, self.view.close)

    # Export workflow -----------------------------------------------------
    def export_labels(self):
        return self.exports.export_labels()

    def save_preview(self, silent=False):
        return self.exports.save_preview(silent=silent)

    def open_preview(self):
        return self.exports.open_preview()

    def open_output(self):
        return self.exports.open_output()

    def _output_dir(self):
        return self.exports.output_dir()

    # Shared presentation policy -----------------------------------------
    def _mark_dirty(self, *, refresh=True):
        self.dirty = True
        if refresh:
            self._update_actions()
            self._update_context()

    @staticmethod
    def _box_fields_changed(annotation, values):
        return not all(
            isclose(old, new, rel_tol=0.0, abs_tol=0.005)
            for old, new in zip(annotation.box_xyxy, values)
        )

    def _box_editor_changed(self, annotation):
        if annotation is None:
            return False
        try:
            return self._box_fields_changed(
                annotation,
                self.view.annotation.box_values(),
            )
        except (TypeError, ValueError):
            return True

    def _update_actions(self):
        actions = self.view.actions
        idle = self.mode in {UiMode.EMPTY, UiMode.READY}
        project_open = self.project is not None
        ready = self.mode == UiMode.READY and project_open
        image = self.current_image
        annotation = self.selected_annotation
        prompts = parse_prompts(self.view.setup.prompts_text())
        prompt_error = self._prompt_validation_error(prompts)
        settings_valid = prompt_error is None
        model_text = self.view.setup.model_path_edit.text().strip()
        model_ok = bool(model_text) and Path(model_text).is_file()
        model_error = None
        if model_text and not model_ok:
            model_error = "Model file was not found. Select an existing local file."
        image_ready = ready and image is not None and self._current_image_loaded
        targets = remaining_prediction_targets(self.project) if project_open else []
        pending_text = f"Run Pending ({len(targets)})"
        if actions.run_remaining.text() != pending_text:
            actions.run_remaining.setText(pending_text)

        visible_count = self.view.dataset.filter_model.rowCount()
        visible_row = self.view.dataset.selected_visible_row()
        class_changed = bool(
            annotation is not None
            and (
                self.view.annotation.class_combo.currentIndex()
                != annotation.class_id
                or self.view.annotation.class_combo.currentText().strip()
                != annotation.class_name
            )
        )
        box_changed = self._box_editor_changed(annotation)

        self.view.setup.set_prompt_error(prompt_error)
        self.view.setup.set_model_error(model_error)

        for action in (actions.open_image, actions.open_folder, actions.open_state):
            action.setEnabled(idle)
        actions.import_yolo.setEnabled(ready and settings_valid)
        output_text = self.view.setup.output_dir_edit.text().strip()
        output_path = Path(output_text) if output_text else None
        destination_changed = (
            output_path is not None
            and self._saved_output_dir is not None
            and output_path.resolve() != self._saved_output_dir.resolve()
        )
        actions.save.setEnabled(
            ready
            and settings_valid
            and (
                self.dirty
                or self.current_state_path is None
                or destination_changed
            )
        )
        actions.export.setEnabled(ready and settings_valid)
        actions.run_current.setEnabled(
            image_ready and settings_valid and bool(prompts) and model_ok
        )
        actions.run_remaining.setEnabled(
            ready
            and settings_valid
            and bool(prompts)
            and model_ok
            and bool(targets)
        )
        actions.draw_box.setEnabled(
            image_ready and settings_valid and bool(prompts)
        )
        actions.fit.setEnabled(image_ready)
        actions.previous_image.setEnabled(ready and visible_row > 0)
        actions.next_image.setEnabled(
            ready and 0 <= visible_row < visible_count - 1
        )
        actions.apply_class.setEnabled(
            image_ready
            and settings_valid
            and annotation is not None
            and bool(prompts)
            and class_changed
        )
        actions.apply_box.setEnabled(
            image_ready and settings_valid and annotation is not None and box_changed
        )
        actions.delete_annotation.setEnabled(
            image_ready and settings_valid and annotation is not None
        )
        actions.reset_sam3.setEnabled(
            image_ready
            and settings_valid
            and annotation is not None
            and annotation.is_modified_from_sam3
        )
        actions.resegment.setEnabled(
            image_ready
            and settings_valid
            and annotation is not None
            and bool(prompts)
            and model_ok
        )
        actions.mark_reviewed.setEnabled(
            image_ready and image.status != ImageStatus.REVIEWED
        )
        actions.save_preview.setEnabled(image_ready)
        actions.open_preview.setEnabled(
            bool(self.last_preview_path) and Path(self.last_preview_path).is_file()
        )
        actions.open_output.setEnabled(
            project_open and bool(output_text) and Path(output_text).is_dir()
        )
        actions.cancel_batch.setEnabled(
            self.mode == UiMode.BATCH and self.tasks.is_running
        )

        self.view.setup.set_settings_enabled(idle, project_open=project_open)
        self.view.dataset.image_list.setEnabled(ready)
        self.view.annotation.annotation_table.setEnabled(image_ready)
        self.view.canvas.setEnabled(image_ready)
        self._set_detail_fields_enabled(
            image_ready and settings_valid and annotation is not None
        )

    def _set_detail_fields_enabled(self, enabled):
        for widget in (
            self.view.annotation.class_combo,
            self.view.annotation.x1_edit,
            self.view.annotation.y1_edit,
            self.view.annotation.x2_edit,
            self.view.annotation.y2_edit,
        ):
            widget.setEnabled(enabled)

    def _update_context(self):
        image = self.current_image
        project_title = (
            "No project loaded"
            if self.project is None
            else self.project.project_name or "Current project"
        )
        self.view.set_project_title(
            f"{project_title} *" if self.dirty else project_title
        )
        if image is None:
            image_text, count = "No image", 0
        else:
            size = (
                f"{image.width}×{image.height}"
                if image.width is not None and image.height is not None
                else "size unknown"
            )
            image_text = f"{image.image_name} ({size})"
            count = len(image.active_annotations)
        state = "unsaved" if self.dirty else "saved"
        self.view.set_status_context(
            f"{image_text} | {count} annotations | {state}"
        )

    def _report_error(self, title, message, next_action, exc):
        logger.exception("%s: %s", title, exc)
        details = f"{type(exc).__name__}: {exc}"
        if self.view.diagnostic_log_path:
            details += f"\n\nDiagnostic log: {self.view.diagnostic_log_path}"
        self.view.set_message(title)
        self.view.show_error(
            title,
            message,
            next_action=next_action,
            details=details,
        )

    def handle_close_event(self, event):
        if self.tasks.is_running:
            self._close_pending = True
            self.tasks.request_cancel()
            self.view.set_message(
                "Closing after the current SAM3 operation finishes safely…"
            )
            event.ignore()
            return
        if self.dirty:
            decision = self.view.ask_unsaved_changes()
            if decision == "save":
                self.save_project()
                if self.dirty:
                    event.ignore()
                    return
            elif decision != "discard":
                event.ignore()
                return
        self.settings.save_window(self.view, self.view.workspace)
        event.accept()
