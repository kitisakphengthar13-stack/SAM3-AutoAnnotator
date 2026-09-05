from __future__ import annotations

from PySide6.QtCore import QObject

from sam3_auto_annotator.gui.controllers.annotation_controller import AnnotationController
from sam3_auto_annotator.gui.controllers.export_controller import ExportController
from sam3_auto_annotator.gui.controllers.inference_controller import InferenceController
from sam3_auto_annotator.gui.controllers.presentation_controller import PresentationController
from sam3_auto_annotator.gui.controllers.project_controller import ProjectController
from sam3_auto_annotator.gui.controllers.state import UiMode
from sam3_auto_annotator.gui.tasks.inference_task_manager import InferenceTaskManager
from sam3_auto_annotator.services.prediction_service import PredictionService


class WorkstationController(QObject):
    """Application composition root for focused workstation use-case controllers."""

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

        self.projects = ProjectController(self)
        self.annotations = AnnotationController(self)
        self.inference = InferenceController(self)
        self.exports = ExportController(self)
        self.presentation = PresentationController(self)

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

    # Project -------------------------------------------------------------
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

    # Dataset/image presentation -----------------------------------------
    def _show_initial_state(self):
        return self.presentation.show_initial_state()

    def select_image(self, image_index):
        return self.presentation.select_image(image_index)

    def dataset_filter_changed(self):
        return self.presentation.dataset_filter_changed()

    def load_current_image(self):
        return self.presentation.load_current_image()

    # Annotation ----------------------------------------------------------
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

    # Inference -----------------------------------------------------------
    def _inference_settings(self):
        return self.inference.settings()

    def run_current(self):
        return self.inference.run_current()

    def run_remaining(self):
        return self.inference.run_remaining()

    def resegment_selected(self):
        return self.inference.resegment_selected()

    def _start_task(self, mode):
        return self.inference.start_task(mode)

    def _finish_start_failure(self, exc):
        return self.inference.finish_start_failure(exc)

    def cancel_task(self):
        return self.inference.cancel_task()

    def task_status(self, message):
        return self.inference.task_status(message)

    def task_started(self, kind):
        return self.inference.task_started(kind)

    def batch_progress(self, current, total, image_path):
        return self.inference.batch_progress(current, total, image_path)

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

    def task_finished(self, kind):
        return self.inference.task_finished(kind)

    # Export --------------------------------------------------------------
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

    # Presentation helpers ------------------------------------------------
    def _mark_dirty(self, *, refresh=True):
        return self.presentation.mark_dirty(refresh=refresh)

    @staticmethod
    def _box_fields_changed(annotation, values):
        return PresentationController.box_fields_changed(annotation, values)

    def _box_editor_changed(self, annotation):
        return self.presentation.box_editor_changed(annotation)

    def _update_actions(self):
        return self.presentation.update_actions()

    def _set_detail_fields_enabled(self, enabled):
        return self.presentation.set_detail_fields_enabled(enabled)

    def _update_context(self):
        return self.presentation.update_context()

    def _report_error(self, title, message, next_action, exc):
        return self.presentation.report_error(title, message, next_action, exc)

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
