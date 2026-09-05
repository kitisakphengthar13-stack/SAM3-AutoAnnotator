from __future__ import annotations

from PySide6.QtCore import QObject

from gui.controllers.annotation_controller import AnnotationController
from gui.controllers.export_controller import ExportController
from gui.controllers.inference_controller import InferenceController
from gui.controllers.presentation_controller import PresentationController
from gui.controllers.project_controller import ProjectController
from gui.controllers.state import UiMode
from gui.tasks.inference_task_manager import InferenceTaskManager
from services.prediction_service import PredictionService


class WorkstationController(QObject):
    """Compose focused controllers and own shared workstation state."""

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
        self.presentation.show_initial_state()

    def _connect_actions(self):
        actions = self.view.actions
        connections = {
            actions.open_image: self.projects.open_image,
            actions.open_folder: self.projects.open_folder,
            actions.open_state: self.projects.open_project,
            actions.import_yolo: self.projects.import_yolo,
            actions.save: self.projects.save_project,
            actions.run_current: self.inference.run_current,
            actions.run_remaining: self.inference.run_remaining,
            actions.delete_annotation: self.annotations.delete_selected,
            actions.export: self.exports.export_labels,
            actions.previous_image: lambda: self.view.dataset.select_relative(-1),
            actions.next_image: lambda: self.view.dataset.select_relative(1),
            actions.apply_class: self.annotations.apply_selected_class,
            actions.apply_box: self.annotations.apply_box_fields,
            actions.resegment: self.inference.resegment_selected,
            actions.reset_sam3: self.annotations.reset_selected,
            actions.mark_reviewed: self.annotations.review_current_and_select_next,
            actions.save_preview: self.exports.save_preview,
            actions.open_preview: self.exports.open_preview,
            actions.open_output: self.exports.open_output,
            actions.cancel_batch: self.inference.cancel_task,
        }
        for action, callback in connections.items():
            action.triggered.connect(lambda _checked=False, fn=callback: fn())
        actions.draw_box.toggled.connect(self.annotations.toggle_draw_mode)

    def _connect_views(self):
        self.view.dataset.image_selected.connect(self.presentation.select_image)
        self.view.dataset.filter_changed.connect(self.presentation.dataset_filter_changed)
        self.view.canvas_area.empty_state.open_image_requested.connect(
            self.projects.open_image
        )
        self.view.canvas_area.empty_state.open_folder_requested.connect(
            self.projects.open_folder
        )
        self.view.canvas_area.image_load_error.retry_requested.connect(
            self.presentation.load_current_image
        )
        self.view.canvas.box_drawn.connect(self.annotations.add_manual_box)
        self.view.canvas.annotation_selected.connect(self.annotations.select_annotation)
        self.view.canvas.annotation_changed.connect(self.annotations.canvas_box_changed)
        self.view.annotation.annotation_selected.connect(self.annotations.select_annotation)
        self.view.annotation.editing_changed.connect(self.presentation.update_actions)
        self.view.setup.browse_model_requested.connect(self.projects.browse_model)
        self.view.setup.browse_output_requested.connect(self.projects.browse_output)
        self.view.setup.settings_changed.connect(self.projects.settings_changed)
        for checkbox in (
            self.view.canvas_area.show_boxes_check,
            self.view.canvas_area.show_masks_check,
            self.view.canvas_area.show_polygons_check,
        ):
            checkbox.toggled.connect(self.annotations.update_overlays)

    def _connect_tasks(self):
        self.tasks.task_started.connect(self.inference.task_started)
        self.tasks.status.connect(self.inference.task_status)
        self.tasks.progress.connect(self.inference.batch_progress)
        self.tasks.prediction_ready.connect(self.inference.prediction_ready)
        self.tasks.prediction_failed.connect(self.inference.prediction_failed)
        self.tasks.segmentation_ready.connect(self.inference.segmentation_ready)
        self.tasks.segmentation_failed.connect(self.inference.segmentation_failed)
        self.tasks.batch_image_ready.connect(self.inference.batch_image_ready)
        self.tasks.batch_image_failed.connect(self.inference.batch_image_failed)
        self.tasks.batch_completed.connect(self.inference.batch_completed)
        self.tasks.batch_cancelled.connect(self.inference.batch_cancelled)
        self.tasks.task_failed.connect(self.inference.task_failed)
        self.tasks.task_finished.connect(self.inference.task_finished)

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
                self.projects.save_project()
                if self.dirty:
                    event.ignore()
                    return
            elif decision != "discard":
                event.ignore()
                return
        self.settings.save_window(self.view)
        event.accept()
