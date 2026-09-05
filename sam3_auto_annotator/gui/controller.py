from __future__ import annotations

import logging
from enum import Enum
from math import isclose
from pathlib import Path

from PySide6.QtCore import QObject, QTimer

from sam3_auto_annotator.app_paths import MODELS_DIR, discover_default_model
from sam3_auto_annotator.core import ImageStatus
from sam3_auto_annotator.gui.rendering.annotation_preview import (
    OverlayOptions,
    render_annotation_preview,
)
from sam3_auto_annotator.gui.tasks.inference_task_manager import InferenceTaskManager
from sam3_auto_annotator.services.annotation_service import (
    add_manual_annotation,
    apply_box_segmentation,
    change_annotation_class,
    delete_annotation,
    edit_annotation_box,
    mark_image_reviewed,
    reset_annotation_to_sam3,
)
from sam3_auto_annotator.services.prediction_service import PredictionService
from sam3_auto_annotator.services.project_service import (
    create_project,
    default_output_dir,
    export_project,
    import_yolo_project,
    load_state,
    parse_prompts,
    remaining_prediction_targets,
    save_state_to_output,
)
from sam3_auto_annotator.storage.image_catalog import validate_model_path


logger = logging.getLogger(__name__)


class UiMode(str, Enum):
    EMPTY = "empty"
    READY = "ready"
    PREDICTING = "predicting"
    BATCH = "batch"
    RESEGMENTING = "resegmenting"


class AppController(QObject):
    """Coordinate user intent, project services and presentation state."""

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
            actions.fit: self.view.canvas.fit_to_window,
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

    def _last_directory(self):
        return self.settings.last_directory()

    def _remember_path(self, path):
        if not path:
            return
        candidate = Path(path)
        self.settings.set_last_directory(candidate if candidate.is_dir() else candidate.parent)

    def _can_replace_project(self):
        if not self.dirty:
            return True
        decision = self.view.ask_unsaved_changes()
        if decision == "save":
            self.save_project()
            return not self.dirty
        return decision == "discard"

    def open_image(self):
        path = self.view.choose_image(self._last_directory())
        if not path or not self._can_replace_project():
            return
        self._remember_path(path)
        self._create_project(path)

    def open_folder(self):
        path = self.view.choose_folder("Open Image Folder", self._last_directory())
        if not path or not self._can_replace_project():
            return
        self._remember_path(path)
        self._create_project(path)

    def _create_project(self, input_path):
        try:
            model_path = self.view.setup.model_path_edit.text().strip() or None
            project = create_project(
                input_path=input_path,
                prompts=parse_prompts(self.view.setup.prompts_text()),
                model_path=model_path,
                confidence=self.view.setup.conf_edit.value(),
                half=self.view.setup.half_check.isChecked(),
            )
            self._load_project(project)
            self.view.set_message(
                "Configure classes, then run SAM3 or draw boxes manually."
            )
        except Exception as exc:
            self._report_error(
                "Could Not Open Input",
                "The selected image or folder could not be opened.",
                "Check that it exists and contains supported image files, then retry.",
                exc,
            )

    def open_project(self):
        path = self.view.choose_project(self._last_directory())
        if not path or not self._can_replace_project():
            return
        try:
            project = load_state(path)
            self._remember_path(path)
            self._load_project(project, state_path=Path(path))
            self.view.results.set_status("Project loaded. Continue reviewing or export.")
            self.view.set_message("Annotation project loaded.")
        except Exception as exc:
            self._report_error(
                "Could Not Open Project",
                "The annotation project could not be loaded.",
                "Choose a valid annotation_state.json file or restore a known-good copy.",
                exc,
            )

    def _load_project(self, project, state_path=None):
        if not project.model_path:
            default_model = discover_default_model()
            if default_model is not None:
                project.model_path = str(default_model)
        output_dir = Path(state_path).parent if state_path else default_output_dir(project)

        self.project = project
        self.current_state_path = state_path
        self._saved_output_dir = Path(state_path).parent if state_path else None
        self.last_export_result = None
        self.last_preview_path = None
        self.current_image_index = None
        self.selected_annotation_id = None
        self._current_image_loaded = False
        self.dirty = False
        self.mode = UiMode.READY

        self.view.actions.draw_box.setChecked(False)
        self.view.show_canvas(False)
        self.view.task_progress.hide_when_idle()
        self.view.inspector.setCurrentWidget(self.view.setup)
        if not project.images:
            self.view.canvas.clear_image()
            self.view.annotation.set_annotations([])
            self._clear_annotation_selection()
            self.view.canvas_area.canvas_hint.setText(
                "This project does not contain any images."
            )

        self._rendering = True
        try:
            self.view.setup.set_project(project, output_dir)
            self.view.annotation.set_classes(project.prompts)
            self.view.dataset.set_images(project.images, project.project_name)
        finally:
            self._rendering = False

        self.view.set_project_title(project.project_name or "Current project")
        self.view.results.reset(output_dir)
        self.view.dataset.select_first()
        self._update_actions()
        self._update_context()

    def import_yolo(self):
        if self.project is None:
            return
        label_dir = self.view.choose_folder(
            "Select YOLO Detection Label Folder", self._last_directory()
        )
        if not label_dir:
            return
        try:
            summary = import_yolo_project(self.project, label_dir)
            self._remember_path(label_dir)
            self._rendering = True
            try:
                self.view.setup.set_prompts(self.project.prompts)
                self.view.annotation.set_classes(self.project.prompts)
            finally:
                self._rendering = False
            self._mark_dirty(refresh=False)
            self.view.dataset.refresh()
            self._render_current_annotations()
            self.view.inspector.setCurrentWidget(self.view.annotation)
            message = summary.to_message()
            self.view.results.set_status(message)
            self.view.set_message(message)
        except Exception as exc:
            self._report_error(
                "Could Not Import YOLO Labels",
                "The labels could not be imported into this project.",
                "Check the label folder and YOLO detection format, then retry.",
                exc,
            )

    def browse_model(self):
        current = self.view.setup.model_path_edit.text().strip()
        start = str(Path(current).parent) if current else str(MODELS_DIR)
        path = self.view.choose_model(start)
        if path:
            self._remember_path(path)
            self.view.setup.model_path_edit.setText(path)

    def browse_output(self):
        current = self.view.setup.output_dir_edit.text().strip()
        path = self.view.choose_folder("Select Output Folder", current or self._last_directory())
        if path:
            self._remember_path(path)
            self.view.setup.output_dir_edit.setText(path)

    def settings_changed(self):
        if self._rendering:
            return
        prompts = parse_prompts(self.view.setup.prompts_text())
        if self.project is not None:
            prompt_error = self._prompt_validation_error(prompts)
            self.view.annotation.set_classes(
                self.project.prompts if prompt_error else prompts
            )
            changed = self._apply_settings_if_valid(
                prompts,
                prompts_valid=prompt_error is None,
            )
            if changed:
                self._mark_dirty(refresh=False)
        else:
            self.view.annotation.set_classes(prompts)
        self._update_actions()
        self._update_context()

    def _prompt_validation_error(self, prompts):
        if self.project is None:
            return None
        used_names = {item.class_name for item in self.project.active_annotations()}
        missing = sorted(used_names.difference(prompts))
        if not missing:
            return None
        return (
            "Classes in use cannot be removed: "
            + ", ".join(missing)
            + ". Restore them or change those annotations first."
        )

    def _apply_settings_if_valid(self, prompts, *, prompts_valid=None):
        project = self.project
        if prompts_valid is None:
            prompts_valid = self._prompt_validation_error(prompts) is None
        changed = False
        if prompts_valid and prompts != project.prompts:
            project.prompts = list(prompts)
            for annotation in project.active_annotations():
                annotation.class_id = project.class_map[annotation.class_name]
            changed = True

        model_path = self.view.setup.model_path_edit.text().strip() or None
        confidence = self.view.setup.conf_edit.value()
        half = self.view.setup.half_check.isChecked()
        if project.model_path != model_path:
            project.model_path = model_path
            changed = True
        if project.confidence != confidence:
            project.confidence = confidence
            changed = True
        if project.half != half:
            project.half = half
            changed = True
        return changed

    def _sync_project_settings(self, require_prompts=False):
        if self.project is None:
            raise RuntimeError("No project is open.")
        prompts = parse_prompts(self.view.setup.prompts_text())
        if require_prompts and not prompts:
            raise ValueError("Enter at least one class prompt before running SAM3.")
        prompt_error = self._prompt_validation_error(prompts)
        if prompt_error:
            raise ValueError(prompt_error)
        self._apply_settings_if_valid(prompts, prompts_valid=True)
        return prompts

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
            self.view.actions.draw_box.setChecked(False)
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

    def _render_current_annotations(self, select_id=None):
        image = self.current_image
        annotations = (
            image.active_annotations
            if image is not None and self._current_image_loaded
            else []
        )
        self.view.canvas.set_annotations(annotations)
        self.view.annotation.set_annotations(annotations)
        self._clear_annotation_selection()
        self._update_canvas_hint()
        if select_id:
            self.select_annotation(select_id)
        self._update_actions()
        self._update_context()

    def select_annotation(self, annotation_id):
        if self._selecting:
            return
        annotation_id = str(annotation_id) if annotation_id else None
        image = self.current_image
        annotation = image.annotation_by_id(annotation_id) if image and annotation_id else None
        if annotation is None or not annotation.is_active:
            self._clear_annotation_selection()
            self._update_actions()
            return

        self._selecting = True
        try:
            self.selected_annotation_id = annotation.id
            if self.view.canvas.selected_annotation_id() != annotation.id:
                self.view.canvas.select_annotation(annotation.id)
            if self.view.annotation.selected_annotation_id() != annotation.id:
                self.view.annotation.select_annotation(annotation.id)
            self.view.annotation.show_details(annotation)
            self.view.inspector.setCurrentWidget(self.view.annotation)
        finally:
            self._selecting = False
        self._update_actions()

    def _clear_annotation_selection(self):
        self._selecting = True
        try:
            self.selected_annotation_id = None
            self.view.canvas.select_annotation(None)
            self.view.annotation.select_annotation(None)
            self.view.annotation.clear_details()
        finally:
            self._selecting = False

    def toggle_draw_mode(self, checked):
        self.view.canvas.set_draw_mode(bool(checked))
        if checked:
            self.view.canvas_area.canvas_hint.setText(
                "Draw mode: drag on the image to add a bounding box."
            )
        else:
            self._update_canvas_hint()
        self.view.set_message("Draw mode enabled." if checked else "Draw mode disabled.")

    def _update_canvas_hint(self):
        image = self.current_image
        if image is None:
            self.view.canvas_area.canvas_hint.setText(
                "Open an image or folder to start reviewing annotations."
            )
            return
        size = (
            f"{image.width}×{image.height}"
            if image.width is not None and image.height is not None
            else "size unknown"
        )
        self.view.canvas_area.canvas_hint.setText(
            f"{image.image_name}  ·  {size}  ·  "
            f"{len(image.active_annotations)} annotations"
        )

    def update_overlays(self):
        self.view.canvas.set_overlay_visibility(
            show_boxes=self.view.canvas_area.show_boxes_check.isChecked(),
            show_masks=self.view.canvas_area.show_masks_check.isChecked(),
            show_polygons=self.view.canvas_area.show_polygons_check.isChecked(),
        )

    def add_manual_box(self, box_xyxy):
        image = self.current_image
        if image is None:
            return
        try:
            prompts = self._sync_project_settings(require_prompts=True)
        except Exception as exc:
            self.view.show_error(
                "Class Required",
                "The class list is not valid for this project.",
                next_action="Restore every class already in use, then draw the box again.",
                details=str(exc),
            )
            return
        class_id = max(0, self.view.annotation.class_combo.currentIndex())
        class_id = min(class_id, len(prompts) - 1)
        try:
            annotation = add_manual_annotation(
                image, class_id, prompts[class_id], box_xyxy
            )
            self._after_annotation_change(annotation.id)
            self.view.set_message("Manual annotation added.")
        except Exception as exc:
            self._report_error(
                "Could Not Add Annotation",
                "The bounding box could not be added.",
                "Draw a box fully inside the image and retry.",
                exc,
            )

    def canvas_box_changed(self, annotation_id, box_xyxy):
        image = self.current_image
        annotation = image.annotation_by_id(annotation_id) if image else None
        if annotation is None:
            return
        if all(abs(old - new) < 0.5 for old, new in zip(annotation.box_xyxy, box_xyxy)):
            return
        try:
            edit_annotation_box(image, annotation_id, box_xyxy)
            self._after_annotation_change(annotation_id)
            self.view.set_message(
                "Box updated. Re-segment it before segmentation export."
            )
        except Exception as exc:
            self._report_error(
                "Could Not Update Box",
                "The bounding box could not be updated.",
                "Check the coordinates and keep the box inside the image.",
                exc,
            )

    def apply_box_fields(self):
        annotation = self.selected_annotation
        image = self.current_image
        if annotation is None or image is None:
            return
        try:
            values = self.view.annotation.box_values()
            if not self._box_fields_changed(annotation, values):
                return
            edit_annotation_box(image, annotation.id, values)
            self._after_annotation_change(annotation.id)
            self.view.set_message(
                "Box updated. Re-segment it before segmentation export."
            )
        except Exception as exc:
            self._report_error(
                "Invalid Box Coordinates",
                "The box coordinates are not valid.",
                "Use x1 < x2 and y1 < y2 within the image bounds.",
                exc,
            )

    def apply_selected_class(self):
        annotation = self.selected_annotation
        image = self.current_image
        class_id = self.view.annotation.class_combo.currentIndex()
        class_name = self.view.annotation.class_combo.currentText().strip()
        if annotation is None or image is None or class_id < 0 or not class_name:
            return
        if (
            annotation.class_id == class_id
            and annotation.class_name == class_name
        ):
            return
        try:
            change_annotation_class(image, annotation.id, class_id, class_name)
            self._after_annotation_change(annotation.id)
            self.view.set_message(
                "Class updated. Re-segment it before segmentation export."
            )
        except Exception as exc:
            self._report_error(
                "Could Not Change Class",
                "The selected class could not be applied.",
                "Check the class list in Setup and retry.",
                exc,
            )

    def delete_selected(self):
        annotation = self.selected_annotation
        image = self.current_image
        if annotation is None or image is None:
            return
        if not self.view.confirm(
            "Delete Annotation",
            f"Delete the selected {annotation.class_name} annotation?",
            confirm_text="Delete Annotation",
        ):
            return
        delete_annotation(image, annotation.id)
        self._after_annotation_change()
        self.view.set_message("Annotation deleted.")

    def reset_selected(self):
        annotation = self.selected_annotation
        image = self.current_image
        if annotation is None or image is None or not annotation.is_modified_from_sam3:
            return
        try:
            reset_annotation_to_sam3(image, annotation.id)
            self._after_annotation_change(annotation.id)
            self.view.set_message("Original SAM3 annotation restored.")
        except Exception as exc:
            self._report_error(
                "Could Not Reset Annotation",
                "This annotation has no restorable SAM3 geometry.",
                "Keep the current edit or run SAM3 again on the image.",
                exc,
            )

    def mark_current_reviewed(self):
        image = self.current_image
        if image is None or image.status == ImageStatus.REVIEWED:
            return
        mark_image_reviewed(image)
        self._mark_dirty(refresh=False)
        self.view.dataset.refresh(image.image_index)
        self.view.set_message("Image marked as reviewed.")
        self.dataset_filter_changed()

    def _after_annotation_change(self, select_id=None):
        image = self.current_image
        self._mark_dirty(refresh=False)
        if image is not None:
            self.view.dataset.refresh(image.image_index)
        self._render_current_annotations(select_id)

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
        # Emitted only after the manager owns a live thread, so cancellation can
        # be enabled without racing the task manager's state assignment.
        self._update_actions()

    def batch_progress(self, current, total, image_path):
        message = f"{current}/{total}  {Path(image_path).name}"
        self.view.task_progress.update_progress(current - 1, total, message)
        self.view.set_message(f"Running SAM3: {message}")

    def prediction_ready(self, image_index, prediction):
        if self.project is not self._task_project:
            return
        image = self.project.get_image(image_index)
        if prediction.width is not None and prediction.height is not None:
            image.width, image.height = prediction.width, prediction.height
        image.replace_sam3_drafts(prediction.annotations)
        self._mark_dirty(refresh=False)
        self.view.dataset.refresh(image_index)
        if self.current_image_index == image_index:
            self._render_current_annotations()
            self.view.inspector.setCurrentWidget(self.view.annotation)
        else:
            self._update_context()
        self.view.task_progress.show_result(
            f"SAM3 complete: {len(prediction.annotations)} annotations"
        )

    def prediction_failed(self, image_index, message):
        if self.project is self._task_project:
            image = self.project.get_image(image_index)
            image.mark_error(message)
            self._mark_dirty(refresh=False)
            self.view.dataset.refresh(image_index)
        self._prediction_error(message)

    def segmentation_ready(self, image_index, annotation_id, result):
        if self.project is not self._task_project:
            return
        image = self.project.get_image(image_index)
        try:
            apply_box_segmentation(
                image,
                annotation_id,
                result.polygon_xyn,
                result.confidence,
            )
            self._mark_dirty(refresh=False)
            if self.current_image_index == image_index:
                self._render_current_annotations(annotation_id)
            else:
                self._update_context()
            self.view.dataset.refresh(image_index)
            self.view.task_progress.show_result("Re-segmentation complete.")
        except Exception as exc:
            self._prediction_error(str(exc), exc)

    def segmentation_failed(self, _image_index, _annotation_id, message):
        self._prediction_error(message)

    def batch_image_ready(self, image_index, prediction):
        if self.project is not self._task_project:
            return
        image = self.project.get_image(image_index)
        if prediction.width is not None and prediction.height is not None:
            image.width, image.height = prediction.width, prediction.height
        image.replace_sam3_drafts(prediction.annotations)
        self._mark_dirty(refresh=False)
        self.view.dataset.refresh(image_index)
        if self.current_image_index == image_index:
            self._render_current_annotations()

    def batch_image_failed(self, image_index, message):
        if self.project is not self._task_project:
            return
        image = self.project.get_image(image_index)
        image.mark_error(message)
        self._mark_dirty(refresh=False)
        self.view.dataset.refresh(image_index)

    def batch_completed(self, summary):
        message = self._batch_message("Batch complete", summary)
        self.view.task_progress.update_progress(summary["total"], summary["total"], message)
        self.view.results.set_status(message)
        self.view.set_message(message + " Save the project to persist the results.")

    def batch_cancelled(self, summary):
        message = self._batch_message("Batch cancelled", summary)
        self.view.task_progress.show_result(message)
        self.view.results.set_status(message)
        self.view.set_message(message + " Partial results remain available.")

    @staticmethod
    def _batch_message(prefix, summary):
        return (
            f"{prefix}: {summary['processed']} processed, "
            f"{summary['predicted']} detected, {summary['no_detection']} empty, "
            f"{summary['errors']} errors."
        )

    def task_failed(self, message):
        self._prediction_error(message)

    def _prediction_error(self, message, exc=None):
        logger.error("SAM3 task failed: %s", message, exc_info=exc is not None)
        self.view.task_progress.show_result("SAM3 stopped with an error.")
        self.view.show_error(
            "SAM3 Error",
            "SAM3 could not complete the requested operation.",
            next_action="Check the model path, available memory, and input image, then retry.",
            details=str(message),
        )

    def task_finished(self, _kind):
        self.mode = UiMode.READY if self.project is not None else UiMode.EMPTY
        self._task_project = None
        self._update_actions()
        self._update_context()
        if self._close_pending:
            self._close_pending = False
            QTimer.singleShot(0, self.view.close)

    def save_project(self):
        if self.project is None:
            return
        try:
            self._sync_project_settings()
            output_dir = self._output_dir()
            path = save_state_to_output(self.project, output_dir)
            self.current_state_path = Path(path)
            self._saved_output_dir = Path(output_dir)
            self.dirty = False
            self.view.results.set_status("Project state saved.")
            self.view.results.set_output_dir(output_dir)
            self.view.set_message(f"Saved project to {path}")
            self._update_actions()
            self._update_context()
        except Exception as exc:
            self._report_error(
                "Could Not Save Project",
                "The annotation project could not be saved.",
                "Check the output folder permissions and available disk space, then retry.",
                exc,
            )

    def export_labels(self):
        if self.project is None:
            return
        try:
            self._sync_project_settings()
        except Exception as exc:
            self._report_error(
                "Could Not Export Labels",
                "The project settings are not valid for export.",
                "Restore every class in use, then retry.",
                exc,
            )
            return
        incomplete = [
            image
            for image in self.project.images
            if image.status in {ImageStatus.NOT_PREDICTED, ImageStatus.ERROR}
        ]
        if incomplete and not self.view.confirm(
            "Incomplete Images",
            f"{len(incomplete)} image(s) are unpredicted or failed. "
            "They will receive empty YOLO label files unless they contain manual boxes.",
            confirm_text="Export Anyway",
        ):
            return
        try:
            output_dir = self._output_dir()
            self.current_state_path = save_state_to_output(self.project, output_dir)
            self._saved_output_dir = Path(output_dir)
            result = export_project(self.project, output_dir)
            self._validate_export(result)
            self.last_export_result = result
            preview = self.save_preview(silent=True)
            self.dirty = False
            skipped = len(result.get("skipped_segmentation_rows", []))
            counts = (
                f"Detection: {len(result['rows'])}\n"
                f"Segmentation: {len(result['segmentation_rows'])}\n"
                f"Skipped segmentation: {skipped}"
            )
            self.view.results.set_status("Export complete.", counts)
            self.view.results.set_output_paths(
                output_dir=output_dir,
                box_csv=result["box_csv"],
                detection_dir=result["yolo_detection_dir"],
                segmentation_dir=result["yolo_segmentation_dir"],
                skipped_report=result.get("segmentation_skipped_report"),
            )
            if preview:
                self.view.results.set_preview(preview)
            self.view.inspector.setCurrentWidget(self.view.results)
            self.view.set_message(f"Exported corrected labels to {output_dir}")
            self._update_actions()
            self._update_context()
        except Exception as exc:
            self._report_error(
                "Could Not Export Labels",
                "The corrected labels could not be exported.",
                "Check the output folder and project data, then retry.",
                exc,
            )

    @staticmethod
    def _validate_export(result):
        for key in ("box_csv", "run_summary"):
            path = result.get(key)
            if path is not None and not Path(path).is_file():
                raise FileNotFoundError(f"Expected export file was not written: {path}")
        for key in ("yolo_detection_dir", "yolo_segmentation_dir"):
            path = result.get(key)
            if path is not None and not Path(path).is_dir():
                raise FileNotFoundError(f"Expected export folder was not written: {path}")

    def save_preview(self, silent=False):
        image = self.current_image
        if image is None:
            return None
        output_path = (
            self._output_dir()
            / "preview_results"
            / f"{Path(image.image_path).stem}_reviewed.png"
        )
        try:
            render_annotation_preview(
                image.image_path,
                image.active_annotations,
                output_path,
                OverlayOptions(
                    boxes=self.view.canvas_area.show_boxes_check.isChecked(),
                    masks=self.view.canvas_area.show_masks_check.isChecked(),
                    polygons=self.view.canvas_area.show_polygons_check.isChecked(),
                ),
            )
            self.last_preview_path = output_path
            self.view.results.set_preview(output_path)
            if not silent:
                self.view.inspector.setCurrentWidget(self.view.results)
                self.view.set_message(f"Saved preview to {output_path}")
            self._update_actions()
            return output_path
        except Exception as exc:
            if not silent:
                self._report_error(
                    "Could Not Save Preview",
                    "The preview image could not be created.",
                    "Check the source image and output folder, then retry.",
                    exc,
                )
            else:
                logger.exception("Could not save preview")
            return None

    def open_preview(self):
        if self.last_preview_path and Path(self.last_preview_path).is_file():
            self.view.open_local_path(self.last_preview_path)

    def open_output(self):
        if self.project is not None:
            self.view.open_local_path(self._output_dir())

    def _output_dir(self):
        text = self.view.setup.output_dir_edit.text().strip()
        output = Path(text) if text else default_output_dir(self.project)
        if not text:
            self.view.setup.output_dir_edit.setText(str(output))
        return output

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
        # YOLO files can supply numeric ids even when no class names were
        # configured; the importer creates stable placeholder names for them.
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
            image_ready
            and settings_valid
            and bool(prompts)
            and model_ok
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
        actions.cancel_batch.setEnabled(self.mode == UiMode.BATCH and self.tasks.is_running)

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
        self.view.set_status_context(f"{image_text} | {count} annotations | {state}")

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
