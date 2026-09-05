from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QTimer

from gui.controllers.state import UiMode
from services.annotation_service import apply_box_segmentation
from services.project_service import remaining_prediction_targets
from storage.image_catalog import validate_model_path


logger = logging.getLogger(__name__)


class InferenceController:
    """Own SAM3 command, task lifecycle, and application of inference results."""

    def __init__(self, host):
        self.host = host

    def settings(self):
        host = self.host
        prompts = host.projects.sync_project_settings(require_prompts=True)
        model_path = host.view.setup.model_path_edit.text().strip()
        validate_model_path(model_path)
        return {
            "model_path": model_path,
            "prompts": prompts,
            "confidence": host.view.setup.conf_edit.value(),
            "half": host.view.setup.half_check.isChecked(),
        }

    def run_current(self):
        host = self.host
        image = host.current_image
        if image is None or not host._current_image_loaded:
            return
        try:
            settings = self.settings()
        except Exception as exc:
            self.finish_start_failure(exc)
            return
        if image.active_annotations and not host.view.confirm(
            "Replace Current Annotations",
            "Running SAM3 will replace all annotations on the selected image.",
            confirm_text="Replace and Run",
        ):
            return
        try:
            self.start_task(UiMode.PREDICTING)
            host.tasks.start_prediction(
                image.image_index,
                image_path=image.image_path,
                **settings,
            )
            host.view.task_progress.show_running("Running SAM3…")
        except Exception as exc:
            self.finish_start_failure(exc)

    def run_remaining(self):
        host = self.host
        if host.project is None:
            return
        targets = remaining_prediction_targets(host.project)
        if not targets:
            host.view.show_info(
                "Nothing to Run",
                "All images are already predicted, edited, reviewed, or marked as no detection.",
            )
            return
        try:
            settings = self.settings()
            self.start_task(UiMode.BATCH)
            host.tasks.start_batch(targets, **settings)
            host.view.task_progress.show_running(
                f"Preparing {len(targets)} images…",
                len(targets),
                cancellable=True,
            )
        except Exception as exc:
            self.finish_start_failure(exc)

    def resegment_selected(self):
        host = self.host
        annotation = host.selected_annotation
        image = host.current_image
        if annotation is None or image is None:
            return
        try:
            settings = self.settings()
            settings.pop("prompts")
            self.start_task(UiMode.RESEGMENTING)
            host.tasks.start_segmentation(
                image.image_index,
                annotation.id,
                image_path=image.image_path,
                box_xyxy=annotation.box_xyxy,
                class_name=annotation.class_name,
                **settings,
            )
            host.view.task_progress.show_running("Re-segmenting selected box…")
        except Exception as exc:
            self.finish_start_failure(exc)

    def start_task(self, mode):
        host = self.host
        if host.tasks.is_running:
            raise RuntimeError("Another SAM3 task is already running.")
        host.mode = mode
        host._task_project = host.project
        host.presentation.update_actions()

    def finish_start_failure(self, exc):
        host = self.host
        host.mode = UiMode.READY if host.project is not None else UiMode.EMPTY
        host._task_project = None
        host.presentation.update_actions()
        host.presentation.report_error(
            "Could Not Start SAM3",
            "SAM3 could not be started with the current settings.",
            "Select a valid model, enter at least one class, and retry.",
            exc,
        )

    def cancel_task(self):
        host = self.host
        if host.tasks.request_cancel():
            host.view.actions.cancel_batch.setEnabled(False)
            host.view.task_progress.status_label.setText(
                "Cancel requested; waiting for the current image…"
            )

    def task_status(self, message):
        self.host.view.task_progress.status_label.setText(message)
        self.host.view.set_message(message)

    def task_started(self, _kind):
        host = self.host
        host.view.history.clear_for_inference_boundary()
        host.presentation.update_actions()

    def batch_progress(self, current, total, image_path):
        host = self.host
        message = f"{current}/{total}  {Path(image_path).name}"
        host.view.task_progress.update_progress(current - 1, total, message)
        host.view.set_message(f"Running SAM3: {message}")

    def prediction_ready(self, image_index, prediction):
        host = self.host
        if host.project is not host._task_project:
            return
        image = host.project.get_image(image_index)
        if prediction.width is not None and prediction.height is not None:
            image.width, image.height = prediction.width, prediction.height
        image.replace_sam3_drafts(prediction.annotations)
        host.presentation.mark_dirty(refresh=False)
        host.view.dataset.refresh(image_index)
        if host.current_image_index == image_index:
            host.annotations.render_current_annotations()
            host.view.show_review()
        else:
            host.presentation.update_context()
        host.view.task_progress.show_result(
            f"SAM3 complete: {len(prediction.annotations)} annotations"
        )

    def prediction_failed(self, image_index, message):
        host = self.host
        if host.project is host._task_project:
            image = host.project.get_image(image_index)
            image.mark_error(message)
            host.presentation.mark_dirty(refresh=False)
            host.view.dataset.refresh(image_index)
        self.prediction_error(message)

    def segmentation_ready(self, image_index, annotation_id, result):
        host = self.host
        if host.project is not host._task_project:
            return
        image = host.project.get_image(image_index)
        try:
            apply_box_segmentation(
                image,
                annotation_id,
                result.polygon_xyn,
                result.confidence,
            )
            host.presentation.mark_dirty(refresh=False)
            if host.current_image_index == image_index:
                host.annotations.render_current_annotations(annotation_id)
            else:
                host.presentation.update_context()
            host.view.dataset.refresh(image_index)
            host.view.task_progress.show_result("Re-segmentation complete.")
        except Exception as exc:
            self.prediction_error(str(exc), exc)

    def segmentation_failed(self, _image_index, _annotation_id, message):
        self.prediction_error(message)

    def batch_image_ready(self, image_index, prediction):
        host = self.host
        if host.project is not host._task_project:
            return
        image = host.project.get_image(image_index)
        if prediction.width is not None and prediction.height is not None:
            image.width, image.height = prediction.width, prediction.height
        image.replace_sam3_drafts(prediction.annotations)
        host.presentation.mark_dirty(refresh=False)
        host.view.dataset.refresh(image_index)
        if host.current_image_index == image_index:
            host.annotations.render_current_annotations()

    def batch_image_failed(self, image_index, message):
        host = self.host
        if host.project is not host._task_project:
            return
        image = host.project.get_image(image_index)
        image.mark_error(message)
        host.presentation.mark_dirty(refresh=False)
        host.view.dataset.refresh(image_index)

    def batch_completed(self, summary):
        host = self.host
        message = self.batch_message("Batch complete", summary)
        host.view.task_progress.update_progress(summary["total"], summary["total"], message)
        host.view.results.set_status(message)
        host.view.set_message(message + " Save the project to persist the results.")

    def batch_cancelled(self, summary):
        host = self.host
        message = self.batch_message("Batch cancelled", summary)
        host.view.task_progress.show_result(message)
        host.view.results.set_status(message)
        host.view.set_message(message + " Partial results remain available.")

    @staticmethod
    def batch_message(prefix, summary):
        return (
            f"{prefix}: {summary['processed']} processed, "
            f"{summary['predicted']} detected, {summary['no_detection']} empty, "
            f"{summary['errors']} errors."
        )

    def task_failed(self, message):
        self.prediction_error(message)

    def prediction_error(self, message, exc=None):
        logger.error("SAM3 task failed: %s", message, exc_info=exc is not None)
        self.host.view.task_progress.show_result("SAM3 stopped with an error.")
        self.host.view.show_error(
            "SAM3 Error",
            "SAM3 could not complete the requested operation.",
            next_action="Check the model path, available memory, and input image, then retry.",
            details=str(message),
        )

    def task_finished(self, _kind):
        host = self.host
        host.mode = UiMode.READY if host.project is not None else UiMode.EMPTY
        host._task_project = None
        host.presentation.update_actions()
        host.presentation.update_context()
        if host._close_pending:
            host._close_pending = False
            QTimer.singleShot(0, host.view.close)
