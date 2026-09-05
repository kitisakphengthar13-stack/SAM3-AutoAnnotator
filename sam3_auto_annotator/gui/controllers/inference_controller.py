from __future__ import annotations

import logging

from sam3_auto_annotator.services.annotation_service import apply_box_segmentation


logger = logging.getLogger(__name__)


class InferenceController:
    """Apply prediction/task results to the active project and workstation views."""

    def __init__(self, host):
        self.host = host

    def prediction_ready(self, image_index, prediction):
        host = self.host
        if host.project is not host._task_project:
            return
        image = host.project.get_image(image_index)
        if prediction.width is not None and prediction.height is not None:
            image.width, image.height = prediction.width, prediction.height
        image.replace_sam3_drafts(prediction.annotations)
        host._mark_dirty(refresh=False)
        host.view.dataset.refresh(image_index)
        if host.current_image_index == image_index:
            host._render_current_annotations()
            host.view.show_review()
        else:
            host._update_context()
        host.view.task_progress.show_result(
            f"SAM3 complete: {len(prediction.annotations)} annotations"
        )

    def prediction_failed(self, image_index, message):
        host = self.host
        if host.project is host._task_project:
            image = host.project.get_image(image_index)
            image.mark_error(message)
            host._mark_dirty(refresh=False)
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
            host._mark_dirty(refresh=False)
            if host.current_image_index == image_index:
                host._render_current_annotations(annotation_id)
            else:
                host._update_context()
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
        host._mark_dirty(refresh=False)
        host.view.dataset.refresh(image_index)
        if host.current_image_index == image_index:
            host._render_current_annotations()

    def batch_image_failed(self, image_index, message):
        host = self.host
        if host.project is not host._task_project:
            return
        image = host.project.get_image(image_index)
        image.mark_error(message)
        host._mark_dirty(refresh=False)
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
