from __future__ import annotations

from sam3_auto_annotator.gui.controller import AppController
from sam3_auto_annotator.gui.controllers.annotation_controller import AnnotationController
from sam3_auto_annotator.gui.controllers.export_controller import ExportController
from sam3_auto_annotator.gui.controllers.inference_controller import InferenceController
from sam3_auto_annotator.gui.controllers.project_controller import ProjectController


class WorkstationController(AppController):
    """Active application controller while legacy AppController is decomposed.

    New use cases move into focused controllers behind this facade. Once every
    legacy responsibility has migrated, the inherited implementation can be
    deleted instead of preserved as permanent architecture.
    """

    def __init__(self, *args, **kwargs):
        self.annotations = AnnotationController(self)
        self.projects = ProjectController(self)
        self.inference = InferenceController(self)
        self.exports = ExportController(self)
        super().__init__(*args, **kwargs)

    # Project workflow ----------------------------------------------------
    def _load_project(self, project, state_path=None):
        return self.projects.load_project(project, state_path)

    def import_yolo(self):
        return self.projects.import_yolo()

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
