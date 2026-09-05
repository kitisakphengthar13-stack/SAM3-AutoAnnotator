from __future__ import annotations

from domain import ImageStatus
from services.annotation_service import (
    add_manual_annotation,
    change_annotation_class,
    delete_annotation,
    edit_annotation_box,
    mark_image_reviewed,
    reset_annotation_to_sam3,
)


class AnnotationController:
    """Own selection, manual editing, review state, and canvas annotation refresh."""

    def __init__(self, host):
        self.host = host

    def render_current_annotations(self, select_id=None):
        host = self.host
        image = host.current_image
        annotations = (
            image.active_annotations
            if image is not None and host._current_image_loaded
            else []
        )
        host.view.canvas.set_annotations(annotations)
        host.view.annotation.set_annotations(annotations)
        self.clear_selection()
        self.update_canvas_hint()
        if select_id:
            self.select_annotation(select_id)
        host.presentation.update_actions()
        host.presentation.update_context()

    def select_annotation(self, annotation_id):
        host = self.host
        if host._selecting:
            return
        annotation_id = str(annotation_id) if annotation_id else None
        image = host.current_image
        annotation = (
            image.annotation_by_id(annotation_id)
            if image is not None and annotation_id
            else None
        )
        if annotation is None or not annotation.is_active:
            self.clear_selection()
            host.presentation.update_actions()
            return

        host._selecting = True
        try:
            host.selected_annotation_id = annotation.id
            if host.view.canvas.selected_annotation_id() != annotation.id:
                host.view.canvas.select_annotation(annotation.id)
            if host.view.annotation.selected_annotation_id() != annotation.id:
                host.view.annotation.select_annotation(annotation.id)
            host.view.annotation.show_details(annotation)
            host.view.show_review()
        finally:
            host._selecting = False
        host.presentation.update_actions()

    def clear_selection(self):
        host = self.host
        host._selecting = True
        try:
            host.selected_annotation_id = None
            host.view.canvas.select_annotation(None)
            host.view.annotation.select_annotation(None)
            host.view.annotation.clear_details()
        finally:
            host._selecting = False

    def toggle_draw_mode(self, checked):
        host = self.host
        host.view.canvas.set_draw_mode(bool(checked))
        if checked:
            host.view.canvas_area.canvas_hint.setText(
                "Box tool: drag on the image to add a bounding box."
            )
        else:
            self.update_canvas_hint()
        host.view.set_message("Box tool enabled." if checked else "Select tool enabled.")

    def update_canvas_hint(self):
        host = self.host
        image = host.current_image
        if image is None:
            host.view.canvas_area.canvas_hint.setText(
                "Open an image or folder to start reviewing annotations."
            )
            return
        size = (
            f"{image.width}×{image.height}"
            if image.width is not None and image.height is not None
            else "size unknown"
        )
        host.view.canvas_area.canvas_hint.setText(
            f"{image.image_name}  ·  {size}  ·  "
            f"{len(image.active_annotations)} annotations"
        )

    def update_overlays(self):
        view = self.host.view
        view.canvas.set_overlay_visibility(
            show_boxes=view.canvas_area.show_boxes_check.isChecked(),
            show_masks=view.canvas_area.show_masks_check.isChecked(),
            show_polygons=view.canvas_area.show_polygons_check.isChecked(),
        )

    def add_manual_box(self, box_xyxy):
        host = self.host
        image = host.current_image
        if image is None:
            return
        try:
            prompts = host.projects.sync_project_settings(require_prompts=True)
        except Exception as exc:
            host.view.show_error(
                "Class Required",
                "The class list is not valid for this project.",
                next_action="Restore every class already in use, then draw the box again.",
                details=str(exc),
            )
            return

        class_id = host.view.canvas_area.active_class_combo.currentIndex()
        if class_id < 0 or class_id >= len(prompts):
            host.view.show_error(
                "Class Required",
                "Choose an active class beside the Box tool before drawing.",
            )
            return

        capture = host.view.history.capture_edit("Add annotation")
        try:
            annotation = add_manual_annotation(
                image,
                class_id,
                prompts[class_id],
                box_xyxy,
            )
            host.view.history.commit_edit(capture, annotation.id)
            self.after_annotation_change(annotation.id)
            host.view.set_message("Manual annotation added.")
        except Exception as exc:
            host.presentation.report_error(
                "Could Not Add Annotation",
                "The bounding box could not be added.",
                "Draw a box fully inside the image and retry.",
                exc,
            )

    def canvas_box_changed(self, annotation_id, box_xyxy):
        host = self.host
        image = host.current_image
        annotation = image.annotation_by_id(annotation_id) if image else None
        if annotation is None:
            return
        if all(abs(old - new) < 0.5 for old, new in zip(annotation.box_xyxy, box_xyxy)):
            return
        capture = host.view.history.capture_edit("Edit box")
        try:
            edit_annotation_box(image, annotation_id, box_xyxy)
            host.view.history.commit_edit(capture, annotation_id)
            self.after_annotation_change(annotation_id)
            host.view.set_message(
                "Box updated. Re-segment it before segmentation export."
            )
        except Exception as exc:
            host.presentation.report_error(
                "Could Not Update Box",
                "The bounding box could not be updated.",
                "Check the coordinates and keep the box inside the image.",
                exc,
            )

    def apply_box_fields(self):
        host = self.host
        annotation = host.selected_annotation
        image = host.current_image
        if annotation is None or image is None:
            return
        try:
            values = host.view.annotation.box_values()
            if not host.presentation.box_fields_changed(annotation, values):
                return
            capture = host.view.history.capture_edit("Edit box")
            edit_annotation_box(image, annotation.id, values)
            host.view.history.commit_edit(capture, annotation.id)
            self.after_annotation_change(annotation.id)
            host.view.set_message(
                "Box updated. Re-segment it before segmentation export."
            )
        except Exception as exc:
            host.presentation.report_error(
                "Invalid Box Coordinates",
                "The box coordinates are not valid.",
                "Use x1 < x2 and y1 < y2 within the image bounds.",
                exc,
            )

    def apply_selected_class(self):
        host = self.host
        annotation = host.selected_annotation
        image = host.current_image
        class_id = host.view.annotation.class_combo.currentIndex()
        class_name = host.view.annotation.class_combo.currentText().strip()
        if annotation is None or image is None or class_id < 0 or not class_name:
            return
        if annotation.class_id == class_id and annotation.class_name == class_name:
            return
        capture = host.view.history.capture_edit("Change class")
        try:
            change_annotation_class(image, annotation.id, class_id, class_name)
            host.view.history.commit_edit(capture, annotation.id)
            self.after_annotation_change(annotation.id)
            host.view.set_message(
                "Class updated. Re-segment it before segmentation export."
            )
        except Exception as exc:
            host.presentation.report_error(
                "Could Not Change Class",
                "The selected class could not be applied.",
                "Check the class list in Setup and retry.",
                exc,
            )

    def delete_selected(self):
        host = self.host
        annotation = host.selected_annotation
        image = host.current_image
        if annotation is None or image is None:
            return
        capture = host.view.history.capture_edit("Delete annotation")
        delete_annotation(image, annotation.id)
        host.view.history.commit_edit(capture)
        self.after_annotation_change()
        host.view.set_message("Annotation deleted. Use Undo to restore it.")

    def reset_selected(self):
        host = self.host
        annotation = host.selected_annotation
        image = host.current_image
        if annotation is None or image is None or not annotation.is_modified_from_sam3:
            return
        capture = host.view.history.capture_edit("Reset annotation")
        try:
            reset_annotation_to_sam3(image, annotation.id)
            host.view.history.commit_edit(capture, annotation.id)
            self.after_annotation_change(annotation.id)
            host.view.set_message("Original SAM3 annotation restored.")
        except Exception as exc:
            host.presentation.report_error(
                "Could Not Reset Annotation",
                "This annotation has no restorable SAM3 geometry.",
                "Keep the current edit or run SAM3 again on the image.",
                exc,
            )

    def mark_current_reviewed(self):
        host = self.host
        image = host.current_image
        if image is None or image.status == ImageStatus.REVIEWED:
            return False
        mark_image_reviewed(image)
        host.presentation.mark_dirty(refresh=False)
        host.view.dataset.refresh(image.image_index)
        host.view.set_message("Image marked as reviewed.")
        host.presentation.dataset_filter_changed()
        return True

    def review_current_and_select_next(self):
        host = self.host
        image = host.current_image
        if image is None:
            return
        reviewed_index = image.image_index
        if not self.mark_current_reviewed():
            return
        if host.current_image_index != reviewed_index:
            return
        if host.view.actions.next_image.isEnabled():
            host.view.dataset.select_relative(1)

    def after_annotation_change(self, select_id=None):
        host = self.host
        image = host.current_image
        host.presentation.mark_dirty(refresh=False, history_managed=True)
        if image is not None:
            host.view.dataset.refresh(image.image_index)
        self.render_current_annotations(select_id)
