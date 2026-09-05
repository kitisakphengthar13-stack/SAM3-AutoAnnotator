from __future__ import annotations

import logging
from math import isclose
from pathlib import Path

from app_paths import discover_default_model
from domain import ImageStatus
from gui.controllers.state import UiMode
from services.project_service import (
    parse_prompts,
    remaining_prediction_targets,
)


logger = logging.getLogger(__name__)


class PresentationController:
    """Own dataset/image presentation, action policy, status, and GUI error reporting."""

    def __init__(self, host):
        self.host = host

    def show_initial_state(self):
        host = self.host
        model_path = discover_default_model()
        if model_path is not None:
            host.view.setup.model_path_edit.setText(str(model_path))
        host.view.dataset.clear()
        host.view.canvas.clear_image()
        host.view.show_canvas(False)
        host.view.results.set_status("No export yet")
        host.view.set_project_title("No project loaded")
        self.update_actions()
        self.update_context()

    def select_image(self, image_index):
        host = self.host
        if host.project is None:
            return
        host.current_image_index = int(image_index)
        host.selected_annotation_id = None
        self.load_current_image()

    def dataset_filter_changed(self):
        host = self.host
        image = host.current_image
        current_visible = False
        if image is not None:
            current_visible = host.view.dataset.filter_model.index_for_image_index(
                image.image_index
            ).isValid()
            if (
                current_visible
                and host.view.dataset.selected_image_index() != image.image_index
            ):
                host.view.dataset.select_image(image.image_index, notify=False)
        if image is not None and not current_visible:
            if host.view.dataset.filter_model.rowCount() > 0:
                host.view.dataset.select_first()
                return
            host.view.canvas_area.canvas_hint.setText(
                f"{image.image_name} is hidden by the current Dataset filter."
            )
        elif image is not None:
            host.annotations.update_canvas_hint()
        self.update_actions()
        self.update_context()

    def load_current_image(self):
        host = self.host
        image = host.current_image
        if image is None:
            host._current_image_loaded = False
            host.view.canvas.clear_image()
            host.view.show_canvas(False)
            self.update_actions()
            return
        host._current_image_loaded = False
        try:
            width, height = host.view.canvas.load_image(image.image_path)
            host._current_image_loaded = True
            host.view.show_canvas(True)
            if image.width is None or image.height is None:
                image.width, image.height = width, height
                self.mark_dirty(refresh=False)
            host.annotations.render_current_annotations()
            host.view.set_message(
                "Run SAM3, draw a box, or select an annotation to edit."
            )
            return
        except Exception as exc:
            image.mark_error(exc)
            self.mark_dirty(refresh=False)
            host.view.actions.select_tool.setChecked(True)
            host.view.dataset.refresh(image.image_index)
            host.view.annotation.set_annotations([])
            host.annotations.clear_selection()
            host.view.canvas_area.canvas_hint.setText(
                f"Could not display {image.image_name}. Select another image or retry."
            )
            host.view.show_canvas_error(image.image_name, str(exc))
            self.report_error(
                "Could Not Load Image",
                "The selected image could not be displayed.",
                "Check the image file and select another image if it is damaged.",
                exc,
            )
        self.update_actions()
        self.update_context()

    def mark_dirty(self, *, refresh=True, history_managed=False):
        host = self.host
        host.dirty = True
        if not history_managed:
            host.view.history.mark_external_dirty()
        if refresh:
            self.update_actions()
            self.update_context()

    @staticmethod
    def box_fields_changed(annotation, values):
        return not all(
            isclose(old, new, rel_tol=0.0, abs_tol=0.005)
            for old, new in zip(annotation.box_xyxy, values)
        )

    def box_editor_changed(self, annotation):
        if annotation is None:
            return False
        try:
            return self.box_fields_changed(
                annotation,
                self.host.view.annotation.box_values(),
            )
        except (TypeError, ValueError):
            return True

    def update_actions(self):
        host = self.host
        actions = host.view.actions
        idle = host.mode in {UiMode.EMPTY, UiMode.READY}
        project_open = host.project is not None
        ready = host.mode == UiMode.READY and project_open
        image = host.current_image
        annotation = host.selected_annotation
        prompts = parse_prompts(host.view.setup.prompts_text())
        prompt_error = host.projects.prompt_validation_error(prompts)
        settings_valid = prompt_error is None
        model_text = host.view.setup.model_path_edit.text().strip()
        model_ok = bool(model_text) and Path(model_text).is_file()
        model_error = None
        if model_text and not model_ok:
            model_error = "Model file was not found. Select an existing local file."
        image_ready = ready and image is not None and host._current_image_loaded
        targets = remaining_prediction_targets(host.project) if project_open else []
        pending_text = f"Run Pending ({len(targets)})"
        if actions.run_remaining.text() != pending_text:
            actions.run_remaining.setText(pending_text)

        visible_count = host.view.dataset.filter_model.rowCount()
        visible_row = host.view.dataset.selected_visible_row()
        class_changed = bool(
            annotation is not None
            and (
                host.view.annotation.class_combo.currentIndex()
                != annotation.class_id
                or host.view.annotation.class_combo.currentText().strip()
                != annotation.class_name
            )
        )
        box_changed = self.box_editor_changed(annotation)

        host.view.setup.set_prompt_error(prompt_error)
        host.view.setup.set_model_error(model_error)

        for action in (actions.open_image, actions.open_folder, actions.open_state):
            action.setEnabled(idle)
        actions.import_yolo.setEnabled(ready and settings_valid)
        output_text = host.view.setup.output_dir_edit.text().strip()
        output_path = Path(output_text) if output_text else None
        destination_changed = (
            output_path is not None
            and host._saved_output_dir is not None
            and output_path.resolve() != host._saved_output_dir.resolve()
        )
        actions.save.setEnabled(
            ready
            and settings_valid
            and (
                host.dirty
                or host.current_state_path is None
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
            bool(host.last_preview_path) and Path(host.last_preview_path).is_file()
        )
        actions.open_output.setEnabled(
            project_open and bool(output_text) and Path(output_text).is_dir()
        )
        actions.cancel_batch.setEnabled(
            host.mode == UiMode.BATCH and host.tasks.is_running
        )

        host.view.setup.set_settings_enabled(idle, project_open=project_open)
        host.view.dataset.image_list.setEnabled(ready)
        host.view.annotation.annotation_table.setEnabled(image_ready)
        host.view.canvas.setEnabled(image_ready)
        self.set_detail_fields_enabled(
            image_ready and settings_valid and annotation is not None
        )

    def set_detail_fields_enabled(self, enabled):
        view = self.host.view
        for widget in (
            view.annotation.class_combo,
            view.annotation.x1_edit,
            view.annotation.y1_edit,
            view.annotation.x2_edit,
            view.annotation.y2_edit,
        ):
            widget.setEnabled(enabled)

    def update_context(self):
        host = self.host
        image = host.current_image
        project_title = (
            "No project loaded"
            if host.project is None
            else host.project.project_name or "Current project"
        )
        host.view.set_project_title(
            f"{project_title} *" if host.dirty else project_title
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
        state = "unsaved" if host.dirty else "saved"
        host.view.set_status_context(
            f"{image_text} | {count} annotations | {state}"
        )

    def report_error(self, title, message, next_action, exc):
        host = self.host
        logger.exception("%s: %s", title, exc)
        details = f"{type(exc).__name__}: {exc}"
        if host.view.diagnostic_log_path:
            details += f"\n\nDiagnostic log: {host.view.diagnostic_log_path}"
        host.view.set_message(title)
        host.view.show_error(
            title,
            message,
            next_action=next_action,
            details=details,
        )
