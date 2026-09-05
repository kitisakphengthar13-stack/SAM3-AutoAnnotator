from __future__ import annotations

from pathlib import Path

from sam3_auto_annotator.app_paths import discover_default_model
from sam3_auto_annotator.gui.controller import UiMode
from sam3_auto_annotator.services.project_service import (
    default_output_dir,
    import_yolo_project,
)


class ProjectController:
    """Own project activation and project-level label import workflows."""

    def __init__(self, host):
        self.host = host

    def load_project(self, project, state_path=None):
        host = self.host
        if not project.model_path:
            default_model = discover_default_model()
            if default_model is not None:
                project.model_path = str(default_model)
        output_dir = Path(state_path).parent if state_path else default_output_dir(project)

        host.project = project
        host.current_state_path = state_path
        host._saved_output_dir = Path(state_path).parent if state_path else None
        host.last_export_result = None
        host.last_preview_path = None
        host.current_image_index = None
        host.selected_annotation_id = None
        host._current_image_loaded = False
        host.dirty = False
        host.mode = UiMode.READY

        host.view.actions.select_tool.setChecked(True)
        host.view.show_canvas(False)
        host.view.task_progress.hide_when_idle()
        if not project.images:
            host.view.canvas.clear_image()
            host.view.annotation.set_annotations([])
            host._clear_annotation_selection()
            host.view.canvas_area.canvas_hint.setText(
                "This project does not contain any images."
            )

        host._rendering = True
        try:
            host.view.setup.set_project(project, output_dir)
            host.view.annotation.set_classes(project.prompts)
            host.view.dataset.set_images(project.images, project.project_name)
        finally:
            host._rendering = False

        host.view.set_project_title(project.project_name or "Current project")
        host.view.results.reset(output_dir)
        host.view.dataset.select_first()
        host._update_actions()
        host._update_context()

    def import_yolo(self):
        host = self.host
        if host.project is None:
            return
        label_dir = host.view.choose_folder(
            "Select YOLO Detection Label Folder",
            host._last_directory(),
        )
        if not label_dir:
            return
        try:
            summary = import_yolo_project(host.project, label_dir)
            host._remember_path(label_dir)
            host._rendering = True
            try:
                host.view.setup.set_prompts(host.project.prompts)
                host.view.annotation.set_classes(host.project.prompts)
            finally:
                host._rendering = False
            host._mark_dirty(refresh=False)
            host.view.dataset.refresh()
            host._render_current_annotations()
            host.view.show_review()
            message = summary.to_message()
            host.view.results.set_status(message)
            host.view.set_message(message)
        except Exception as exc:
            host._report_error(
                "Could Not Import YOLO Labels",
                "The labels could not be imported into this project.",
                "Check the label folder and YOLO detection format, then retry.",
                exc,
            )
