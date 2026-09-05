from __future__ import annotations

from pathlib import Path

from app_paths import MODELS_DIR, discover_default_model
from gui.controllers.state import UiMode
from services.project_service import (
    create_project,
    default_output_dir,
    import_yolo_project,
    load_state,
    parse_prompts,
    save_state_to_output,
)


class ProjectController:
    """Own project lifecycle, staged settings commits, saving, and label import."""

    def __init__(self, host):
        self.host = host

    def last_directory(self):
        return self.host.settings.last_directory()

    def remember_path(self, path):
        if not path:
            return
        candidate = Path(path)
        self.host.settings.set_last_directory(
            candidate if candidate.is_dir() else candidate.parent
        )

    def can_replace_project(self):
        host = self.host
        if not host.dirty:
            return True
        decision = host.view.ask_unsaved_changes()
        if decision == "save":
            self.save_project()
            return not host.dirty
        return decision == "discard"

    def open_image(self):
        host = self.host
        path = host.view.choose_image(self.last_directory())
        if not path or not self.can_replace_project():
            return
        self.remember_path(path)
        self.create_project(path)

    def open_folder(self):
        host = self.host
        path = host.view.choose_folder("Open Image Folder", self.last_directory())
        if not path or not self.can_replace_project():
            return
        self.remember_path(path)
        self.create_project(path)

    def create_project(self, input_path):
        host = self.host
        try:
            model_path = host.view.setup.model_path_edit.text().strip() or None
            project = create_project(
                input_path=input_path,
                prompts=parse_prompts(host.view.setup.prompts_text()),
                model_path=model_path,
                confidence=host.view.setup.conf_edit.value(),
                half=host.view.setup.half_check.isChecked(),
            )
            self.load_project(project)
            host.view.set_message(
                "Configure classes, then run SAM3 or draw boxes manually."
            )
        except Exception as exc:
            host._report_error(
                "Could Not Open Input",
                "The selected image or folder could not be opened.",
                "Check that it exists and contains supported image files, then retry.",
                exc,
            )

    def open_project(self):
        host = self.host
        path = host.view.choose_project(self.last_directory())
        if not path or not self.can_replace_project():
            return
        try:
            project = load_state(path)
            self.remember_path(path)
            self.load_project(project, state_path=Path(path))
            host.view.results.set_status("Project loaded. Continue reviewing or export.")
            host.view.set_message("Annotation project loaded.")
        except Exception as exc:
            host._report_error(
                "Could Not Open Project",
                "The annotation project could not be loaded.",
                "Choose a valid annotation_state.json file or restore a known-good copy.",
                exc,
            )

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

    def browse_model(self):
        host = self.host
        current = host.view.setup.model_path_edit.text().strip()
        start = str(Path(current).parent) if current else str(MODELS_DIR)
        path = host.view.choose_model(start)
        if path:
            self.remember_path(path)
            host.view.setup.model_path_edit.setText(path)

    def browse_output(self):
        host = self.host
        current = host.view.setup.output_dir_edit.text().strip()
        path = host.view.choose_folder(
            "Select Output Folder",
            current or self.last_directory(),
        )
        if path:
            self.remember_path(path)
            host.view.setup.output_dir_edit.setText(path)

    def settings_changed(self):
        host = self.host
        if host._rendering:
            return
        prompts = parse_prompts(host.view.setup.prompts_text())
        if host.project is not None:
            prompt_error = self.prompt_validation_error(prompts)
            host.view.annotation.set_classes(
                host.project.prompts if prompt_error else prompts
            )
            changed = self.apply_settings_if_valid(
                prompts,
                prompts_valid=prompt_error is None,
            )
            if changed:
                host._mark_dirty(refresh=False)
        else:
            host.view.annotation.set_classes(prompts)
        host._update_actions()
        host._update_context()

    def prompt_validation_error(self, prompts):
        project = self.host.project
        if project is None:
            return None
        used_names = {item.class_name for item in project.active_annotations()}
        missing = sorted(used_names.difference(prompts))
        if not missing:
            return None
        return (
            "Classes in use cannot be removed: "
            + ", ".join(missing)
            + ". Restore them or change those annotations first."
        )

    def apply_settings_if_valid(self, prompts, *, prompts_valid=None):
        host = self.host
        project = host.project
        if project is None:
            return False
        if prompts_valid is None:
            prompts_valid = self.prompt_validation_error(prompts) is None
        changed = False
        if prompts_valid and prompts != project.prompts:
            project.prompts = list(prompts)
            for annotation in project.active_annotations():
                annotation.class_id = project.class_map[annotation.class_name]
            changed = True

        model_path = host.view.setup.model_path_edit.text().strip() or None
        confidence = host.view.setup.conf_edit.value()
        half = host.view.setup.half_check.isChecked()
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

    def sync_project_settings(self, require_prompts=False):
        host = self.host
        if host.project is None:
            raise RuntimeError("No project is open.")
        prompts = parse_prompts(host.view.setup.prompts_text())
        if require_prompts and not prompts:
            raise ValueError("Enter at least one class prompt before running SAM3.")
        prompt_error = self.prompt_validation_error(prompts)
        if prompt_error:
            raise ValueError(prompt_error)
        self.apply_settings_if_valid(prompts, prompts_valid=True)
        return prompts

    def import_yolo(self):
        host = self.host
        if host.project is None:
            return
        label_dir = host.view.choose_folder(
            "Select YOLO Detection Label Folder",
            self.last_directory(),
        )
        if not label_dir:
            return
        try:
            summary = import_yolo_project(host.project, label_dir)
            self.remember_path(label_dir)
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

    def save_project(self):
        host = self.host
        if host.project is None:
            return
        try:
            self.sync_project_settings()
            output_dir = host._output_dir()
            path = save_state_to_output(host.project, output_dir)
            host.current_state_path = Path(path)
            host._saved_output_dir = Path(output_dir)
            host.dirty = False
            host.view.results.set_status("Project state saved.")
            host.view.results.set_output_dir(output_dir)
            host.view.set_message(f"Saved project to {path}")
            host._update_actions()
            host._update_context()
        except Exception as exc:
            host._report_error(
                "Could Not Save Project",
                "The annotation project could not be saved.",
                "Check the output folder permissions and available disk space, then retry.",
                exc,
            )
