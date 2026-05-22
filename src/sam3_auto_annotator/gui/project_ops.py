import re
from pathlib import Path

from sam3_auto_annotator.annotation.export import export_corrected_detection
from sam3_auto_annotator.annotation.models import ImageStatus, ProjectState
from sam3_auto_annotator.annotation.store import STATE_FILE_NAME, load_project_state, save_project_state
from sam3_auto_annotator.paths import find_images, sanitize_name


def parse_prompts(text):
    prompts = [part.strip() for part in re.split(r"[\n,]+", text) if part.strip()]
    return prompts


def create_project(input_path, prompts, model_path=None, project_name=None):
    image_paths = find_images(input_path)
    if not prompts:
        prompts = ["object"]
    if project_name is None:
        project_name = sanitize_name(f"{Path(input_path).stem or Path(input_path).name}_gui")
    return ProjectState.from_image_paths(
        input_path=input_path,
        image_paths=image_paths,
        prompts=prompts,
        model_path=model_path,
        project_name=project_name,
    )


def default_output_dir(project_state):
    project_name = project_state.project_name or "sam3_gui_project"
    return Path("outputs") / sanitize_name(project_name)


def save_state_to_output(project_state, output_dir):
    output_dir = Path(output_dir)
    project_state.project_name = output_dir.name
    return save_project_state(project_state, output_dir / STATE_FILE_NAME)


def load_state(path):
    return load_project_state(path)


def export_project(project_state, output_dir):
    return export_corrected_detection(project_state, output_dir)


def remaining_prediction_targets(project_state):
    return [
        image
        for image in project_state.images
        if image.status in {ImageStatus.NOT_PREDICTED, ImageStatus.ERROR}
    ]
