import re
from pathlib import Path

from app_paths import OUTPUTS_DIR
from services.export_service import export_corrected_detection
from domain import ImageStatus, ProjectState
from storage.image_catalog import find_images, sanitize_name
from storage.image_reader import read_image_size
from storage.project_store import (
    STATE_FILE_NAME,
    load_project_state,
    save_project_state,
)
from storage.yolo_importer import import_yolo_detection_labels


def parse_prompts(text):
    prompts = []
    for part in re.split(r"[\n,]+", text):
        prompt = part.strip()
        if prompt and prompt not in prompts:
            prompts.append(prompt)
    return prompts


def create_project(
    input_path,
    prompts,
    model_path=None,
    project_name=None,
    confidence=0.5,
    half=True,
):
    image_paths = find_images(input_path)
    if project_name is None:
        input_name = Path(input_path).stem or Path(input_path).name
        project_name = sanitize_name(f"{input_name}_annotations")
    return ProjectState.from_image_paths(
        input_path=input_path,
        image_paths=image_paths,
        prompts=prompts,
        model_path=model_path,
        project_name=project_name,
        confidence=confidence,
        half=half,
    )


def default_output_dir(project_state):
    project_name = project_state.project_name or "sam3_annotations"
    return OUTPUTS_DIR / sanitize_name(project_name)


def save_state_to_output(project_state, output_dir):
    output_dir = Path(output_dir)
    project_state.project_name = output_dir.name
    return save_project_state(project_state, output_dir / STATE_FILE_NAME)


def load_state(path):
    return load_project_state(path)


def ensure_image_sizes(project_state):
    for image in project_state.images:
        if image.width is None or image.height is None:
            image.width, image.height = read_image_size(image.image_path)
    return project_state


def import_yolo_project(project_state, label_dir):
    ensure_image_sizes(project_state)
    return import_yolo_detection_labels(project_state, label_dir)


def export_project(project_state, output_dir):
    return export_corrected_detection(project_state, output_dir)


def remaining_prediction_targets(project_state):
    return [
        image
        for image in project_state.images
        if image.status in {ImageStatus.NOT_PREDICTED, ImageStatus.ERROR}
    ]
