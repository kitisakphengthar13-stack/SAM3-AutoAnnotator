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
from storage.source_fingerprint import fingerprint_file
from storage.yolo_importer import import_yolo_detection_labels


def parse_prompts(text):
    prompts = []
    for part in re.split(r"[\n,]+", text):
        prompt = part.strip()
        if prompt and prompt not in prompts:
            prompts.append(prompt)
    return prompts


def _record_source_fingerprint(image, fingerprint):
    image.source_size_bytes = fingerprint.size_bytes
    image.source_mtime_ns = fingerprint.mtime_ns
    image.source_sha256 = fingerprint.sha256


def initialize_source_integrity(project_state):
    """Capture immutable source baselines for newly created or legacy projects."""
    for image in project_state.images:
        actual_width, actual_height = read_image_size(image.image_path)
        if image.width is not None and image.height is not None:
            if (image.width, image.height) != (actual_width, actual_height):
                raise ValueError(
                    f"Source image dimensions changed for {image.image_path}: "
                    f"project has {image.width}x{image.height}, file is "
                    f"{actual_width}x{actual_height}."
                )
        else:
            image.width, image.height = actual_width, actual_height
        if image.source_sha256 is None:
            _record_source_fingerprint(image, fingerprint_file(image.image_path))
    return project_state


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
    project = ProjectState.from_image_paths(
        input_path=input_path,
        image_paths=image_paths,
        prompts=prompts,
        model_path=model_path,
        project_name=project_name,
        confidence=confidence,
        half=half,
    )
    return initialize_source_integrity(project)


def default_output_dir(project_state):
    project_name = project_state.project_name or "sam3_annotations"
    return OUTPUTS_DIR / sanitize_name(project_name)


def save_state_to_output(project_state, output_dir):
    output_dir = Path(output_dir)
    previous_name = project_state.project_name
    project_state.project_name = output_dir.name
    try:
        verify_source_image_sizes(project_state)
        return save_project_state(project_state, output_dir / STATE_FILE_NAME)
    except Exception:
        project_state.project_name = previous_name
        raise


def load_state(path):
    project = load_project_state(path)
    return initialize_source_integrity(project)


def ensure_image_sizes(project_state):
    for image in project_state.images:
        if image.width is None or image.height is None:
            image.width, image.height = read_image_size(image.image_path)
    return project_state


def verify_source_image_sizes(project_state):
    """Reject stale geometry or changed source content before save/export."""
    for image in project_state.images:
        actual_width, actual_height = read_image_size(image.image_path)
        if image.width is None or image.height is None:
            image.width, image.height = actual_width, actual_height
        elif (image.width, image.height) != (actual_width, actual_height):
            raise ValueError(
                f"Source image dimensions changed for {image.image_path}: "
                f"project has {image.width}x{image.height}, file is "
                f"{actual_width}x{actual_height}. Re-open the source as a new project "
                "or restore the original image before saving or exporting."
            )

        current = fingerprint_file(image.image_path)
        if image.source_sha256 is None:
            _record_source_fingerprint(image, current)
            continue
        if current.sha256 != image.source_sha256:
            raise ValueError(
                f"Source image content changed for {image.image_path} even though its "
                "dimensions may be unchanged. Restore the original source image or "
                "open the changed file as a new project before saving or exporting."
            )
        # Content is identical; refresh metadata in case the same bytes were copied again.
        image.source_size_bytes = current.size_bytes
        image.source_mtime_ns = current.mtime_ns
    return project_state


def import_yolo_project(project_state, label_dir):
    ensure_image_sizes(project_state)
    return import_yolo_detection_labels(project_state, label_dir)


def export_project(project_state, output_dir):
    verify_source_image_sizes(project_state)
    return export_corrected_detection(project_state, output_dir)


def remaining_prediction_targets(project_state):
    return [
        image
        for image in project_state.images
        if image.status in {ImageStatus.NOT_PREDICTED, ImageStatus.ERROR}
    ]
