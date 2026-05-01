import re
from datetime import datetime
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
XYN_CSV_NAME = "sam3_auto_annotation_xyn_outputs.csv"
BOX_CSV_NAME = "sam3_auto_annotation_box_outputs.csv"
EXPORT_FORMATS = ["csv", "yolo"]


def normalize_export_formats(export_formats):
    normalized = []
    for export_format in export_formats:
        if export_format not in normalized:
            normalized.append(export_format)
    return normalized


def sanitize_name(value):
    name = str(value).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_") or "sam3_auto_annotation"


def timestamp_suffix():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_project_name(input_path, prompts):
    path = Path(input_path)
    input_name = path.stem if path.is_file() else path.name
    prompt_name = "_".join(sanitize_name(prompt) for prompt in prompts)
    return sanitize_name(f"{input_name}_{prompt_name}_auto_annotation")


def create_project_output_dir(output_root, project_name, force_timestamp=False, overwrite=False):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    sanitized_project_name = sanitize_name(project_name)
    folder_name = sanitized_project_name
    if force_timestamp:
        folder_name = f"{sanitized_project_name}_{timestamp_suffix()}"

    output_dir = root / folder_name
    if output_dir.exists() and not overwrite:
        output_dir = root / f"{sanitized_project_name}_{timestamp_suffix()}"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, sanitized_project_name


def find_images(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Input file is not a supported image: {path}")
        return [path]

    if path.is_dir():
        images = [
            item
            for item in sorted(path.iterdir())
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not images:
            raise ValueError(f"No supported images found in folder: {path}")
        return images

    raise FileNotFoundError(f"Input path does not exist: {path}")


def validate_model_path(model_path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"SAM3 model path does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"SAM3 model path is not a file: {path}")
    return path

