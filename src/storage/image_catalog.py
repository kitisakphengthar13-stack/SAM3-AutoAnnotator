"""Discover local inputs and build safe output paths."""

import hashlib
import re
import unicodedata
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
BOX_CSV_NAME = "sam3_auto_annotation_box_outputs.csv"
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def sanitize_name(value):
    raw = str(value).strip()
    normalized = unicodedata.normalize("NFKC", raw).casefold()
    name = re.sub(r"[^\w]+", "_", normalized, flags=re.UNICODE).strip("_")
    if not name:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
        name = f"sam3_auto_annotation_{digest}"
    if name.casefold() in _WINDOWS_RESERVED:
        name = f"_{name}"
    return name[:96]


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
        _validate_unique_stems(images)
        return images

    raise FileNotFoundError(f"Input path does not exist: {path}")


def _validate_unique_stems(image_paths):
    by_stem = {}
    for image_path in image_paths:
        by_stem.setdefault(image_path.stem.casefold(), []).append(image_path.name)
    collisions = [names for names in by_stem.values() if len(names) > 1]
    if collisions:
        examples = "; ".join(", ".join(names) for names in collisions[:3])
        raise ValueError(
            "Images must have unique filename stems for YOLO export. "
            f"Rename the colliding files first: {examples}"
        )


def validate_model_path(model_path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"SAM3 model path does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"SAM3 model path is not a file: {path}")
    return path
