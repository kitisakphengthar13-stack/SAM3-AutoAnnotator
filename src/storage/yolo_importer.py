from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from domain.geometry import clip_xyxy, validate_image_size
from domain import Annotation, AnnotationSource, ImageStatus


SKIP_IMPORT_STATUSES = {ImageStatus.EDITED, ImageStatus.REVIEWED}
MAX_YOLO_CLASS_ID = 10_000


@dataclass
class YoloImportSummary:
    processed_images: int = 0
    imported_images: int = 0
    no_detection_images: int = 0
    missing_label_files: int = 0
    skipped_images: int = 0
    imported_boxes: int = 0
    invalid_lines: int = 0
    added_classes: int = 0

    def to_message(self):
        return (
            f"Imported YOLO labels: {self.imported_boxes} boxes, "
            f"{self.imported_images} images with labels, "
            f"{self.no_detection_images} empty labels, "
            f"{self.missing_label_files} missing files, "
            f"{self.skipped_images} skipped, "
            f"{self.added_classes} generated classes, "
            f"{self.invalid_lines} invalid lines."
        )


def class_name_for_id(class_id, prompts):
    if 0 <= class_id < len(prompts):
        return prompts[class_id]
    return f"class_{class_id}"


def extend_prompts_for_class_ids(prompts, class_ids):
    class_ids = list(class_ids)
    if not class_ids:
        return []
    highest_id = max(class_ids)
    if highest_id < 0:
        raise ValueError("Class id must be non-negative.")
    if highest_id > MAX_YOLO_CLASS_ID:
        raise ValueError(
            f"Class id {highest_id} exceeds the supported import limit "
            f"of {MAX_YOLO_CLASS_ID}."
        )

    added = []
    while len(prompts) <= highest_id:
        slot = len(prompts)
        base = f"class_{slot}"
        candidate = base
        suffix = 2
        while candidate in prompts:
            candidate = f"{base}_{suffix}"
            suffix += 1
        prompts.append(candidate)
        added.append(candidate)
    return added


def yolo_xywhn_to_xyxy(x_center, y_center, width, height, image_width, image_height):
    validate_image_size(image_width, image_height)
    x_center_abs = float(x_center) * float(image_width)
    y_center_abs = float(y_center) * float(image_height)
    width_abs = float(width) * float(image_width)
    height_abs = float(height) * float(image_height)
    return clip_xyxy(
        (
            x_center_abs - width_abs / 2.0,
            y_center_abs - height_abs / 2.0,
            x_center_abs + width_abs / 2.0,
            y_center_abs + height_abs / 2.0,
        ),
        image_width,
        image_height,
    )


def parse_yolo_detection_line(line, image_width, image_height, prompts):
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError("YOLO detection line must have 5 values.")

    class_id = int(parts[0])
    values = [float(value) for value in parts[1:]]
    if class_id < 0 or class_id > MAX_YOLO_CLASS_ID:
        raise ValueError(
            f"Class id must be between 0 and {MAX_YOLO_CLASS_ID}."
        )
    if not all(isfinite(value) for value in values):
        raise ValueError("YOLO coordinates must be finite numbers.")
    x_center, y_center, width, height = values
    if not 0.0 <= x_center <= 1.0 or not 0.0 <= y_center <= 1.0:
        raise ValueError("YOLO box centers must be normalized to [0, 1].")
    if not 0.0 < width <= 1.0 or not 0.0 < height <= 1.0:
        raise ValueError("YOLO box width and height must be in (0, 1].")

    box_xyxy = yolo_xywhn_to_xyxy(*values, image_width=image_width, image_height=image_height)
    return Annotation(
        class_id=class_id,
        class_name=class_name_for_id(class_id, prompts),
        box_xyxy=box_xyxy,
        source=AnnotationSource.IMPORTED,
    )


def annotations_from_yolo_file(label_path, image_width, image_height, prompts):
    annotations = []
    invalid_lines = 0
    text = Path(label_path).read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            annotations.append(
                parse_yolo_detection_line(line, image_width, image_height, prompts)
            )
        except (TypeError, ValueError):
            invalid_lines += 1
    return annotations, invalid_lines


def _import_yolo_detection_labels(project_state, label_dir):
    label_dir = Path(label_dir)
    summary = YoloImportSummary()
    prompts = project_state.prompts

    for image in project_state.images:
        if image.status in SKIP_IMPORT_STATUSES:
            summary.skipped_images += 1
            continue

        summary.processed_images += 1
        label_path = label_dir / f"{Path(image.image_path).stem}.txt"
        if not label_path.exists():
            summary.missing_label_files += 1
            continue

        text = label_path.read_text(encoding="utf-8")
        if not text.strip():
            image.annotations = []
            image.status = ImageStatus.NO_DETECTION
            image.error_message = None
            summary.no_detection_images += 1
            continue

        if image.width is None or image.height is None:
            raise ValueError(f"Image dimensions are required before importing labels: {image.image_path}")

        annotations, invalid_lines = annotations_from_yolo_file(
            label_path,
            image_width=image.width,
            image_height=image.height,
            prompts=prompts,
        )
        summary.invalid_lines += invalid_lines
        if not annotations:
            # A non-empty file with no valid rows is malformed, not an assertion
            # that the image contains no objects. Preserve the previous state.
            continue

        added = extend_prompts_for_class_ids(
            prompts, (annotation.class_id for annotation in annotations)
        )
        summary.added_classes += len(added)
        for annotation in annotations:
            annotation.class_name = prompts[annotation.class_id]
        image.annotations = annotations
        image.status = ImageStatus.PREDICTED
        image.error_message = None
        summary.imported_boxes += len(annotations)
        summary.imported_images += 1

    return summary


def import_yolo_detection_labels(project_state, label_dir):
    """Import labels transactionally: failures never leave a half-mutated project."""
    working = deepcopy(project_state)
    summary = _import_yolo_detection_labels(working, label_dir)
    project_state.prompts = working.prompts
    project_state.images = working.images
    return summary
