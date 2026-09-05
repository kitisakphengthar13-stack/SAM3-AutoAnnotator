from dataclasses import dataclass
from pathlib import Path

from domain.geometry import clip_xyxy, validate_image_size
from domain import Annotation, AnnotationSource, ImageStatus


SKIP_IMPORT_STATUSES = {ImageStatus.EDITED, ImageStatus.REVIEWED}


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
    """Extend a positional class list so every imported YOLO id is addressable.

    YOLO stores only integer class ids.  Placeholder names fill any missing
    positions while remaining unique, preserving the invariant that
    ``prompts[annotation.class_id] == annotation.class_name``.
    """

    class_ids = list(class_ids)
    if not class_ids:
        return []
    highest_id = max(class_ids)
    if highest_id < 0:
        raise ValueError("Class id must be non-negative.")

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
    if class_id < 0:
        raise ValueError("Class id must be non-negative.")
    if any(value < 0 for value in values):
        raise ValueError("YOLO coordinates must be non-negative.")

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


def import_yolo_detection_labels(project_state, label_dir):
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
            image.status = ImageStatus.NOT_PREDICTED
            image.error_message = None
            summary.missing_label_files += 1
            continue

        if label_path.stat().st_size == 0 or not label_path.read_text(encoding="utf-8").strip():
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
        added = extend_prompts_for_class_ids(
            prompts, (annotation.class_id for annotation in annotations)
        )
        summary.added_classes += len(added)
        for annotation in annotations:
            annotation.class_name = prompts[annotation.class_id]
        summary.invalid_lines += invalid_lines
        image.annotations = annotations
        image.status = ImageStatus.PREDICTED if annotations else ImageStatus.NO_DETECTION
        image.error_message = None
        summary.imported_boxes += len(annotations)
        if annotations:
            summary.imported_images += 1
        else:
            summary.no_detection_images += 1

    return summary
