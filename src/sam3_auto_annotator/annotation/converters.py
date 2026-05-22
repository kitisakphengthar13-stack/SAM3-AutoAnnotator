from collections import Counter
from pathlib import Path

from sam3_auto_annotator.annotation.geometry import clip_xyxy, xyxy_to_xywh, xyxy_to_yolo_xywhn


def _format_float(value):
    return f"{float(value):.6f}"


def _confidence_value(confidence):
    return "" if confidence is None else _format_float(confidence)


def image_paths_for_export(project_state):
    return [Path(image.image_path) for image in sorted(project_state.images, key=lambda item: item.image_index)]


def build_box_rows(project_state):
    total_class_counts = Counter()
    for image in project_state.images:
        total_class_counts.update(annotation.class_name for annotation in image.active_annotations)

    rows = []
    for image in sorted(project_state.images, key=lambda item: item.image_index):
        image_class_counts = Counter(
            annotation.class_name for annotation in image.active_annotations
        )
        if image.width is None or image.height is None:
            if image.active_annotations:
                raise ValueError(
                    f"Image dimensions are required for export: {image.image_path}"
                )
            continue

        for object_index, annotation in enumerate(image.active_annotations):
            box_xyxy = clip_xyxy(annotation.box_xyxy, image.width, image.height)
            x1, y1, x2, y2 = box_xyxy
            width = x2 - x1
            height = y2 - y1
            x_center, y_center, _, _ = xyxy_to_xywh(box_xyxy)
            x_center_norm, y_center_norm, width_norm, height_norm = xyxy_to_yolo_xywhn(
                box_xyxy,
                image_width=image.width,
                image_height=image.height,
                clip=False,
            )
            rows.append(
                {
                    "image_path": image.image_path,
                    "image_name": image.image_name,
                    "image_index": image.image_index,
                    "object_index": object_index,
                    "class_id": annotation.class_id,
                    "class_name": annotation.class_name,
                    "class_count_in_image": image_class_counts[annotation.class_name],
                    "total_class_count": total_class_counts[annotation.class_name],
                    "x1": _format_float(x1),
                    "y1": _format_float(y1),
                    "x2": _format_float(x2),
                    "y2": _format_float(y2),
                    "width": _format_float(width),
                    "height": _format_float(height),
                    "x_center": _format_float(x_center),
                    "y_center": _format_float(y_center),
                    "x_center_norm": _format_float(x_center_norm),
                    "y_center_norm": _format_float(y_center_norm),
                    "width_norm": _format_float(width_norm),
                    "height_norm": _format_float(height_norm),
                    "confidence": _confidence_value(annotation.confidence),
                }
            )
    return rows


def build_detection_export(project_state):
    return image_paths_for_export(project_state), build_box_rows(project_state)
