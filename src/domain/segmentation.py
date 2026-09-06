from collections import Counter
from math import isfinite


def polygon_point_count(annotation):
    return len(annotation.polygon_xyn or [])


def validate_polygon_xyn(polygon_xyn, *, require_three=True):
    if polygon_xyn is None:
        raise ValueError("Segmentation polygon is missing.")
    points = []
    for point in polygon_xyn:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Every segmentation point must contain x and y values.")
        x, y = float(point[0]), float(point[1])
        if not isfinite(x) or not isfinite(y):
            raise ValueError("Segmentation polygon coordinates must be finite.")
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError("Segmentation polygon coordinates must be normalized to [0, 1].")
        points.append([x, y])
    if require_three and len(points) < 3:
        raise ValueError("Segmentation polygon requires at least three points.")
    if require_three and len({(x, y) for x, y in points}) < 3:
        raise ValueError("Segmentation polygon requires at least three distinct points.")
    return points


def has_valid_segmentation(annotation):
    if not annotation.is_active or not bool(annotation.segmentation_valid):
        return False
    try:
        validate_polygon_xyn(annotation.polygon_xyn)
    except (TypeError, ValueError):
        return False
    return True


def segmentation_status(annotation):
    if not annotation.is_active:
        return "none"
    if has_valid_segmentation(annotation):
        return "valid"
    if not annotation.polygon_xyn:
        return "none"
    try:
        validate_polygon_xyn(annotation.polygon_xyn)
    except (TypeError, ValueError):
        return "invalid"
    if not annotation.segmentation_valid:
        source = getattr(annotation, "source", None)
        source_value = getattr(source, "value", source)
        if source_value == "edited":
            return "stale"
        return "invalid"
    return "invalid"


def segmentation_status_text(annotation):
    status = segmentation_status(annotation)
    if status == "stale":
        return "Segmentation: stale - click Re-segment"
    return f"Segmentation: {status}"


def segmentation_skip_reason(annotation):
    status = segmentation_status(annotation)
    if status == "valid":
        return None
    if not annotation.polygon_xyn:
        return "no polygon"
    if polygon_point_count(annotation) < 3:
        return "polygon has too few points"
    try:
        validate_polygon_xyn(annotation.polygon_xyn)
    except (TypeError, ValueError) as exc:
        return str(exc)
    if status == "stale":
        return "segmentation stale after bbox/class edit"
    if not annotation.segmentation_valid:
        return "segmentation invalid"
    return "unknown invalid segmentation"


def polygon_xyn_to_pixels(polygon_xyn, image_width, image_height):
    width = float(image_width)
    height = float(image_height)
    points = []
    for x_norm, y_norm in polygon_xyn or []:
        x = min(max(float(x_norm), 0.0), 1.0) * width
        y = min(max(float(y_norm), 0.0), 1.0) * height
        points.append((x, y))
    return points


def build_yolo_segmentation_line(class_id, polygon_xyn):
    points = validate_polygon_xyn(polygon_xyn)
    values = []
    for x_norm, y_norm in points:
        values.append(f"{x_norm:.6f}")
        values.append(f"{y_norm:.6f}")
    return f"{int(class_id)} {' '.join(values)}"


def build_segmentation_rows(project_state):
    total_class_counts = Counter()
    for image in project_state.images:
        total_class_counts.update(
            annotation.class_name
            for annotation in image.active_annotations
            if has_valid_segmentation(annotation)
        )

    rows = []
    for image in sorted(project_state.images, key=lambda item: item.image_index):
        image_segmentations = [
            annotation
            for annotation in image.active_annotations
            if has_valid_segmentation(annotation)
        ]
        image_class_counts = Counter(annotation.class_name for annotation in image_segmentations)
        for object_index, annotation in enumerate(image_segmentations):
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
                    "polygon_point_count": len(annotation.polygon_xyn or []),
                    "polygon_xyn": annotation.polygon_xyn,
                    "yolo_segmentation_line": build_yolo_segmentation_line(
                        annotation.class_id,
                        annotation.polygon_xyn,
                    ),
                    "confidence": "" if annotation.confidence is None else f"{annotation.confidence:.6f}",
                }
            )
    return rows


def build_skipped_segmentation_rows(project_state):
    rows = []
    for image in sorted(project_state.images, key=lambda item: item.image_index):
        for object_index, annotation in enumerate(image.active_annotations):
            reason = segmentation_skip_reason(annotation)
            if reason is None:
                continue
            rows.append(
                {
                    "image_path": image.image_path,
                    "image_name": image.image_name,
                    "image_index": image.image_index,
                    "object_index": object_index,
                    "annotation_id": annotation.id,
                    "class_id": annotation.class_id,
                    "class_name": annotation.class_name,
                    "segmentation_status": segmentation_status(annotation),
                    "reason": reason,
                }
            )
    return rows
