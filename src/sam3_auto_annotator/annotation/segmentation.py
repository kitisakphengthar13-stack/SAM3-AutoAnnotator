from collections import Counter

def has_valid_segmentation(annotation):
    return (
        annotation.is_active
        and bool(annotation.segmentation_valid)
        and bool(annotation.polygon_xyn)
    )


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
    values = []
    for x_norm, y_norm in polygon_xyn or []:
        values.append(f"{min(max(float(x_norm), 0.0), 1.0):.6f}")
        values.append(f"{min(max(float(y_norm), 0.0), 1.0):.6f}")
    if len(values) < 6:
        raise ValueError("YOLO segmentation export requires at least three polygon points.")
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
