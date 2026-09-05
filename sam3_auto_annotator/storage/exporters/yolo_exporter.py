"""YOLO detection and segmentation label writer."""

from pathlib import Path


def build_yolo_detection_line(row):
    values = [
        row.get("x_center_norm", ""),
        row.get("y_center_norm", ""),
        row.get("width_norm", ""),
        row.get("height_norm", ""),
    ]
    if any(value == "" for value in values):
        raise ValueError(
            "Cannot build YOLO detection label because normalized box values are missing "
            f"for image {row.get('image_name')} object {row.get('object_index')}."
        )

    return "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
        int(row["class_id"]),
        *(float(value) for value in values),
    )


def _image_key(path):
    return str(Path(path).resolve()).casefold()


def group_rows_by_image(rows):
    grouped = {}
    for row in sorted(
        rows,
        key=lambda item: (int(item["image_index"]), int(item["object_index"])),
    ):
        grouped.setdefault(_image_key(row["image_path"]), []).append(row)
    return grouped


def write_yolo_labels(output_dir, image_paths, xyn_rows, box_rows):
    output_dir = Path(output_dir)
    image_paths = [Path(path) for path in image_paths]
    stems = [path.stem.casefold() for path in image_paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Cannot export YOLO labels because image filename stems are not unique.")
    yolo_root = output_dir / "yolo_labels"
    segmentation_dir = yolo_root / "segmentation"
    detection_dir = yolo_root / "detection"
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    detection_dir.mkdir(parents=True, exist_ok=True)

    xyn_rows_by_image = group_rows_by_image(xyn_rows)
    box_rows_by_image = group_rows_by_image(box_rows)

    for image_path in image_paths:
        image_key = _image_key(image_path)
        segmentation_lines = [
            row["yolo_segmentation_line"] for row in xyn_rows_by_image.get(image_key, [])
        ]
        detection_lines = [
            build_yolo_detection_line(row) for row in box_rows_by_image.get(image_key, [])
        ]

        segmentation_path = segmentation_dir / f"{image_path.stem}.txt"
        detection_path = detection_dir / f"{image_path.stem}.txt"
        segmentation_text = "\n".join(segmentation_lines)
        detection_text = "\n".join(detection_lines)
        segmentation_path.write_text(
            segmentation_text + ("\n" if segmentation_text else ""),
            encoding="utf-8",
        )
        detection_path.write_text(
            detection_text + ("\n" if detection_text else ""),
            encoding="utf-8",
        )

    return segmentation_dir, detection_dir
