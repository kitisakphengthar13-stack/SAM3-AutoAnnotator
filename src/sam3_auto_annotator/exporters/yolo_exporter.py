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


def group_rows_by_image(rows):
    grouped = {}
    for row in sorted(
        rows,
        key=lambda item: (int(item["image_index"]), int(item["object_index"])),
    ):
        grouped.setdefault(int(row["image_index"]), []).append(row)
    return grouped


def write_yolo_labels(output_dir, image_paths, xyn_rows, box_rows):
    yolo_root = output_dir / "yolo_labels"
    segmentation_dir = yolo_root / "segmentation"
    detection_dir = yolo_root / "detection"
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    detection_dir.mkdir(parents=True, exist_ok=True)

    xyn_rows_by_image = group_rows_by_image(xyn_rows)
    box_rows_by_image = group_rows_by_image(box_rows)

    for image_index, image_path in enumerate(image_paths):
        segmentation_lines = [
            row["yolo_segmentation_line"] for row in xyn_rows_by_image.get(image_index, [])
        ]
        detection_lines = [
            build_yolo_detection_line(row) for row in box_rows_by_image.get(image_index, [])
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

