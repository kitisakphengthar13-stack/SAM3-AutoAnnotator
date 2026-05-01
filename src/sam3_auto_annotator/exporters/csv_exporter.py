import csv


XYN_FIELDS = [
    "image_path",
    "image_name",
    "image_index",
    "object_index",
    "class_id",
    "class_name",
    "class_count_in_image",
    "total_class_count",
    "polygon_point_count",
    "polygon_xyn",
    "yolo_segmentation_line",
    "confidence",
]

BOX_FIELDS = [
    "image_path",
    "image_name",
    "image_index",
    "object_index",
    "class_id",
    "class_name",
    "class_count_in_image",
    "total_class_count",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
    "x_center",
    "y_center",
    "x_center_norm",
    "y_center_norm",
    "width_norm",
    "height_norm",
    "confidence",
]


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

