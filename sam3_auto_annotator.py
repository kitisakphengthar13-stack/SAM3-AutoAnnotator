import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics.models.sam import SAM3SemanticPredictor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
XYN_CSV_NAME = "sam3_auto_annotation_xyn_outputs.csv"
BOX_CSV_NAME = "sam3_auto_annotation_box_outputs.csv"
EXPORT_FORMATS = ["csv", "yolo"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Ultralytics SAM3 auto-annotation on one image or a folder of images."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to one image file or a folder of images.",
    )
    parser.add_argument(
        "--text",
        nargs="+",
        required=True,
        help="Text prompt class names. Example: --text \"siamese cat\" dog car",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the SAM3 model file.",
    )
    parser.add_argument("--conf", type=float, default=0.7, help="Confidence threshold.")
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fp16 inference when supported by the selected device/backend.",
    )
    parser.add_argument(
        "--project-name",
        default=None,
        help="Name of the auto-annotation project output folder.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root folder where project output folders will be created.",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Always append a timestamp to the project output folder name.",
    )
    parser.add_argument(
        "--save-predictions",
        "--save-annotated",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="save_predictions",
        help=(
            "Save SAM3 prediction visualization images into prediction_results/. "
            "--save-annotated is a legacy alias."
        ),
    )
    parser.add_argument(
        "--export-formats",
        nargs="+",
        choices=EXPORT_FORMATS,
        default=["csv"],
        help="Output formats to create. Use one or more of: csv yolo. Default: csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing project output folder.",
    )
    parser.add_argument(
        "--run-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save run_summary.json in the project output folder.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show prediction visualization images with matplotlib.",
    )
    return parser.parse_args()


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


def tensor_item(value, default=None):
    try:
        return value.item()
    except AttributeError:
        return value if value is not None else default


def get_sequence_value(sequence, index, default=None):
    if sequence is None:
        return default
    try:
        return sequence[index]
    except (IndexError, TypeError):
        return default


def get_class_name(class_id, prompts):
    if 0 <= class_id < len(prompts):
        return prompts[class_id]
    return "unknown"


def format_polygon_xyn(poly):
    return " ".join(f"{float(x):.6f},{float(y):.6f}" for x, y in poly)


def format_yolo_segmentation_line(class_id, poly):
    coords = []
    for x, y in poly:
        coords.append(f"{float(x):.6f}")
        coords.append(f"{float(y):.6f}")
    return f"{class_id} " + " ".join(coords)


def image_shape(result):
    shape = getattr(result, "orig_shape", None)
    if shape and len(shape) >= 2:
        height, width = shape[:2]
        return float(width), float(height)
    return None, None


def xyxy_values(boxes, object_index):
    xyxy = getattr(boxes, "xyxy", None)
    values = get_sequence_value(xyxy, object_index)
    if values is None:
        return None
    return [float(tensor_item(value)) for value in values[:4]]


def xywhn_values(boxes, object_index):
    xywhn = getattr(boxes, "xywhn", None)
    values = get_sequence_value(xywhn, object_index)
    if values is None:
        return None
    return [float(tensor_item(value)) for value in values[:4]]


def prediction_image_path(prediction_results_dir, image_path):
    return prediction_results_dir / f"{image_path.stem}_predicted.png"


def save_or_show_prediction(result, image_path, prediction_results_dir=None, show=False):
    if prediction_results_dir is None and not show:
        return

    prediction = result.plot()
    if prediction_results_dir is not None:
        prediction_results_dir.mkdir(parents=True, exist_ok=True)
        plt.imsave(prediction_image_path(prediction_results_dir, image_path), prediction)
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(prediction)
        plt.axis("off")
        plt.show()


def process_image(predictor, image_path, prompts, image_index, prediction_results_dir=None, show=False):
    predictor.set_image(str(image_path))
    results = predictor(text=prompts)
    result = results[0]

    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    polygons = getattr(masks, "xyn", None) if masks is not None else None

    if boxes is None or polygons is None or len(polygons) == 0:
        save_or_show_prediction(
            result,
            image_path,
            prediction_results_dir=prediction_results_dir,
            show=show,
        )
        return [], [], Counter(), 0

    class_ids = getattr(boxes, "cls", None)
    confidences = getattr(boxes, "conf", None)
    image_width, image_height = image_shape(result)

    xyn_rows = []
    box_rows = []
    class_counter = Counter()

    for object_index, poly in enumerate(polygons):
        class_value = get_sequence_value(class_ids, object_index, 0)
        class_id = int(tensor_item(class_value, 0))
        class_name = get_class_name(class_id, prompts)
        confidence_value = get_sequence_value(confidences, object_index)
        confidence = tensor_item(confidence_value)

        class_counter[class_name] += 1
        polygon_point_count = len(poly)

        xyn_rows.append(
            {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "image_index": image_index,
                "object_index": object_index,
                "class_id": class_id,
                "class_name": class_name,
                "class_count_in_image": "",
                "total_class_count": "",
                "polygon_point_count": polygon_point_count,
                "polygon_xyn": format_polygon_xyn(poly),
                "yolo_segmentation_line": format_yolo_segmentation_line(class_id, poly),
                "confidence": "" if confidence is None else f"{float(confidence):.6f}",
            }
        )

        xyxy = xyxy_values(boxes, object_index)
        xywhn = xywhn_values(boxes, object_index)

        if xyxy is None:
            x1 = y1 = x2 = y2 = width = height = x_center = y_center = ""
        else:
            x1, y1, x2, y2 = xyxy
            width = x2 - x1
            height = y2 - y1
            x_center = x1 + width / 2
            y_center = y1 + height / 2

        if xywhn is not None:
            x_center_norm, y_center_norm, width_norm, height_norm = xywhn
        elif xyxy is not None and image_width and image_height:
            x_center_norm = x_center / image_width
            y_center_norm = y_center / image_height
            width_norm = width / image_width
            height_norm = height / image_height
        else:
            x_center_norm = y_center_norm = width_norm = height_norm = ""

        box_rows.append(
            {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "image_index": image_index,
                "object_index": object_index,
                "class_id": class_id,
                "class_name": class_name,
                "class_count_in_image": "",
                "total_class_count": "",
                "x1": "" if xyxy is None else f"{x1:.6f}",
                "y1": "" if xyxy is None else f"{y1:.6f}",
                "x2": "" if xyxy is None else f"{x2:.6f}",
                "y2": "" if xyxy is None else f"{y2:.6f}",
                "width": "" if xyxy is None else f"{width:.6f}",
                "height": "" if xyxy is None else f"{height:.6f}",
                "x_center": "" if xyxy is None else f"{x_center:.6f}",
                "y_center": "" if xyxy is None else f"{y_center:.6f}",
                "x_center_norm": ""
                if x_center_norm == ""
                else f"{float(x_center_norm):.6f}",
                "y_center_norm": ""
                if y_center_norm == ""
                else f"{float(y_center_norm):.6f}",
                "width_norm": "" if width_norm == "" else f"{float(width_norm):.6f}",
                "height_norm": "" if height_norm == "" else f"{float(height_norm):.6f}",
                "confidence": "" if confidence is None else f"{float(confidence):.6f}",
            }
        )

    save_or_show_prediction(
        result,
        image_path,
        prediction_results_dir=prediction_results_dir,
        show=show,
    )

    return xyn_rows, box_rows, class_counter, len(polygons)


def apply_image_counts(rows, image_class_counts):
    for row in rows:
        class_name = row["class_name"]
        row["class_count_in_image"] = image_class_counts[class_name]


def apply_total_counts(rows, total_class_counts):
    for row in rows:
        class_name = row["class_name"]
        row["total_class_count"] = total_class_counts[class_name]


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def save_run_summary(path, summary):
    with path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write("\n")


def main():
    args = parse_args()
    export_formats = normalize_export_formats(args.export_formats)
    image_paths = find_images(args.input)
    project_name = args.project_name or default_project_name(args.input, args.text)
    output_dir, project_name = create_project_output_dir(
        output_root=args.output_root,
        project_name=project_name,
        force_timestamp=args.timestamp,
        overwrite=args.overwrite,
    )
    prediction_results_dir = output_dir / "prediction_results" if args.save_predictions else None
    created_at = datetime.now().isoformat(timespec="seconds")
    validated_model_path = validate_model_path(args.model)

    overrides = dict(
        conf=args.conf,
        task="segment",
        mode="predict",
        model=str(validated_model_path),
        half=args.half,
        save=False,
    )

    predictor = SAM3SemanticPredictor(overrides=overrides)

    all_xyn_rows = []
    all_box_rows = []
    total_class_counts = Counter()
    no_detection_images = []

    for image_index, image_path in enumerate(image_paths):
        print(f"Processing [{image_index + 1}/{len(image_paths)}]: {image_path}")
        xyn_rows, box_rows, image_class_counts, detection_count = process_image(
            predictor=predictor,
            image_path=image_path,
            prompts=args.text,
            image_index=image_index,
            prediction_results_dir=prediction_results_dir,
            show=args.show,
        )

        if detection_count == 0:
            no_detection_images.append(image_path)
            print("  No detections.")
            continue

        total_class_counts.update(image_class_counts)
        apply_image_counts(xyn_rows, image_class_counts)
        apply_image_counts(box_rows, image_class_counts)
        all_xyn_rows.extend(xyn_rows)
        all_box_rows.extend(box_rows)

        class_summary = ", ".join(
            f"{class_name}: {count}" for class_name, count in sorted(image_class_counts.items())
        )
        print(f"  Detections: {detection_count} ({class_summary})")

    apply_total_counts(all_xyn_rows, total_class_counts)
    apply_total_counts(all_box_rows, total_class_counts)

    xyn_fields = [
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
    box_fields = [
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

    xyn_csv_path = output_dir / XYN_CSV_NAME if "csv" in export_formats else None
    box_csv_path = output_dir / BOX_CSV_NAME if "csv" in export_formats else None
    run_summary_path = output_dir / "run_summary.json"
    yolo_segmentation_dir = None
    yolo_detection_dir = None

    if "csv" in export_formats:
        write_csv(xyn_csv_path, all_xyn_rows, xyn_fields)
        write_csv(box_csv_path, all_box_rows, box_fields)

    if "yolo" in export_formats:
        yolo_segmentation_dir, yolo_detection_dir = write_yolo_labels(
            output_dir=output_dir,
            image_paths=image_paths,
            xyn_rows=all_xyn_rows,
            box_rows=all_box_rows,
        )

    if args.run_summary:
        summary = {
            "project_name": project_name,
            "output_folder": str(output_dir),
            "export_formats": export_formats,
            "input_path": str(Path(args.input)),
            "model_path": str(validated_model_path),
            "prompts": args.text,
            "confidence_threshold": args.conf,
            "images_processed": len(image_paths),
            "images_with_detections": len(image_paths) - len(no_detection_images),
            "images_with_no_detections": len(no_detection_images),
            "total_detections": len(all_box_rows),
            "class_counts": dict(sorted(total_class_counts.items())),
            "output_files": {
                "xyn_csv": str(xyn_csv_path) if xyn_csv_path is not None else None,
                "box_csv": str(box_csv_path) if box_csv_path is not None else None,
                "run_summary_json": str(run_summary_path),
                "prediction_results": (
                    str(prediction_results_dir) if prediction_results_dir is not None else None
                ),
                "yolo_segmentation_labels": (
                    str(yolo_segmentation_dir) if yolo_segmentation_dir is not None else None
                ),
                "yolo_detection_labels": (
                    str(yolo_detection_dir) if yolo_detection_dir is not None else None
                ),
            },
            "created_at": created_at,
        }
        save_run_summary(run_summary_path, summary)

    print("\nSummary")
    print(f"Project name: {project_name}")
    print(f"Output folder: {output_dir}")
    print(f"Export formats: {', '.join(export_formats)}")
    print(f"Images processed: {len(image_paths)}")
    print(f"Images with no detections: {len(no_detection_images)}")
    print(f"Total detections: {len(all_box_rows)}")
    if total_class_counts:
        for class_name, count in sorted(total_class_counts.items()):
            print(f"  {class_name}: {count}")
    if no_detection_images:
        print("No-detection images:")
        for image_path in no_detection_images:
            print(f"  {image_path}")
    if "csv" in export_formats:
        print(f"Saved polygon CSV: {xyn_csv_path}")
        print(f"Saved box CSV: {box_csv_path}")
    if "yolo" in export_formats:
        print(f"Saved YOLO segmentation labels: {yolo_segmentation_dir}")
        print(f"Saved YOLO detection labels: {yolo_detection_dir}")
    if args.run_summary:
        print(f"Saved run summary: {output_dir / 'run_summary.json'}")
    if prediction_results_dir is not None:
        print(f"Saved prediction results: {prediction_results_dir}")


if __name__ == "__main__":
    main()


# Example commands:
# python sam3_auto_annotator.py --input "path\to\image.jpg" --model "path\to\sam3.pt" --text "siamese cat"
# python sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text "siamese cat" dog car --project-name sample_run
