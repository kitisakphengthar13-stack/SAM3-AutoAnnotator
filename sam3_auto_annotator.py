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
        default=["siamese cat"],
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
        help="Use fp16 inference when supported.",
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
        "--save-annotated",
        action="store_true",
        help="Save annotated result images into the project output folder.",
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
        help="Show annotated images with matplotlib.",
    )
    return parser.parse_args()


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


def annotated_image_path(annotated_dir, image_path):
    return annotated_dir / f"{image_path.stem}_annotated.png"


def save_or_show_annotated(result, image_path, annotated_dir=None, show=False):
    if annotated_dir is None and not show:
        return

    annotated = result.plot()
    if annotated_dir is not None:
        annotated_dir.mkdir(parents=True, exist_ok=True)
        plt.imsave(annotated_image_path(annotated_dir, image_path), annotated)
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(annotated)
        plt.axis("off")
        plt.show()


def process_image(predictor, image_path, prompts, image_index, annotated_dir=None, show=False):
    predictor.set_image(str(image_path))
    results = predictor(text=prompts)
    result = results[0]

    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    polygons = getattr(masks, "xyn", None) if masks is not None else None

    if boxes is None or polygons is None or len(polygons) == 0:
        save_or_show_annotated(result, image_path, annotated_dir=annotated_dir, show=show)
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

    save_or_show_annotated(result, image_path, annotated_dir=annotated_dir, show=show)

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


def save_run_summary(path, summary):
    with path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write("\n")


def main():
    args = parse_args()
    image_paths = find_images(args.input)
    project_name = args.project_name or default_project_name(args.input, args.text)
    output_dir, project_name = create_project_output_dir(
        output_root=args.output_root,
        project_name=project_name,
        force_timestamp=args.timestamp,
        overwrite=args.overwrite,
    )
    annotated_dir = output_dir / "annotated_images" if args.save_annotated else None
    created_at = datetime.now().isoformat(timespec="seconds")

    overrides = dict(
        conf=args.conf,
        task="segment",
        mode="predict",
        model=args.model,
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
            annotated_dir=annotated_dir,
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

    xyn_csv_path = output_dir / XYN_CSV_NAME
    box_csv_path = output_dir / BOX_CSV_NAME
    write_csv(xyn_csv_path, all_xyn_rows, xyn_fields)
    write_csv(box_csv_path, all_box_rows, box_fields)

    if args.run_summary:
        summary = {
            "project_name": project_name,
            "output_folder": str(output_dir),
            "input_path": str(Path(args.input)),
            "model_path": str(Path(args.model)),
            "prompts": args.text,
            "confidence_threshold": args.conf,
            "images_processed": len(image_paths),
            "images_with_detections": len(image_paths) - len(no_detection_images),
            "images_with_no_detections": len(no_detection_images),
            "total_detections": len(all_box_rows),
            "class_counts": dict(sorted(total_class_counts.items())),
            "output_files": {
                "xyn_csv": str(xyn_csv_path),
                "box_csv": str(box_csv_path),
                "annotated_images": str(annotated_dir) if annotated_dir is not None else None,
            },
            "created_at": created_at,
        }
        save_run_summary(output_dir / "run_summary.json", summary)

    print("\nSummary")
    print(f"Project name: {project_name}")
    print(f"Output folder: {output_dir}")
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
    print(f"Saved polygon CSV: {xyn_csv_path}")
    print(f"Saved box CSV: {box_csv_path}")
    if args.run_summary:
        print(f"Saved run summary: {output_dir / 'run_summary.json'}")
    if annotated_dir is not None:
        print(f"Saved annotated images: {annotated_dir}")


if __name__ == "__main__":
    main()


# Example commands:
# python sam3_auto_annotator.py --input "path\to\image.jpg" --model "path\to\sam3.pt" --text "siamese cat"
# python sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text "siamese cat" dog car --project-name sample_run
