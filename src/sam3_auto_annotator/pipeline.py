from collections import Counter
from datetime import datetime
from pathlib import Path

from sam3_auto_annotator.exporters.csv_exporter import BOX_FIELDS, XYN_FIELDS, write_csv
from sam3_auto_annotator.exporters.yolo_exporter import write_yolo_labels
from sam3_auto_annotator.paths import (
    BOX_CSV_NAME,
    XYN_CSV_NAME,
    create_project_output_dir,
    default_project_name,
    find_images,
    normalize_export_formats,
    validate_model_path,
)
from sam3_auto_annotator.predictor import create_predictor
from sam3_auto_annotator.summary import save_run_summary
from sam3_auto_annotator.visualization import save_or_show_prediction


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


def run(args):
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

    predictor = create_predictor(
        model_path=validated_model_path,
        conf=args.conf,
        half=args.half,
    )

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

    xyn_csv_path = output_dir / XYN_CSV_NAME if "csv" in export_formats else None
    box_csv_path = output_dir / BOX_CSV_NAME if "csv" in export_formats else None
    run_summary_path = output_dir / "run_summary.json"
    yolo_segmentation_dir = None
    yolo_detection_dir = None

    if "csv" in export_formats:
        write_csv(xyn_csv_path, all_xyn_rows, XYN_FIELDS)
        write_csv(box_csv_path, all_box_rows, BOX_FIELDS)

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

