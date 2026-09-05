from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from domain.segmentation import (
    build_segmentation_rows,
    build_skipped_segmentation_rows,
)
from services.export_rows import build_box_rows, image_paths_for_export
from storage.exporters.csv_exporter import BOX_FIELDS, write_csv
from storage.exporters.summary_writer import save_run_summary
from storage.exporters.yolo_exporter import write_yolo_labels
from storage.image_catalog import BOX_CSV_NAME


@dataclass(frozen=True)
class ExportReadiness:
    total_images: int
    reviewed_images: int
    needs_review: int
    incomplete_images: int
    stale_segmentations: int

    @property
    def has_warnings(self):
        return bool(
            self.needs_review
            or self.incomplete_images
            or self.stale_segmentations
        )


def evaluate_export_readiness(project_state):
    """Return export readiness facts without presenting UI or writing files."""
    images = list(project_state.images)
    reviewed = sum(
        getattr(image.status, "value", image.status) == "reviewed"
        for image in images
    )
    incomplete = sum(
        getattr(image.status, "value", image.status) in {"not_predicted", "error"}
        for image in images
    )
    stale = sum(
        1
        for image in images
        for annotation in image.active_annotations
        if not annotation.segmentation_valid
    )
    return ExportReadiness(
        total_images=len(images),
        reviewed_images=reviewed,
        needs_review=len(images) - reviewed,
        incomplete_images=incomplete,
        stale_segmentations=stale,
    )


def export_corrected_detection(project_state, output_dir, write_summary=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = image_paths_for_export(project_state)
    box_rows = build_box_rows(project_state)
    segmentation_rows = build_segmentation_rows(project_state)
    skipped_segmentation_rows = build_skipped_segmentation_rows(project_state)
    box_csv_path = output_dir / BOX_CSV_NAME
    write_csv(box_csv_path, box_rows, BOX_FIELDS)
    segmentation_dir, detection_dir = write_yolo_labels(
        output_dir=output_dir,
        image_paths=image_paths,
        xyn_rows=segmentation_rows,
        box_rows=box_rows,
    )

    skipped_report_path = None
    if skipped_segmentation_rows:
        skipped_report_path = output_dir / "segmentation_skipped_report.json"
        save_run_summary(
            skipped_report_path,
            {
                "total_skipped_segmentations": len(skipped_segmentation_rows),
                "skipped_segmentations": skipped_segmentation_rows,
            },
        )

    summary_path = None
    if write_summary:
        unpredicted = project_state.unpredicted_images
        summary_path = output_dir / "run_summary.json"
        save_run_summary(
            summary_path,
            {
                "project_name": project_state.project_name,
                "output_folder": str(output_dir),
                "export_formats": ["csv", "yolo_detection", "yolo_segmentation"],
                "input_path": project_state.input_path,
                "model_path": project_state.model_path,
                "prompts": project_state.prompts,
                "images_processed": len(project_state.images),
                "images_not_predicted": len(unpredicted),
                "unpredicted_images": [image.image_path for image in unpredicted],
                "total_detections": len(box_rows),
                "total_segmentations": len(segmentation_rows),
                "total_skipped_segmentations": len(skipped_segmentation_rows),
                "output_files": {
                    "box_csv": str(box_csv_path),
                    "yolo_detection_labels": str(detection_dir),
                    "yolo_segmentation_labels": str(segmentation_dir),
                    "segmentation_skipped_report": (
                        None if skipped_report_path is None else str(skipped_report_path)
                    ),
                },
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source": "editable_project_state",
            },
        )

    return {
        "box_csv": box_csv_path,
        "yolo_detection_dir": detection_dir,
        "yolo_segmentation_dir": segmentation_dir,
        "run_summary": summary_path,
        "segmentation_skipped_report": skipped_report_path,
        "rows": box_rows,
        "segmentation_rows": segmentation_rows,
        "skipped_segmentation_rows": skipped_segmentation_rows,
    }
