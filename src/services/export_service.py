from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import shutil
import tempfile

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


def _remove_path(path):
    path = Path(path)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _publish_stage(stage_dir, output_dir, managed_names):
    """Publish generated artifacts as one rollback-capable filesystem transaction."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(
        tempfile.mkdtemp(prefix=".sam3-export-backup-", dir=output_dir.parent)
    )
    moved_existing = []
    published = []
    try:
        for name in managed_names:
            target = output_dir / name
            if target.exists():
                os.replace(target, backup_dir / name)
                moved_existing.append(name)

        for name in managed_names:
            staged = Path(stage_dir) / name
            if staged.exists():
                os.replace(staged, output_dir / name)
                published.append(name)
    except Exception:
        for name in reversed(published):
            _remove_path(output_dir / name)
        for name in moved_existing:
            backup = backup_dir / name
            if backup.exists():
                os.replace(backup, output_dir / name)
        raise
    finally:
        shutil.rmtree(backup_dir, ignore_errors=True)


def export_corrected_detection(project_state, output_dir, write_summary=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = image_paths_for_export(project_state)
    box_rows = build_box_rows(project_state)
    segmentation_rows = build_segmentation_rows(project_state)
    skipped_segmentation_rows = build_skipped_segmentation_rows(project_state)
    incomplete_images = [
        image
        for image in project_state.images
        if getattr(image.status, "value", image.status) in {"not_predicted", "error"}
    ]

    with tempfile.TemporaryDirectory(
        prefix=".sam3-export-stage-", dir=output_dir.parent
    ) as temp_dir:
        stage_dir = Path(temp_dir)
        box_csv_stage = stage_dir / BOX_CSV_NAME
        write_csv(box_csv_stage, box_rows, BOX_FIELDS)
        write_yolo_labels(
            output_dir=stage_dir,
            image_paths=image_paths,
            xyn_rows=segmentation_rows,
            box_rows=box_rows,
        )

        if skipped_segmentation_rows:
            save_run_summary(
                stage_dir / "segmentation_skipped_report.json",
                {
                    "total_skipped_segmentations": len(skipped_segmentation_rows),
                    "skipped_segmentations": skipped_segmentation_rows,
                },
            )

        if write_summary:
            save_run_summary(
                stage_dir / "run_summary.json",
                {
                    "project_name": project_state.project_name,
                    "output_folder": str(output_dir),
                    "export_formats": ["csv", "yolo_detection", "yolo_segmentation"],
                    "input_path": project_state.input_path,
                    "model_path": project_state.model_path,
                    "prompts": project_state.prompts,
                    "images_processed": len(project_state.images),
                    "images_incomplete": len(incomplete_images),
                    "incomplete_images": [image.image_path for image in incomplete_images],
                    # Backward-compatible field now includes failed images as incomplete.
                    "images_not_predicted": len(incomplete_images),
                    "unpredicted_images": [image.image_path for image in incomplete_images],
                    "total_detections": len(box_rows),
                    "total_segmentations": len(segmentation_rows),
                    "total_skipped_segmentations": len(skipped_segmentation_rows),
                    "output_files": {
                        "box_csv": str(output_dir / BOX_CSV_NAME),
                        "yolo_detection_labels": str(output_dir / "yolo_labels" / "detection"),
                        "yolo_segmentation_labels": str(output_dir / "yolo_labels" / "segmentation"),
                        "segmentation_skipped_report": (
                            str(output_dir / "segmentation_skipped_report.json")
                            if skipped_segmentation_rows
                            else None
                        ),
                    },
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "source": "editable_project_state",
                },
            )

        managed = [BOX_CSV_NAME, "yolo_labels", "segmentation_skipped_report.json"]
        if write_summary:
            managed.append("run_summary.json")
        _publish_stage(stage_dir, output_dir, managed)

    box_csv_path = output_dir / BOX_CSV_NAME
    segmentation_dir = output_dir / "yolo_labels" / "segmentation"
    detection_dir = output_dir / "yolo_labels" / "detection"
    skipped_report_path = (
        output_dir / "segmentation_skipped_report.json"
        if skipped_segmentation_rows
        else None
    )
    summary_path = output_dir / "run_summary.json" if write_summary else None

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
