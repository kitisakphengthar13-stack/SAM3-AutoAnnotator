from __future__ import annotations

import logging
from pathlib import Path

from domain import ImageStatus
from gui.rendering.annotation_preview import (
    OverlayOptions,
    render_annotation_preview,
)
from services.project_service import (
    default_output_dir,
    export_project,
    save_state_to_output,
)


logger = logging.getLogger(__name__)


class ExportController:
    """Own export, preview, and output-location application workflows."""

    def __init__(self, host):
        self.host = host

    def export_labels(self):
        host = self.host
        if host.project is None:
            return
        try:
            host.projects.sync_project_settings()
        except Exception as exc:
            host.presentation.report_error(
                "Could Not Export Labels",
                "The project settings are not valid for export.",
                "Restore every class in use, then retry.",
                exc,
            )
            return

        incomplete = [
            image
            for image in host.project.images
            if image.status in {ImageStatus.NOT_PREDICTED, ImageStatus.ERROR}
        ]
        if incomplete and not host.view.confirm(
            "Incomplete Images",
            f"{len(incomplete)} image(s) are unpredicted or failed. "
            "They will receive empty YOLO label files unless they contain manual boxes.",
            confirm_text="Export Anyway",
        ):
            return

        try:
            output_dir = self.output_dir()
            host.current_state_path = save_state_to_output(host.project, output_dir)
            host._saved_output_dir = Path(output_dir)
            result = export_project(host.project, output_dir)
            self._validate_export(result)
            host.last_export_result = result
            preview = self.save_preview(silent=True)
            host.dirty = False
            host.view.history.mark_clean()

            skipped = len(result.get("skipped_segmentation_rows", []))
            counts = (
                f"Detection: {len(result['rows'])}\n"
                f"Segmentation: {len(result['segmentation_rows'])}\n"
                f"Skipped segmentation: {skipped}"
            )
            host.view.results.set_status("Export complete.", counts)
            host.view.results.set_output_paths(
                output_dir=output_dir,
                box_csv=result["box_csv"],
                detection_dir=result["yolo_detection_dir"],
                segmentation_dir=result["yolo_segmentation_dir"],
                skipped_report=result.get("segmentation_skipped_report"),
            )
            if preview:
                host.view.results.set_preview(preview)
            host.view.show_results()
            host.view.set_message(f"Exported corrected labels to {output_dir}")
            host.presentation.update_actions()
            host.presentation.update_context()
        except Exception as exc:
            host.presentation.report_error(
                "Could Not Export Labels",
                "The corrected labels could not be exported.",
                "Check the output folder and project data, then retry.",
                exc,
            )

    @staticmethod
    def _validate_export(result):
        for key in ("box_csv", "run_summary"):
            path = result.get(key)
            if path is not None and not Path(path).is_file():
                raise FileNotFoundError(f"Expected export file was not written: {path}")
        for key in ("yolo_detection_dir", "yolo_segmentation_dir"):
            path = result.get(key)
            if path is not None and not Path(path).is_dir():
                raise FileNotFoundError(f"Expected export folder was not written: {path}")

    def save_preview(self, silent=False):
        host = self.host
        image = host.current_image
        if image is None:
            return None
        output_path = (
            self.output_dir()
            / "preview_results"
            / f"{Path(image.image_path).stem}_reviewed.png"
        )
        try:
            render_annotation_preview(
                image.image_path,
                image.active_annotations,
                output_path,
                OverlayOptions(
                    boxes=host.view.canvas_area.show_boxes_check.isChecked(),
                    masks=host.view.canvas_area.show_masks_check.isChecked(),
                    polygons=host.view.canvas_area.show_polygons_check.isChecked(),
                ),
            )
            host.last_preview_path = output_path
            host.view.results.set_preview(output_path)
            if not silent:
                host.view.show_results()
                host.view.set_message(f"Saved preview to {output_path}")
            host.presentation.update_actions()
            return output_path
        except Exception as exc:
            if not silent:
                host.presentation.report_error(
                    "Could Not Save Preview",
                    "The preview image could not be created.",
                    "Check the source image and output folder, then retry.",
                    exc,
                )
            else:
                logger.exception("Could not save preview")
            return None

    def open_preview(self):
        host = self.host
        if host.last_preview_path and Path(host.last_preview_path).is_file():
            host.view.open_local_path(host.last_preview_path)

    def open_output(self):
        if self.host.project is not None:
            self.host.view.open_local_path(self.output_dir())

    def output_dir(self):
        host = self.host
        text = host.view.setup.output_dir_edit.text().strip()
        output = Path(text) if text else default_output_dir(host.project)
        if not text:
            host.view.setup.output_dir_edit.setText(str(output))
        return output
