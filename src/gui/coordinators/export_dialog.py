from __future__ import annotations

from services.export_service import evaluate_export_readiness


class ExportDialogCoordinator:
    """Present export readiness without owning export rules or writing files."""

    def __init__(self, window):
        self.window = window

    def show_preflight(self):
        controller = self.window.controller
        project = controller.project if controller is not None else None
        if project is None:
            return

        readiness = evaluate_export_readiness(project)
        self.window.results.set_status(
            "Review export warnings before writing files."
            if readiness.has_warnings
            else "Project is ready to export.",
            "\n".join(
                (
                    f"Reviewed images: {readiness.reviewed_images}/{readiness.total_images}",
                    f"Needs review: {readiness.needs_review}",
                    f"Unpredicted / failed: {readiness.incomplete_images}",
                    f"Stale / missing segmentation: {readiness.stale_segmentations}",
                )
            ),
        )
        self.window.actions.export.setText(
            "Export Anyway" if readiness.has_warnings else "Export Now"
        )
        self.show_results()

    def show_results(self):
        dialog = self.window.results_dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
