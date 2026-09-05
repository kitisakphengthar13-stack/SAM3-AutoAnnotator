from __future__ import annotations


class ExportDialogCoordinator:
    """Build and present export readiness without mutating project data."""

    def __init__(self, window):
        self.window = window

    def show_preflight(self):
        controller = self.window.controller
        project = controller.project if controller is not None else None
        if project is None:
            return

        images = list(project.images)
        reviewed = sum(
            getattr(image.status, "value", image.status) == "reviewed"
            for image in images
        )
        incomplete = sum(
            getattr(image.status, "value", image.status) in {"not_predicted", "error"}
            for image in images
        )
        stale_segmentation = sum(
            1
            for image in images
            for annotation in image.active_annotations
            if not annotation.segmentation_valid
        )
        needs_review = len(images) - reviewed
        warning = needs_review > 0 or incomplete > 0 or stale_segmentation > 0

        self.window.results.set_status(
            "Review export warnings before writing files."
            if warning
            else "Project is ready to export.",
            "\n".join(
                (
                    f"Reviewed images: {reviewed}/{len(images)}",
                    f"Needs review: {needs_review}",
                    f"Unpredicted / failed: {incomplete}",
                    f"Stale / missing segmentation: {stale_segmentation}",
                )
            ),
        )
        self.window.actions.export.setText("Export Anyway" if warning else "Export Now")
        self.show_results()

    def show_results(self):
        dialog = self.window.results_dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def bypass_incomplete_confirmation(self, title):
        return str(title) == "Incomplete Images" and self.window.results_dialog.isVisible()
