from __future__ import annotations

from sam3_auto_annotator.gui.controller import AppController
from sam3_auto_annotator.gui.controllers.export_controller import ExportController


class WorkstationController(AppController):
    """Active application controller while legacy AppController is decomposed.

    New use cases move into focused controllers behind this facade. Once every
    legacy responsibility has migrated, the inherited implementation can be
    deleted instead of preserved as permanent architecture.
    """

    def __init__(self, *args, **kwargs):
        self.exports = ExportController(self)
        super().__init__(*args, **kwargs)

    def export_labels(self):
        return self.exports.export_labels()

    def save_preview(self, silent=False):
        return self.exports.save_preview(silent=silent)

    def open_preview(self):
        return self.exports.open_preview()

    def open_output(self):
        return self.exports.open_output()

    def _output_dir(self):
        return self.exports.output_dir()
