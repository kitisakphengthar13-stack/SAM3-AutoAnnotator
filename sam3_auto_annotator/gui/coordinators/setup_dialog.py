from __future__ import annotations

from PySide6.QtCore import QTimer

from sam3_auto_annotator.services.project_service import parse_prompts


class SetupDialogCoordinator:
    """Own the staged Apply/Cancel transaction for project configuration."""

    def __init__(self, window):
        self.window = window
        self._snapshot = None
        self._snapshot_pending = False
        window.setup.apply_requested.connect(self.apply)
        window.setup.cancel_requested.connect(window.setup_dialog.reject)
        window.setup_dialog.rejected.connect(self.restore)

    def show(self):
        dialog = self.window.setup_dialog
        if not dialog.isVisible():
            self._snapshot_pending = True
            QTimer.singleShot(0, self._capture)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _capture(self):
        if not self._snapshot_pending or not self.window.setup_dialog.isVisible():
            return
        self._snapshot_pending = False
        self._snapshot = self.window.setup.snapshot()

    def apply(self):
        controller = self.window.controller
        if controller is not None and controller.project is not None:
            prompts = parse_prompts(self.window.setup.prompts_text())
            prompt_error = controller._prompt_validation_error(prompts)
            if prompt_error:
                self.window.setup.set_prompt_error(prompt_error)
                controller._update_actions()
                controller._update_context()
                return

        self.window.setup.settings_changed.emit()
        if self.window.setup.prompt_validation_label.isVisible():
            return

        self._snapshot = None
        self._snapshot_pending = False
        self.window.setup_dialog.accept()

    def restore(self):
        if self._snapshot is not None:
            self.window.setup.restore_snapshot(self._snapshot)
        self._snapshot = None
        self._snapshot_pending = False
        controller = self.window.controller
        if controller is not None:
            controller._update_actions()
            controller._update_context()
