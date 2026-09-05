import inspect
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from sam3_auto_annotator.gui.coordinators import (
    AnnotationHistoryCoordinator,
    AnnotationInteractionCoordinator,
    ExportDialogCoordinator,
    SetupDialogCoordinator,
)
from sam3_auto_annotator.gui.main_window import MainWindow


class GuiCoordinatorBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()

    def tearDown(self):
        self.window.controller = None
        self.window.close()
        self.window.deleteLater()

    def test_main_window_composes_workflow_coordinators(self):
        self.assertIsInstance(
            self.window.annotation_interaction,
            AnnotationInteractionCoordinator,
        )
        self.assertIsInstance(self.window.history, AnnotationHistoryCoordinator)
        self.assertIsInstance(self.window.setup_flow, SetupDialogCoordinator)
        self.assertIsInstance(self.window.export_flow, ExportDialogCoordinator)
        self.assertIs(self.window.undo_stack, self.window.history.stack)

    def test_workflow_private_state_is_not_owned_by_main_window(self):
        for retired_name in (
            "_undo_project",
            "_pending_undo_capture",
            "_draw_class_restore",
            "_setup_snapshot",
            "_setup_snapshot_pending",
        ):
            with self.subTest(name=retired_name):
                self.assertFalse(hasattr(self.window, retired_name))

    def test_history_does_not_own_active_class_bridge(self):
        self.assertFalse(hasattr(self.window.history, "_draw_class_restore"))
        self.assertTrue(
            hasattr(self.window.annotation_interaction, "_draw_class_restore")
        )

    def test_main_window_source_does_not_reimplement_workflow_algorithms(self):
        source = inspect.getsource(MainWindow)
        self.assertNotIn("ImageSnapshotCommand", source)
        self.assertNotIn("parse_prompts", source)
        self.assertNotIn("stale_segmentation = sum", source)
        self.assertNotIn("_advance_after_review", source)


if __name__ == "__main__":
    unittest.main()
