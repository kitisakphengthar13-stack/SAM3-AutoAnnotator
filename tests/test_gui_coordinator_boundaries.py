import inspect
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.controllers.annotation_controller import AnnotationController
from gui.controllers.export_controller import ExportController
from gui.coordinators import (
    AnnotationHistoryCoordinator,
    ExportDialogCoordinator,
    SetupDialogCoordinator,
)
from gui.main_window import MainWindow


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

    def test_main_window_composes_only_cross_surface_workflow_coordinators(self):
        self.assertFalse(hasattr(self.window, "annotation_interaction"))
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
            "inspector",
            "annotation_interaction",
        ):
            with self.subTest(name=retired_name):
                self.assertFalse(hasattr(self.window, retired_name))

    def test_active_class_compatibility_bridge_no_longer_exists(self):
        self.assertFalse(hasattr(self.window.history, "_draw_class_restore"))

    def test_review_and_next_is_annotation_use_case_not_timer_glue(self):
        source = inspect.getsource(AnnotationController)
        self.assertIn("review_current_and_select_next", source)
        self.assertNotIn("QTimer", source)
        self.assertNotIn("_advance_after_review", inspect.getsource(MainWindow))

    def test_setup_transaction_captures_snapshot_synchronously(self):
        source = inspect.getsource(SetupDialogCoordinator)
        self.assertNotIn("QTimer", source)
        self.assertNotIn("_snapshot_pending", source)

    def test_export_preflight_is_single_warning_acknowledgement(self):
        coordinator_source = inspect.getsource(ExportDialogCoordinator)
        export_source = inspect.getsource(ExportController)
        window_source = inspect.getsource(MainWindow)
        self.assertNotIn("bypass_incomplete_confirmation", coordinator_source)
        self.assertNotIn('"Incomplete Images"', export_source)
        self.assertNotIn('title == "Delete Annotation"', window_source)
        self.assertNotIn("bypass_incomplete_confirmation", window_source)

    def test_main_window_source_does_not_reimplement_workflow_algorithms(self):
        source = inspect.getsource(MainWindow)
        self.assertNotIn("ImageSnapshotCommand", source)
        self.assertNotIn("parse_prompts", source)
        self.assertNotIn("stale_segmentation = sum", source)
        self.assertNotIn("_advance_after_review", source)
        self.assertNotIn("ControllerSurfaceAdapter", source)

    def test_coordinators_do_not_call_private_workstation_facade_methods(self):
        for coordinator_type in (
            AnnotationHistoryCoordinator,
            SetupDialogCoordinator,
            ExportDialogCoordinator,
        ):
            with self.subTest(coordinator=coordinator_type.__name__):
                self.assertNotIn("controller._", inspect.getsource(coordinator_type))


if __name__ == "__main__":
    unittest.main()
