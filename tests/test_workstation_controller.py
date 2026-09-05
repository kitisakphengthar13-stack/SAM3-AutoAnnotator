import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from sam3_auto_annotator.gui.controllers import WorkstationController
from sam3_auto_annotator.gui.controllers.export_controller import ExportController
from sam3_auto_annotator.gui.main_window import MainWindow


class MemorySettings:
    def last_directory(self):
        return ""

    def set_last_directory(self, _path):
        pass

    def save_window(self, _window, _workspace=None):
        pass


class WorkstationControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.controller = WorkstationController(self.window, MemorySettings())

    def tearDown(self):
        self.window.controller = None
        self.window.close()
        self.window.deleteLater()

    def test_window_is_bound_to_active_workstation_controller(self):
        self.assertIs(self.window.controller, self.controller)
        self.assertIsInstance(self.controller.exports, ExportController)

    def test_export_methods_route_through_extracted_controller(self):
        calls = []
        self.controller.exports.export_labels = lambda: calls.append("export")
        self.controller.exports.save_preview = (
            lambda silent=False: calls.append(("preview", silent)) or "preview.png"
        )
        self.controller.exports.open_preview = lambda: calls.append("open-preview")
        self.controller.exports.open_output = lambda: calls.append("open-output")

        self.controller.export_labels()
        result = self.controller.save_preview(silent=True)
        self.controller.open_preview()
        self.controller.open_output()

        self.assertEqual(result, "preview.png")
        self.assertEqual(
            calls,
            ["export", ("preview", True), "open-preview", "open-output"],
        )


if __name__ == "__main__":
    unittest.main()
