import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from sam3_auto_annotator.gui.controllers import WorkstationController
from sam3_auto_annotator.gui.controllers.annotation_controller import AnnotationController
from sam3_auto_annotator.gui.controllers.export_controller import ExportController
from sam3_auto_annotator.gui.main_window import MainWindow
from sam3_auto_annotator.services.project_service import create_project


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
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.window = MainWindow()
        self.controller = WorkstationController(self.window, MemorySettings())

    def tearDown(self):
        self.window.controller = None
        self.window.close()
        self.window.deleteLater()
        QCoreApplication.processEvents()
        self.temp_dir.cleanup()

    def _image(self):
        path = self.root / "image.png"
        image = QImage(100, 80, QImage.Format_RGB32)
        image.fill(Qt.white)
        self.assertTrue(image.save(str(path)))
        return path

    def test_window_is_bound_to_active_workstation_controller(self):
        self.assertIs(self.window.controller, self.controller)
        self.assertIsInstance(self.controller.annotations, AnnotationController)
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

    def test_manual_box_uses_visible_active_class_not_selected_object_editor(self):
        project = create_project(self._image(), ["car", "truck"], half=False)
        self.controller._load_project(project)
        QCoreApplication.processEvents()

        self.window.annotation.class_combo.setCurrentIndex(0)
        self.window.canvas_area.active_class_combo.setCurrentIndex(1)
        self.controller.add_manual_box((10, 10, 40, 40))

        self.assertEqual(len(self.controller.current_image.active_annotations), 1)
        annotation = self.controller.current_image.active_annotations[0]
        self.assertEqual(annotation.class_id, 1)
        self.assertEqual(annotation.class_name, "truck")


if __name__ == "__main__":
    unittest.main()
