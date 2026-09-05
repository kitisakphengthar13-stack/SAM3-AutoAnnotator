import inspect
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from gui.controllers import WorkstationController
from gui.controllers.annotation_controller import AnnotationController
from gui.controllers.export_controller import ExportController
from gui.controllers.inference_controller import InferenceController
from gui.controllers.presentation_controller import PresentationController
from gui.controllers.project_controller import ProjectController
from gui.main_window import MainWindow
from services.project_service import create_project


class MemorySettings:
    def last_directory(self):
        return ""

    def set_last_directory(self, _path):
        pass

    def save_window(self, _window):
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

    def _image(self, name="image.png"):
        path = self.root / name
        image = QImage(100, 80, QImage.Format_RGB32)
        image.fill(Qt.white)
        self.assertTrue(image.save(str(path)))
        return path

    def _two_image_folder(self):
        folder = self.root / "images"
        folder.mkdir()
        for name in ("a.png", "b.png"):
            image = QImage(100, 80, QImage.Format_RGB32)
            image.fill(Qt.white)
            self.assertTrue(image.save(str(folder / name)))
        return folder

    def test_window_is_bound_to_active_workstation_controller(self):
        self.assertIs(self.window.controller, self.controller)
        self.assertIsInstance(self.controller.projects, ProjectController)
        self.assertIsInstance(self.controller.annotations, AnnotationController)
        self.assertIsInstance(self.controller.inference, InferenceController)
        self.assertIsInstance(self.controller.exports, ExportController)
        self.assertIsInstance(self.controller.presentation, PresentationController)

    def test_active_use_case_controllers_do_not_reference_retired_inspector(self):
        for controller_type in (
            ProjectController,
            AnnotationController,
            InferenceController,
            ExportController,
            PresentationController,
        ):
            with self.subTest(controller=controller_type.__name__):
                self.assertNotIn("inspector", inspect.getsource(controller_type))

    def test_workstation_is_composition_not_reimplemented_use_cases(self):
        module_source = inspect.getsource(inspect.getmodule(WorkstationController))
        source = inspect.getsource(WorkstationController)
        self.assertNotIn("services.project_service", module_source)
        self.assertNotIn("storage.image_catalog", module_source)
        self.assertNotIn("save_state_to_output", module_source)
        self.assertNotIn("validate_model_path", module_source)
        self.assertNotIn("remaining_prediction_targets", module_source)
        self.assertNotIn("set_prompt_error(", source)

    def test_active_signal_routing_targets_focused_controllers(self):
        source = inspect.getsource(WorkstationController)
        self.assertIn("actions.open_image: self.projects.open_image", source)
        self.assertIn("actions.run_current: self.inference.run_current", source)
        self.assertIn(
            "actions.mark_reviewed: self.annotations.review_current_and_select_next",
            source,
        )
        self.assertIn("self.tasks.prediction_ready.connect(self.inference.prediction_ready)", source)

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

    def test_project_activation_does_not_open_setup_dialog(self):
        self.assertFalse(self.window.setup_dialog.isVisible())
        project = create_project(self._image(), ["car"], half=False)

        self.controller.projects.load_project(project)
        QCoreApplication.processEvents()

        self.assertFalse(self.window.setup_dialog.isVisible())
        self.assertIs(self.controller.project, project)

    def test_manual_box_uses_visible_active_class_not_selected_object_editor(self):
        project = create_project(self._image(), ["car", "truck"], half=False)
        self.controller.projects.load_project(project)
        QCoreApplication.processEvents()

        self.window.annotation.class_combo.setCurrentIndex(0)
        self.window.canvas_area.active_class_combo.setCurrentIndex(1)
        self.controller.annotations.add_manual_box((10, 10, 40, 40))

        self.assertEqual(len(self.controller.current_image.active_annotations), 1)
        annotation = self.controller.current_image.active_annotations[0]
        self.assertEqual(annotation.class_id, 1)
        self.assertEqual(annotation.class_name, "truck")

    def test_review_and_next_advances_once_in_all_images_filter(self):
        project = create_project(self._two_image_folder(), ["car"], half=False)
        self.controller.projects.load_project(project)
        QCoreApplication.processEvents()
        first_index = self.controller.current_image_index

        self.window.actions.mark_reviewed.trigger()
        QCoreApplication.processEvents()

        self.assertNotEqual(self.controller.current_image_index, first_index)
        self.assertEqual(self.controller.current_image_index, project.images[1].image_index)

    def test_review_and_next_does_not_double_advance_in_needs_review_filter(self):
        project = create_project(self._two_image_folder(), ["car"], half=False)
        self.controller.projects.load_project(project)
        QCoreApplication.processEvents()
        self.window.dataset.status_filter.setCurrentIndex(1)
        QCoreApplication.processEvents()
        first_index = self.controller.current_image_index

        self.window.actions.mark_reviewed.trigger()
        QCoreApplication.processEvents()

        self.assertNotEqual(self.controller.current_image_index, first_index)
        self.assertEqual(self.controller.current_image_index, project.images[1].image_index)
        self.assertEqual(self.window.dataset.filter_model.rowCount(), 1)


if __name__ == "__main__":
    unittest.main()
