import tempfile
import unittest
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from gui.controllers import UiMode, WorkstationController
from gui.main_window import MainWindow
from services.annotation_service import add_manual_annotation
from services.project_service import create_project


class MemorySettings:
    def last_directory(self):
        return ""

    def set_last_directory(self, _path):
        pass

    def save_window(self, _window, _splitter):
        pass


class StubTaskManager(QObject):
    task_started = Signal(str)
    status = Signal(str)
    progress = Signal(int, int, str)
    prediction_ready = Signal(int, object)
    prediction_failed = Signal(int, str)
    segmentation_ready = Signal(int, str, object)
    segmentation_failed = Signal(int, str, str)
    batch_image_ready = Signal(int, object)
    batch_image_failed = Signal(int, str)
    batch_completed = Signal(dict)
    batch_cancelled = Signal(dict)
    task_failed = Signal(str)
    task_finished = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = False

    @property
    def is_running(self):
        return self._running

    def start_batch(self, _items, **_settings):
        self._running = True
        self.task_started.emit("batch")

    def request_cancel(self):
        return self._running

    def finish(self, kind="batch"):
        self._running = False
        self.task_finished.emit(kind)


class GuiControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.image_path = root / "car.jpg"
        Image.new("RGB", (80, 60), "white").save(self.image_path)
        self.model_path = root / "sam3.pt"
        self.model_path.touch()
        self.tasks = StubTaskManager()
        self.window = MainWindow()
        self.errors = []
        self.window.show_error = lambda *args, **kwargs: self.errors.append(
            (args, kwargs)
        )
        self.window.show_info = lambda *_args, **_kwargs: None
        self.controller = WorkstationController(
            self.window,
            MemorySettings(),
            task_manager=self.tasks,
        )
        project = create_project(
            self.image_path,
            ["car"],
            model_path=self.model_path,
            project_name="controller-test",
            half=False,
        )
        self.controller._load_project(project)
        QApplication.processEvents()

    def tearDown(self):
        self.window.controller = None
        self.window.close()
        self.window.deleteLater()
        QApplication.processEvents()
        self.temp_dir.cleanup()

    def test_task_started_enables_cancel_and_disables_all_setup_browsing(self):
        self.controller._start_task(UiMode.BATCH)
        self.tasks.start_batch([])

        self.assertTrue(self.window.actions.cancel_batch.isEnabled())
        self.assertFalse(self.window.setup.browse_model_button.isEnabled())
        self.assertFalse(self.window.setup.browse_output_button.isEnabled())

        self.tasks.finish()

        self.assertFalse(self.window.actions.cancel_batch.isEnabled())
        self.assertTrue(self.window.setup.browse_model_button.isEnabled())
        self.assertTrue(self.window.setup.browse_output_button.isEnabled())

    def test_setup_draft_does_not_mutate_project_until_apply(self):
        self.window.show_setup()
        QApplication.processEvents()
        self.window.setup.prompts_edit.setPlainText("car\ntruck")

        self.assertEqual(self.controller.project.prompts, ["car"])
        self.assertEqual(self.window.annotation.class_combo.count(), 1)

        self.window.setup.apply_button.click()
        QApplication.processEvents()

        self.assertEqual(self.controller.project.prompts, ["car", "truck"])
        self.assertEqual(self.window.annotation.class_combo.count(), 2)
        self.assertFalse(self.window.setup_dialog.isVisible())

    def test_removing_a_class_in_use_is_rejected_on_apply_without_corrupting_data(self):
        image = self.controller.current_image
        add_manual_annotation(image, 0, "car", (5, 5, 30, 30))
        self.controller._render_current_annotations()
        before = len(image.active_annotations)
        self.window.show_setup()
        QApplication.processEvents()

        self.window.setup.prompts_edit.setPlainText("truck")

        self.assertEqual(self.controller.project.prompts, ["car"])
        self.assertEqual(self.window.annotation.class_combo.itemText(0), "car")
        self.assertTrue(self.window.setup.prompt_validation_label.isHidden())

        self.window.setup.apply_button.click()
        QApplication.processEvents()

        self.assertEqual(self.controller.project.prompts, ["car"])
        self.assertFalse(self.window.setup.prompt_validation_label.isHidden())
        self.assertTrue(self.window.setup_dialog.isVisible())
        self.assertFalse(self.window.actions.draw_box.isEnabled())
        self.assertFalse(self.window.actions.export.isEnabled())

        self.controller.add_manual_box((35, 5, 60, 30))
        self.assertEqual(len(image.active_annotations), before)
        self.assertEqual(len(self.errors), 1)
        self.assertIn("Classes in use cannot be removed", self.errors[0][1]["details"])

        self.window.setup.prompts_edit.setPlainText("car\ntruck")
        self.window.setup.apply_button.click()
        QApplication.processEvents()

        self.assertEqual(self.controller.project.prompts, ["car", "truck"])
        self.assertTrue(self.window.setup.prompt_validation_label.isHidden())
        self.assertTrue(self.window.actions.draw_box.isEnabled())
        self.assertFalse(self.window.setup_dialog.isVisible())

    def test_canvas_context_updates_after_annotation_change(self):
        self.assertIn("0 annotations", self.window.canvas_area.canvas_hint.text())

        self.controller.add_manual_box((5, 5, 30, 30))

        self.assertIn("1 annotations", self.window.canvas_area.canvas_hint.text())

    def test_yolo_import_refreshes_generated_classes_in_setup_and_editor(self):
        label_dir = Path(self.temp_dir.name) / "labels"
        label_dir.mkdir()
        (label_dir / "car.txt").write_text(
            "4 0.5 0.5 0.25 0.25\n", encoding="utf-8"
        )
        self.window.choose_folder = lambda *_args, **_kwargs: str(label_dir)

        self.controller.import_yolo()

        self.assertEqual(
            self.controller.project.prompts,
            ["car", "class_1", "class_2", "class_3", "class_4"],
        )
        self.assertEqual(
            self.window.setup.prompts_text(),
            "car\nclass_1\nclass_2\nclass_3\nclass_4",
        )
        self.assertEqual(self.window.annotation.class_combo.itemText(4), "class_4")
        self.assertTrue(self.window.actions.export.isEnabled())


if __name__ == "__main__":
    unittest.main()
