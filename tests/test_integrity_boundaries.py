import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from domain import ImageRecord, ImageStatus, ProjectState
from gui.controllers import WorkstationController
from gui.main_window import MainWindow
from services.export_service import _publish_stage, export_corrected_detection
from test_gui_fields import FakeSettings, FakeTaskManager


class IntegrityBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_display_decode_failure_does_not_mutate_project_state(self):
        with tempfile.TemporaryDirectory() as temp:
            corrupt = Path(temp) / "corrupt.png"
            corrupt.write_bytes(b"not an image")
            project = ProjectState(
                input_path=str(corrupt),
                prompts=["car"],
                images=[
                    ImageRecord(
                        str(corrupt),
                        0,
                        status=ImageStatus.REVIEWED,
                    )
                ],
            )
            window = MainWindow()
            errors = []
            window.show_error = lambda *args, **kwargs: errors.append((args, kwargs))
            controller = WorkstationController(
                window,
                FakeSettings(),
                task_manager=FakeTaskManager(),
            )
            window.show()
            QCoreApplication.processEvents()
            try:
                controller.projects.load_project(project)
                QCoreApplication.processEvents()
                image = project.images[0]
                self.assertEqual(image.status, ImageStatus.REVIEWED)
                self.assertIsNone(image.error_message)
                self.assertFalse(controller.dirty)
                self.assertFalse(controller._current_image_loaded)
                self.assertTrue(errors)
            finally:
                window.controller = None
                window.close()
                window.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
                QCoreApplication.processEvents()

    def test_export_publish_failure_restores_previous_managed_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = root / "stage"
            output = root / "output"
            stage.mkdir()
            output.mkdir()

            (stage / "a.txt").write_text("new-a", encoding="utf-8")
            (stage / "tree").mkdir()
            (stage / "tree" / "new.txt").write_text("new-tree", encoding="utf-8")
            (output / "a.txt").write_text("old-a", encoding="utf-8")
            (output / "tree").mkdir()
            (output / "tree" / "old.txt").write_text("old-tree", encoding="utf-8")

            real_replace = os.replace

            def fail_second_publish(source, destination):
                source = Path(source)
                if source == stage / "tree":
                    raise OSError("simulated publish failure")
                return real_replace(source, destination)

            with patch("services.export_service.os.replace", side_effect=fail_second_publish):
                with self.assertRaisesRegex(OSError, "simulated publish failure"):
                    _publish_stage(stage, output, ["a.txt", "tree"])

            self.assertEqual((output / "a.txt").read_text(encoding="utf-8"), "old-a")
            self.assertTrue((output / "tree" / "old.txt").is_file())
            self.assertFalse((output / "tree" / "new.txt").exists())

    def test_run_summary_counts_failed_images_as_incomplete(self):
        project = ProjectState(
            input_path="images",
            prompts=[],
            images=[
                ImageRecord(
                    "images/a.jpg",
                    0,
                    status=ImageStatus.ERROR,
                    error_message="inference failed",
                )
            ],
        )
        with tempfile.TemporaryDirectory() as temp:
            result = export_corrected_detection(project, temp)
            with result["run_summary"].open(encoding="utf-8") as summary_file:
                summary = json.load(summary_file)

        self.assertEqual(summary["images_incomplete"], 1)
        self.assertEqual(summary["images_not_predicted"], 1)
        self.assertEqual(summary["incomplete_images"], ["images/a.jpg"])


if __name__ == "__main__":
    unittest.main()
