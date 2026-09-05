import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from domain import ImageStatus
from gui.controllers import WorkstationController
from gui.main_window import MainWindow
from services.project_service import create_project


class MemorySettings:
    def last_directory(self):
        return ""

    def set_last_directory(self, _path):
        pass

    def save_window(self, _window):
        pass


class GuiHistoryBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        image_path = self.root / "image.png"
        image = QImage(100, 80, QImage.Format_RGB32)
        image.fill(Qt.white)
        self.assertTrue(image.save(str(image_path)))

        self.window = MainWindow()
        self.controller = WorkstationController(self.window, MemorySettings())
        self.controller.projects.load_project(
            create_project(image_path, ["car"], half=False)
        )
        QCoreApplication.processEvents()
        self.window.canvas_area.active_class_combo.setCurrentIndex(0)

    def tearDown(self):
        self.window.controller = None
        self.window.close()
        self.window.deleteLater()
        QCoreApplication.processEvents()
        self.temp_dir.cleanup()

    def add_undoable_box(self):
        self.window.canvas.box_drawn.emit((10, 10, 40, 40))
        QCoreApplication.processEvents()
        self.assertEqual(self.window.history.stack.count(), 1)

    def test_review_is_external_mutation_barrier_for_older_edit_commands(self):
        self.add_undoable_box()

        self.controller.annotations.mark_current_reviewed()
        QCoreApplication.processEvents()

        self.assertEqual(self.controller.current_image.status, ImageStatus.REVIEWED)
        self.assertEqual(self.window.history.stack.count(), 0)
        self.assertFalse(self.window.actions.undo.isEnabled())
        self.assertTrue(self.controller.dirty)

    def test_external_dirty_mutation_discards_history_without_marking_clean(self):
        self.add_undoable_box()

        self.controller.presentation.mark_dirty(refresh=False)
        QCoreApplication.processEvents()

        self.assertEqual(self.window.history.stack.count(), 0)
        self.assertTrue(self.controller.dirty)

    def test_sync_time_settings_mutation_marks_dirty_and_discards_old_history(self):
        self.add_undoable_box()
        self.window.history.mark_clean()
        QCoreApplication.processEvents()
        self.assertFalse(self.controller.dirty)

        self.window.setup.conf_edit.setValue(0.73)
        prompts = self.controller.projects.sync_project_settings()
        QCoreApplication.processEvents()

        self.assertEqual(prompts, ["car"])
        self.assertEqual(self.controller.project.confidence, 0.73)
        self.assertTrue(self.controller.dirty)
        self.assertEqual(self.window.history.stack.count(), 0)
        self.assertFalse(self.window.actions.undo.isEnabled())


if __name__ == "__main__":
    unittest.main()
