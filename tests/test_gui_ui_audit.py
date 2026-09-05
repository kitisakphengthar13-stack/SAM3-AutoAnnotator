import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QImage, QKeyEvent
from PySide6.QtWidgets import QApplication, QDockWidget, QGraphicsView

from sam3_auto_annotator.gui.main_window import MainWindow


class GuiUiAuditTests(unittest.TestCase):
    """Acceptance tests for the canvas-first workstation contract.

    These tests intentionally describe user-visible outcomes. They do not lock the
    application to the retired Dataset | Canvas | Inspector splitter hierarchy.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.windows = []

    def tearDown(self):
        for window in self.windows:
            window.controller = None
            window.close()
            window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QCoreApplication.processEvents()
        self.temp_dir.cleanup()

    def make_window(self, size=(960, 620)):
        window = MainWindow()
        window.resize(*size)
        window.show()
        self.windows.append(window)
        QCoreApplication.processEvents()
        return window

    def make_image(self, name="canvas.png", size=(160, 100)):
        path = self.root / name
        image = QImage(size[0], size[1], QImage.Format_RGB32)
        image.fill(Qt.white)
        self.assertTrue(image.save(str(path)))
        return path

    def test_canvas_is_the_central_work_surface(self):
        window = self.make_window()
        self.assertIs(window.centralWidget(), window.canvas_area)
        self.assertIs(window.canvas, window.canvas_area.canvas)
        self.assertNotIsInstance(window.centralWidget(), QDockWidget)

    def test_dataset_and_objects_are_independent_docks(self):
        window = self.make_window()
        self.assertIsInstance(window.dataset_dock, QDockWidget)
        self.assertIsInstance(window.annotation_dock, QDockWidget)
        self.assertIs(window.dataset_dock.widget(), window.dataset)
        self.assertIs(window.annotation_dock.widget(), window.annotation)
        self.assertTrue(window.dataset_dock.features() & QDockWidget.DockWidgetClosable)
        self.assertTrue(window.annotation_dock.features() & QDockWidget.DockWidgetClosable)

    def test_focus_workspace_hides_and_restores_side_panels(self):
        window = self.make_window()
        self.assertTrue(window.dataset_dock.isVisible())
        self.assertTrue(window.annotation_dock.isVisible())

        window.actions.focus_workspace.setChecked(True)
        QCoreApplication.processEvents()
        self.assertFalse(window.dataset_dock.isVisible())
        self.assertFalse(window.annotation_dock.isVisible())

        window.actions.focus_workspace.setChecked(False)
        QCoreApplication.processEvents()
        self.assertTrue(window.dataset_dock.isVisible())
        self.assertTrue(window.annotation_dock.isVisible())

    def test_setup_and_export_are_transient_dialog_surfaces(self):
        window = self.make_window()
        self.assertFalse(window.setup_dialog.isVisible())
        self.assertFalse(window.results_dialog.isVisible())

        window.actions.project_settings.trigger()
        QCoreApplication.processEvents()
        self.assertTrue(window.setup_dialog.isVisible())

        window.show_results()
        QCoreApplication.processEvents()
        self.assertTrue(window.results_dialog.isVisible())

    def test_active_drawing_class_is_visible_on_canvas_and_uses_project_class_model(self):
        window = self.make_window()
        window.annotation.set_classes(["car", "person", "truck"])
        QCoreApplication.processEvents()

        active = window.canvas_area.active_class_combo
        self.assertEqual(active.count(), 3)
        self.assertEqual(active.itemText(0), "car")
        self.assertEqual(active.itemText(2), "truck")

        active.setCurrentIndex(2)
        QCoreApplication.processEvents()
        self.assertEqual(window.annotation.class_combo.currentIndex(), 2)

    def test_canvas_has_explicit_zoom_fit_and_actual_size_controls(self):
        window = self.make_window()
        window.canvas.load_image(self.make_image())
        start_scale = window.canvas.transform().m11()

        window.actions.zoom_in.trigger()
        QCoreApplication.processEvents()
        self.assertGreater(window.canvas.transform().m11(), start_scale)

        window.actions.actual_size.trigger()
        QCoreApplication.processEvents()
        self.assertAlmostEqual(window.canvas.transform().m11(), 1.0, places=6)

        window.actions.fit.trigger()
        QCoreApplication.processEvents()
        self.assertTrue(window.canvas._auto_fit)

    def test_space_temporarily_enables_hand_pan(self):
        window = self.make_window()
        window.canvas.load_image(self.make_image())
        press = QKeyEvent(QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier)
        release = QKeyEvent(QEvent.KeyRelease, Qt.Key_Space, Qt.NoModifier)

        window.canvas.keyPressEvent(press)
        self.assertEqual(window.canvas.dragMode(), QGraphicsView.ScrollHandDrag)

        window.canvas.keyReleaseEvent(release)
        self.assertEqual(window.canvas.dragMode(), QGraphicsView.NoDrag)

    def test_fullscreen_is_a_real_window_command_not_the_fit_action(self):
        window = self.make_window()
        self.assertIsNot(window.actions.fullscreen, window.actions.fit)
        self.assertEqual(window.actions.fullscreen.shortcut().toString(), "F11")
        self.assertEqual(window.actions.fit.shortcut().toString(), "F")


if __name__ == "__main__":
    unittest.main()
