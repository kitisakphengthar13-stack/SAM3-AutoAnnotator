import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QImage, QKeyEvent
from PySide6.QtWidgets import QApplication, QDockWidget, QGraphicsView

from sam3_auto_annotator.core import Annotation, AnnotationSource, ImageRecord, ImageStatus
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

    def load_canvas(self, window):
        window.canvas.load_image(self.make_image())
        window.show_canvas(True)
        QCoreApplication.processEvents()

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
        closable = QDockWidget.DockWidgetFeature.DockWidgetClosable
        self.assertTrue(window.dataset_dock.features() & closable)
        self.assertTrue(window.annotation_dock.features() & closable)

    def test_closed_docks_have_explicit_view_menu_restore_actions(self):
        window = self.make_window()
        window.dataset_dock.close()
        QCoreApplication.processEvents()
        self.assertFalse(window.dataset_dock.isVisible())

        actions = [action for action in window.view_menu.actions() if action.text()]
        dataset_action = next(action for action in actions if action.text() == "Dataset Panel")
        dataset_action.trigger()
        QCoreApplication.processEvents()
        self.assertTrue(window.dataset_dock.isVisible())

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

    def test_export_entry_is_preflight_not_the_disk_write_action(self):
        window = self.make_window()
        reviewed = ImageRecord("reviewed.png", 0, status=ImageStatus.REVIEWED)
        pending = ImageRecord(
            "pending.png",
            1,
            status=ImageStatus.NOT_PREDICTED,
            annotations=[
                Annotation(
                    0,
                    "car",
                    (1, 1, 20, 20),
                    source=AnnotationSource.MANUAL,
                )
            ],
        )
        window.controller = SimpleNamespace(
            project=SimpleNamespace(images=[reviewed, pending])
        )
        window.actions.export.setEnabled(True)

        window.actions.export_dialog.trigger()
        QCoreApplication.processEvents()

        self.assertTrue(window.results_dialog.isVisible())
        self.assertIn("Reviewed images: 1/2", window.results.result_counts_label.text())
        self.assertIn("Unpredicted / failed: 1", window.results.result_counts_label.text())
        self.assertEqual(window.actions.export.text(), "Export Anyway")
        self.assertEqual(window.actions.export_dialog.shortcut().toString(), "Ctrl+E")
        self.assertTrue(window.actions.export.shortcut().isEmpty())

    def test_active_drawing_class_is_visible_and_independent_from_selected_class(self):
        window = self.make_window()
        window.annotation.set_classes(["car", "person", "truck"])
        window.annotation.class_combo.setCurrentIndex(0)
        QCoreApplication.processEvents()

        active = window.canvas_area.active_class_combo
        self.assertEqual(active.count(), 3)
        active.setCurrentIndex(2)
        QCoreApplication.processEvents()

        self.assertEqual(active.currentText(), "truck")
        self.assertEqual(window.annotation.class_combo.currentText(), "car")

    def test_canvas_tools_are_explicit_and_exclusive(self):
        window = self.make_window()
        self.load_canvas(window)
        self.assertTrue(window.actions.select_tool.isChecked())

        window.actions.pan_tool.trigger()
        QCoreApplication.processEvents()
        self.assertTrue(window.actions.pan_tool.isChecked())
        self.assertFalse(window.actions.select_tool.isChecked())
        self.assertEqual(window.canvas.dragMode(), QGraphicsView.ScrollHandDrag)

        window.actions.draw_box.trigger()
        QCoreApplication.processEvents()
        self.assertTrue(window.actions.draw_box.isChecked())
        self.assertFalse(window.actions.pan_tool.isChecked())
        self.assertTrue(window.canvas._draw_mode)
        self.assertEqual(window.canvas.dragMode(), QGraphicsView.NoDrag)

        press = QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
        window.canvas.keyPressEvent(press)
        self.assertTrue(window.actions.select_tool.isChecked())
        self.assertFalse(window.canvas._draw_mode)

    def test_canvas_object_labels_show_class_and_confidence(self):
        window = self.make_window()
        self.load_canvas(window)
        annotation = Annotation(
            0,
            "car",
            (10, 10, 80, 60),
            id="ann-label",
            confidence=0.91,
        )
        window.canvas.set_annotations([annotation])
        QCoreApplication.processEvents()

        box = window.canvas._items_by_id[annotation.id]
        text_items = [
            child for child in box.childItems() if hasattr(child, "text")
        ]
        self.assertEqual(len(text_items), 1)
        self.assertEqual(text_items[0].text(), "car  0.91")
        self.assertEqual(text_items[0].acceptedMouseButtons(), Qt.NoButton)

    def test_canvas_has_explicit_zoom_fit_and_actual_size_controls(self):
        window = self.make_window()
        self.load_canvas(window)
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

    def test_space_temporarily_enables_hand_pan_from_select_mode(self):
        window = self.make_window()
        self.load_canvas(window)
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
