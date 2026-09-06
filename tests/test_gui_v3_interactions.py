"""Pointer-driven regression checks for workstation v3.

Offscreen Qt exercises actual widget events. It does not certify Windows chrome,
GPU inference, or operating-system scaling behavior.
"""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import (
    QCoreApplication,
    QEvent,
    QPoint,
    QPointF,
    QSettings,
    Qt,
    QTimer,
)
from PySide6.QtGui import QImage, QKeyEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QStyle,
    QStyleOptionSpinBox,
    QStyleOptionComboBox,
)
from gui.main_window import MainWindow
from gui.controllers import WorkstationController
from gui.settings import UiSettings
from services.project_service import create_project, load_state
from test_gui_fields import FakeSettings, FakeTaskManager


class WorkstationV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        image = QImage(800, 600, QImage.Format_RGB32)
        image.fill(Qt.white)
        for n in range(3):
            image.save(str(self.root / f"image-{n}.png"))
        self.window = MainWindow()
        self.tasks = FakeTaskManager()
        self.controller = WorkstationController(
            self.window, FakeSettings(), task_manager=self.tasks
        )
        self.errors = []
        self.window.show_error = lambda *args, **kw: self.errors.append((args, kw))
        self.window.confirm = lambda *args, **kw: True
        self.window.show()
        self.window.resize(960, 620)
        QCoreApplication.processEvents()
        project = create_project(self.root, prompts=["car", "person"])
        self.controller.projects.load_project(project)
        self.window.setup.output_dir_edit.setText(str(self.root / "output"))
        QCoreApplication.processEvents()

    def tearDown(self):
        self.window.controller = None
        self.window.close()
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QCoreApplication.processEvents()
        self.temp.cleanup()

    def add_box(self):
        self.controller.annotations.add_manual_box((100, 100, 350, 350))
        QCoreApplication.processEvents()
        return self.controller.selected_annotation

    def test_pan_over_box_changes_view_without_editing_annotation(self):
        annotation = self.add_box()
        canvas = self.window.canvas
        before = annotation.box_xyxy
        canvas.actual_size()
        canvas.zoom_in()
        canvas.centerOn(300, 300)
        self.window.actions.pan_tool.trigger()
        QCoreApplication.processEvents()
        pos = canvas.mapFromScene(QPointF(220, 220))
        initial = canvas.horizontalScrollBar().value()
        QTest.mousePress(canvas.viewport(), Qt.LeftButton, pos=pos)
        QTest.mouseMove(canvas.viewport(), pos + QPoint(45, 10))
        QTest.mouseRelease(canvas.viewport(), Qt.LeftButton, pos=pos + QPoint(45, 10))
        self.assertEqual(annotation.box_xyxy, before)
        self.assertNotEqual(canvas.horizontalScrollBar().value(), initial)
        self.assertFalse(canvas.isInteractive())
        self.assertEqual(self.window.undo_stack.count(), 1)

    def test_draw_with_pointer_then_undo_redo_and_save_reload_export(self):
        w = self.window
        c = self.controller
        canvas = w.canvas
        w.actions.draw_box.trigger()
        QCoreApplication.processEvents()
        start = canvas.mapFromScene(QPointF(80, 90))
        end = canvas.mapFromScene(QPointF(360, 390))
        QTest.mousePress(canvas.viewport(), Qt.LeftButton, pos=start)
        QTest.mouseMove(canvas.viewport(), end)
        QTest.mouseRelease(canvas.viewport(), Qt.LeftButton, pos=end)
        self.assertEqual(len(c.current_image.active_annotations), 1)
        self.assertEqual(c.current_image.active_annotations[0].class_name, "car")
        w.actions.undo.trigger()
        self.assertEqual(len(c.current_image.active_annotations), 0)
        w.actions.redo.trigger()
        self.assertEqual(len(c.current_image.active_annotations), 1)
        c.projects.save_project()
        saved = self.root / "output" / "annotation_state.json"
        self.assertTrue(saved.is_file())
        self.assertFalse(c.dirty)
        loaded = load_state(saved)
        self.assertEqual(len(loaded.images[0].active_annotations), 1)
        w.actions.export_dialog.trigger()
        QCoreApplication.processEvents()
        self.assertFalse((self.root / "output" / "yolo_labels").exists())
        QTest.mouseClick(w.results.export_button, Qt.LeftButton)
        self.assertTrue(
            (
                self.root / "output" / "yolo_labels" / "detection" / "image-0.txt"
            ).is_file()
        )
        self.assertEqual(w.results.phase, "complete")
        self.assertFalse(w.results.export_button.isVisible())
        self.assertEqual(self.errors, [])

    def test_pointer_move_and_resize_preserve_undo_and_selection(self):
        ann = self.add_box()
        w = self.window
        canvas = w.canvas
        c = self.controller
        w.actions.select_tool.trigger()
        before = ann.box_xyxy
        start = canvas.mapFromScene(QPointF(220, 220))
        end = start + QPoint(20, 15)
        QTest.mousePress(canvas.viewport(), Qt.LeftButton, pos=start)
        QTest.mouseMove(canvas.viewport(), end)
        QTest.mouseRelease(canvas.viewport(), Qt.LeftButton, pos=end)
        changed = c.selected_annotation.box_xyxy
        self.assertNotEqual(changed, before)
        self.assertAlmostEqual(changed[2] - changed[0], before[2] - before[0])
        self.assertEqual(c.selected_annotation_id, ann.id)
        w.actions.undo.trigger()
        self.assertEqual(c.selected_annotation.box_xyxy, before)
        item = canvas._items_by_id[ann.id]
        handle = item._handle_items["bottom_right"]
        start = canvas.mapFromScene(handle.scenePos())
        end = start + QPoint(18, 12)
        QTest.mousePress(canvas.viewport(), Qt.LeftButton, pos=start)
        QTest.mouseMove(canvas.viewport(), end)
        QTest.mouseRelease(canvas.viewport(), Qt.LeftButton, pos=end)
        resized = c.selected_annotation.box_xyxy
        self.assertGreater(resized[2], before[2])
        self.assertGreater(resized[3], before[3])
        w.actions.undo.trigger()
        self.assertEqual(c.selected_annotation.box_xyxy, before)
        self.assertEqual(self.errors, [])

    def test_escape_cancels_unfinished_box(self):
        w = self.window
        canvas = w.canvas
        w.actions.draw_box.trigger()
        canvas.setFocus()
        start = canvas.mapFromScene(QPointF(80, 90))
        end = canvas.mapFromScene(QPointF(200, 200))
        QTest.mousePress(canvas.viewport(), Qt.LeftButton, pos=start)
        QTest.mouseMove(canvas.viewport(), end)
        self.assertTrue(canvas._drawing)
        QTest.keyClick(canvas, Qt.Key_Escape)
        QTest.mouseRelease(canvas.viewport(), Qt.LeftButton, pos=end)
        self.assertIsNone(canvas._draft_item)
        self.assertFalse(canvas._drawing)
        self.assertTrue(w.actions.select_tool.isChecked())
        self.assertEqual(self.controller.current_image.active_annotations, [])

    def test_space_pan_restores_box_tool_on_release_and_focus_loss(self):
        w = self.window
        canvas = w.canvas
        w.actions.draw_box.trigger()
        QTest.keyPress(canvas, Qt.Key_Space)
        self.assertFalse(canvas._draw_mode)
        self.assertFalse(canvas.isInteractive())
        QTest.keyRelease(canvas, Qt.Key_Space)
        self.assertTrue(canvas._draw_mode)
        QTest.keyPress(canvas, Qt.Key_Space)
        QApplication.sendEvent(canvas, QEvent(QEvent.FocusOut))
        self.assertFalse(canvas._temporary_pan)
        self.assertTrue(canvas._draw_mode)

    def test_wheel_zoom_is_bounded_and_zero_delta_is_noop(self):
        canvas = self.window.canvas

        def wheel(delta):
            event = QWheelEvent(
                QPointF(100, 100),
                QPointF(100, 100),
                QPoint(),
                QPoint(0, delta),
                Qt.NoButton,
                Qt.NoModifier,
                Qt.NoScrollPhase,
                False,
            )
            QApplication.sendEvent(canvas.viewport(), event)

        for _ in range(100):
            wheel(120)
        self.assertAlmostEqual(canvas.transform().m11(), 20.0)
        for _ in range(100):
            wheel(-120)
        self.assertAlmostEqual(canvas.transform().m11(), 0.05)
        wheel(0)
        self.assertAlmostEqual(canvas.transform().m11(), 0.05)

    def test_selection_does_not_reopen_hidden_or_focused_objects_panel(self):
        ann = self.add_box()
        w = self.window
        w.annotation_dock.close()
        self.controller.annotations.select_annotation(ann.id)
        self.assertFalse(w.annotation_dock.isVisible())
        w.reset_workspace_layout()
        w.actions.focus_workspace.setChecked(True)
        self.controller.annotations.select_annotation(ann.id)
        self.assertFalse(w.annotation_dock.isVisible())
        self.assertFalse(w.dataset_dock.isVisible())
        w.actions.focus_workspace.setChecked(False)
        self.assertTrue(w.annotation_dock.isVisible())

    def test_dock_controls_close_float_and_restore(self):
        w = self.window
        dock = w.dataset_dock
        title = dock.titleBarWidget()
        QTest.mouseClick(title.float_button, Qt.LeftButton)
        self.assertTrue(dock.isFloating())
        QTest.mouseClick(title.float_button, Qt.LeftButton)
        self.assertFalse(dock.isFloating())
        QTest.mouseClick(title.close_button, Qt.LeftButton)
        self.assertFalse(dock.isVisible())
        dock.toggleViewAction().trigger()
        self.assertTrue(dock.isVisible())

    def test_fullscreen_round_trip_restores_maximized(self):
        w = self.window
        w.showMaximized()
        QCoreApplication.processEvents()
        w.actions.fullscreen.trigger()
        QCoreApplication.processEvents()
        self.assertTrue(w.isFullScreen())
        self.assertTrue(w.actions.fullscreen.isChecked())
        w.actions.fullscreen.trigger()
        QCoreApplication.processEvents()
        self.assertFalse(w.isFullScreen())
        self.assertTrue(w.isMaximized())
        self.assertFalse(w.actions.fullscreen.isChecked())

    def test_open_popup_responds_across_the_entire_button(self):
        button = self.window.command_bar.open_button
        menu = button.menu()
        for fraction in (0.1, 0.5, 0.9):
            observed = []

            def observe():
                observed.append(menu.isVisible())
                menu.close()

            QTimer.singleShot(30, observe)
            QTest.mouseClick(
                button,
                Qt.LeftButton,
                pos=QPoint(int(button.width() * fraction), button.height() // 2),
            )
            self.assertEqual(observed, [True])

    def test_confidence_arrows_work_in_both_directions(self):
        w = self.window
        w.show_setup()
        QCoreApplication.processEvents()
        spin = w.setup.conf_edit
        option = QStyleOptionSpinBox()
        spin.initStyleOption(option)
        for control, expected in (
            (QStyle.SC_SpinBoxUp, 0.55),
            (QStyle.SC_SpinBoxDown, 0.5),
        ):
            rect = spin.style().subControlRect(QStyle.CC_SpinBox, option, control, spin)
            self.assertGreaterEqual(rect.width(), 24)
            QTest.mouseClick(spin, Qt.LeftButton, pos=rect.center())
            self.assertAlmostEqual(spin.value(), expected)

    def test_class_text_fits_and_geometry_is_contextual_at_minimum_size(self):
        w = self.window
        combo = w.canvas_area.active_class_combo
        option = QStyleOptionComboBox()
        combo.initStyleOption(option)
        rect = combo.style().subControlRect(
            QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, combo
        )
        self.assertGreaterEqual(
            rect.width(), combo.fontMetrics().horizontalAdvance("car")
        )
        self.assertFalse(w.annotation.details_scroll.isVisible())
        self.add_box()
        self.assertTrue(w.annotation.details_scroll.isVisible())
        self.assertFalse(w.annotation.coordinates_widget.isVisible())
        before = self.controller.selected_annotation.box_xyxy
        QTest.mouseClick(w.annotation.coordinates_button, Qt.LeftButton)
        self.assertTrue(w.annotation.coordinates_dialog.isVisible())
        self.assertTrue(w.annotation.coordinates_widget.isVisible())
        w.annotation.x1_edit.set_value(120)
        QTest.mouseClick(w.annotation.cancel_coordinates_button, Qt.LeftButton)
        self.assertFalse(w.annotation.coordinates_dialog.isVisible())
        self.assertEqual(w.annotation.x1_edit.value(), before[0])
        self.assertEqual(self.controller.selected_annotation.box_xyxy, before)
        QTest.mouseClick(w.annotation.coordinates_button, Qt.LeftButton)
        w.annotation.x1_edit.set_value(125)
        QTest.mouseClick(w.annotation.apply_box_button, Qt.LeftButton)
        self.assertFalse(w.annotation.coordinates_dialog.isVisible())
        self.assertEqual(self.controller.selected_annotation.box_xyxy[0], 125)
        w.actions.undo.trigger()
        self.assertEqual(self.controller.selected_annotation.box_xyxy, before)

    def test_focus_layout_save_does_not_permanently_hide_panels(self):
        w = self.window
        settings = UiSettings(QSettings(str(self.root / "ui.ini"), QSettings.IniFormat))
        w.actions.focus_workspace.setChecked(True)
        settings.save_window(w)
        restored = MainWindow()
        restored.show()
        settings.restore_window(restored)
        QCoreApplication.processEvents()
        try:
            self.assertTrue(restored.dataset_dock.isVisible())
            self.assertTrue(restored.annotation_dock.isVisible())
        finally:
            restored.close()
            restored.deleteLater()
