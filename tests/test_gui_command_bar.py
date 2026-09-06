import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QApplication, QToolButton
from gui.main_window import MainWindow


class CommandBarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.resize(960, 620)
        self.window.show()
        QCoreApplication.processEvents()

    def tearDown(self):
        self.window.controller = None
        self.window.close()
        self.window.deleteLater()
        QCoreApplication.processEvents()

    def test_project_commands_remain_visible_at_minimum_size(self):
        bar = self.window.command_bar
        buttons = [bar.open_button, bar.run_button, *bar._buttons.values()]
        for button in buttons:
            with self.subTest(button=button.text()):
                self.assertTrue(button.isVisible())
                self.assertTrue(bar.rect().contains(button.geometry()))
        extension = bar.findChild(QToolButton, "qt_toolbar_ext_button")
        self.assertFalse(extension and extension.isVisible())

    def test_navigation_and_history_are_next_to_canvas(self):
        area = self.window.canvas_area
        for button in (
            area.previous_button,
            area.next_button,
            area.tool_buttons[self.window.actions.undo],
            area.tool_buttons[self.window.actions.redo],
        ):
            self.assertEqual(button.toolButtonStyle(), Qt.ToolButtonIconOnly)
            self.assertTrue(button.isVisible())

    def test_open_and_assist_offer_explicit_whole_button_menus(self):
        bar = self.window.command_bar
        self.assertEqual(bar.open_button.popupMode(), QToolButton.InstantPopup)
        self.assertEqual(bar.run_button.popupMode(), QToolButton.InstantPopup)
        self.assertIn(self.window.actions.open_folder, bar.open_menu.actions())
        self.assertIn(self.window.actions.open_state, bar.open_menu.actions())
        self.assertIn(self.window.actions.run_remaining, bar.run_menu.actions())
        self.assertIn(self.window.actions.import_yolo, bar.run_menu.actions())
