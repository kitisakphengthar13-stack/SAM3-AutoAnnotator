import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QApplication

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

    def test_dense_navigation_and_history_actions_are_icon_only(self):
        bar = self.window.command_bar
        for action in (
            self.window.actions.previous_image,
            self.window.actions.next_image,
            self.window.actions.undo,
            self.window.actions.redo,
        ):
            with self.subTest(action=action.text()):
                self.assertEqual(
                    bar.tool_button(action).toolButtonStyle(),
                    Qt.ToolButtonIconOnly,
                )

    def test_secondary_open_folder_action_is_not_forced_into_global_toolbar(self):
        self.assertIsNone(
            self.window.command_bar.tool_button(self.window.actions.open_folder)
        )
        self.assertLessEqual(self.window.command_bar.project_label.maximumWidth(), 160)


if __name__ == "__main__":
    unittest.main()
