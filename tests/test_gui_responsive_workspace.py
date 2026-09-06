import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.views.workspace import NARROW_WORKSPACE_BREAKPOINT


class ResponsiveWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.resize(1360, 840)
        self.window.show()
        QCoreApplication.processEvents()

    def tearDown(self):
        self.window.controller = None
        self.window.close()
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QCoreApplication.processEvents()

    def test_narrow_workspace_auto_hides_dataset_and_restores_when_wide(self):
        self.assertTrue(self.window.dataset_dock.isVisible())
        self.assertTrue(self.window.annotation_dock.isVisible())

        self.window.resize(NARROW_WORKSPACE_BREAKPOINT - 100, 700)
        QCoreApplication.processEvents()

        self.assertFalse(self.window.dataset_dock.isVisible())
        self.assertTrue(self.window.annotation_dock.isVisible())
        self.assertTrue(self.window.canvas_area._responsive_dataset_auto_hidden)

        self.window.resize(NARROW_WORKSPACE_BREAKPOINT + 200, 800)
        QCoreApplication.processEvents()

        self.assertTrue(self.window.dataset_dock.isVisible())
        self.assertTrue(self.window.annotation_dock.isVisible())
        self.assertFalse(self.window.canvas_area._responsive_dataset_auto_hidden)

    def test_user_can_reopen_dataset_at_narrow_width_without_it_being_hidden_again(self):
        self.window.resize(NARROW_WORKSPACE_BREAKPOINT - 100, 700)
        QCoreApplication.processEvents()
        self.assertFalse(self.window.dataset_dock.isVisible())

        self.window.dataset_dock.show()
        QCoreApplication.processEvents()
        self.window.resize(NARROW_WORKSPACE_BREAKPOINT - 80, 700)
        QCoreApplication.processEvents()

        self.assertTrue(self.window.dataset_dock.isVisible())
        self.assertTrue(self.window.canvas_area._responsive_dataset_override)

    def test_manual_hidden_dataset_at_wide_width_is_not_auto_restored(self):
        self.window.dataset_dock.hide()
        QCoreApplication.processEvents()
        self.assertFalse(self.window.dataset_dock.isVisible())

        self.window.resize(NARROW_WORKSPACE_BREAKPOINT - 100, 700)
        QCoreApplication.processEvents()
        self.window.resize(NARROW_WORKSPACE_BREAKPOINT + 200, 800)
        QCoreApplication.processEvents()

        self.assertFalse(self.window.dataset_dock.isVisible())


if __name__ == "__main__":
    unittest.main()
