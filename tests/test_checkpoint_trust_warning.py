import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.actions import AppActions
from gui.views.setup_panel import SetupPanel


class CheckpointTrustWarningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_setup_warns_about_checkpoint_trust(self):
        actions = AppActions()
        panel = SetupPanel(actions)
        try:
            warning = panel.checkpoint_trust_label
            self.assertTrue(warning.isVisibleTo(panel))
            text = warning.text().casefold()
            self.assertIn("checkpoint", text)
            self.assertIn("trust", text)
        finally:
            panel.deleteLater()
            actions.deleteLater()


if __name__ == "__main__":
    unittest.main()
