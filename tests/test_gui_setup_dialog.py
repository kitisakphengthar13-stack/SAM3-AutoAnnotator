import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


class SetupDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.show()
        QCoreApplication.processEvents()

    def tearDown(self):
        self.window.controller = None
        self.window.close()
        self.window.deleteLater()
        QCoreApplication.processEvents()

    def test_cancel_restores_values_present_when_dialog_opened(self):
        self.window.setup.prompts_edit.setPlainText("car")
        self.window.show_setup()
        QCoreApplication.processEvents()

        self.window.setup.prompts_edit.setPlainText("truck")
        self.window.setup.cancel_button.click()
        QCoreApplication.processEvents()

        self.assertFalse(self.window.setup_dialog.isVisible())
        self.assertEqual(self.window.setup.prompts_text(), "car")

    def test_apply_emits_one_commit_signal_and_keeps_staged_values(self):
        commits = []
        self.window.setup.settings_changed.connect(lambda: commits.append(True))
        self.window.setup.prompts_edit.setPlainText("car")
        self.window.show_setup()
        QCoreApplication.processEvents()

        self.window.setup.prompts_edit.setPlainText("person")
        self.window.setup.apply_button.click()
        QCoreApplication.processEvents()

        self.assertEqual(commits, [True])
        self.assertFalse(self.window.setup_dialog.isVisible())
        self.assertEqual(self.window.setup.prompts_text(), "person")

    def test_validation_failure_blocks_commit_signal_entirely(self):
        commits = []
        updates = []
        self.window.setup.settings_changed.connect(lambda: commits.append(True))
        self.window.controller = SimpleNamespace(
            project=object(),
            _prompt_validation_error=lambda _prompts: "car is still in use",
            _update_actions=lambda: updates.append("actions"),
            _update_context=lambda: updates.append("context"),
        )
        self.window.setup.prompts_edit.setPlainText("car")
        self.window.show_setup()
        QCoreApplication.processEvents()

        self.window.setup.prompts_edit.setPlainText("truck")
        self.window.setup.model_path_edit.setText("draft-model.pt")
        self.window.setup.conf_edit.setValue(0.85)
        self.window.setup.apply_button.click()
        QCoreApplication.processEvents()

        self.assertEqual(commits, [])
        self.assertTrue(self.window.setup_dialog.isVisible())
        self.assertEqual(
            self.window.setup.prompt_validation_label.text(),
            "car is still in use",
        )
        self.assertEqual(updates, ["actions", "context"])

    def test_window_close_discards_unapplied_setup_changes(self):
        self.window.setup.model_path_edit.setText("baseline.pt")
        self.window.show_setup()
        QCoreApplication.processEvents()

        self.window.setup.model_path_edit.setText("draft.pt")
        self.window.setup_dialog.reject()
        QCoreApplication.processEvents()

        self.assertEqual(self.window.setup.model_path_edit.text(), "baseline.pt")


if __name__ == "__main__":
    unittest.main()
