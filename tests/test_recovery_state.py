import os
import tempfile
import unittest
from pathlib import Path

from domain import ProjectState
from storage.project_store import (
    RECOVERY_FILE_NAME,
    STATE_FILE_NAME,
    clear_recovery_state,
    load_project_state,
    newer_recovery_for,
    save_project_state,
    save_recovery_state,
)


class RecoveryStateTests(unittest.TestCase):
    def project(self, name):
        return ProjectState(
            input_path="fixture",
            prompts=[],
            images=[],
            project_name=name,
        )

    def test_recovery_snapshot_does_not_replace_manual_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / STATE_FILE_NAME
            save_project_state(self.project("manual"), state_path)
            save_recovery_state(self.project("recovered"), root)

            self.assertEqual(load_project_state(state_path).project_name, "manual")
            self.assertEqual(
                load_project_state(root / RECOVERY_FILE_NAME).project_name,
                "recovered",
            )

    def test_only_newer_recovery_is_offered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / STATE_FILE_NAME
            recovery_path = root / RECOVERY_FILE_NAME
            save_project_state(self.project("manual"), state_path)
            save_recovery_state(self.project("recovered"), root)

            os.utime(state_path, ns=(1_000_000_000, 1_000_000_000))
            os.utime(recovery_path, ns=(2_000_000_000, 2_000_000_000))
            self.assertEqual(newer_recovery_for(state_path), recovery_path)

            os.utime(state_path, ns=(3_000_000_000, 3_000_000_000))
            self.assertIsNone(newer_recovery_for(state_path))

    def test_clear_recovery_preserves_manual_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / STATE_FILE_NAME
            save_project_state(self.project("manual"), state_path)
            save_recovery_state(self.project("recovered"), root)

            clear_recovery_state(root)
            self.assertTrue(state_path.is_file())
            self.assertFalse((root / RECOVERY_FILE_NAME).exists())


if __name__ == "__main__":
    unittest.main()
