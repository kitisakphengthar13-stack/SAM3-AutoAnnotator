import unittest
from pathlib import Path

from app_paths import (
    APP_DATA_DIR,
    APP_HOME_ENV,
    MODELS_DIR,
    OUTPUTS_DIR,
    user_data_dir,
)
from logging_setup import LOG_DIR


class AppPathTests(unittest.TestCase):
    def test_windows_uses_local_app_data(self):
        result = user_data_dir(
            platform_name="win32",
            environ={"LOCALAPPDATA": r"C:\Users\fern\AppData\Local"},
            home=Path(r"C:\Users\fern"),
        )
        self.assertEqual(
            result,
            Path(r"C:\Users\fern\AppData\Local") / "SAM3-AutoAnnotator",
        )

    def test_linux_uses_xdg_data_home(self):
        result = user_data_dir(
            platform_name="linux",
            environ={"XDG_DATA_HOME": "/tmp/user-data"},
            home=Path("/home/fern"),
        )
        self.assertEqual(result, Path("/tmp/user-data/sam3-autoannotator"))

    def test_explicit_app_home_override_wins(self):
        result = user_data_dir(
            platform_name="win32",
            environ={
                APP_HOME_ENV: "/custom/sam3-home",
                "LOCALAPPDATA": "/should/not/win",
            },
            home=Path("/home/fern"),
        )
        self.assertEqual(result, Path("/custom/sam3-home"))

    def test_runtime_directories_share_the_application_home(self):
        self.assertEqual(MODELS_DIR.parent, APP_DATA_DIR)
        self.assertEqual(OUTPUTS_DIR.parent, APP_DATA_DIR)
        self.assertEqual(LOG_DIR.parent, APP_DATA_DIR)
        self.assertEqual(MODELS_DIR.name, "models")
        self.assertEqual(OUTPUTS_DIR.name, "projects")
        self.assertEqual(LOG_DIR.name, "logs")


if __name__ == "__main__":
    unittest.main()
