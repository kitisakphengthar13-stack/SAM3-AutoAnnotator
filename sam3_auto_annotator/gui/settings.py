from __future__ import annotations

from PySide6.QtCore import QSettings


SETTINGS_VERSION = 1


class UiSettings:
    """Persist desktop preferences; project data remains in the project JSON file."""

    def __init__(self, settings=None):
        self._settings = settings or QSettings()

    def restore_window(self, window, splitter):
        if self._settings.value("ui/version", 0, int) != SETTINGS_VERSION:
            return
        geometry = self._settings.value("ui/main_window_geometry")
        window_state = self._settings.value("ui/main_window_state")
        splitter_state = self._settings.value("ui/workspace_splitter")
        if geometry:
            window.restoreGeometry(geometry)
        if window_state:
            window.restoreState(window_state, SETTINGS_VERSION)
        if splitter_state:
            splitter.restoreState(splitter_state)

    def save_window(self, window, splitter):
        self._settings.setValue("ui/version", SETTINGS_VERSION)
        self._settings.setValue("ui/main_window_geometry", window.saveGeometry())
        self._settings.setValue(
            "ui/main_window_state", window.saveState(SETTINGS_VERSION)
        )
        self._settings.setValue("ui/workspace_splitter", splitter.saveState())
        self._settings.sync()

    def last_directory(self):
        return str(self._settings.value("files/last_directory", ""))

    def set_last_directory(self, path):
        self._settings.setValue("files/last_directory", str(path))
