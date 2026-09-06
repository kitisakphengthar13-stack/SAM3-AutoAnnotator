from __future__ import annotations

from PySide6.QtCore import QSettings

SETTINGS_VERSION = 3


class UiSettings:
    """Persist desktop preferences; project data remains in the project JSON file."""

    def __init__(self, settings=None):
        self._settings = settings or QSettings()

    @staticmethod
    def _responsive_area(window):
        area = getattr(window, "canvas_area", None)
        return area if hasattr(area, "_apply_responsive_workspace") else None

    def restore_window(self, window):
        if self._settings.value("ui/version", 0, int) != SETTINGS_VERSION:
            return
        geometry = self._settings.value("ui/main_window_geometry")
        window_state = self._settings.value("ui/main_window_state")
        if geometry:
            window.restoreGeometry(geometry)
        if window_state:
            window.restoreState(window_state, SETTINGS_VERSION)

        # Dock visibility created by the responsive breakpoint is derived UI state,
        # not a user preference. Re-derive it after Qt has restored geometry/state.
        area = self._responsive_area(window)
        if area is not None:
            area._responsive_dataset_override = None
            if window.width() < 1120:
                if window.dataset_dock.isVisible():
                    area._apply_responsive_workspace()
                else:
                    # Older settings may have persisted a Dataset dock that the
                    # responsive layout, rather than the user, had hidden.
                    area._responsive_dataset_auto_hidden = True
            else:
                area._responsive_dataset_auto_hidden = False

    def save_window(self, window):
        focused = window.actions.focus_workspace.isChecked()
        if focused:
            window.actions.focus_workspace.setChecked(False)

        area = self._responsive_area(window)
        responsive_hidden = bool(
            area is not None
            and area._responsive_dataset_auto_hidden
            and not window.dataset_dock.isVisible()
        )
        if responsive_hidden:
            # saveState() must capture user intent, not the narrow-layout projection.
            # Temporarily expose Dataset while signals are guarded, then put the
            # runtime narrow layout back exactly as it was.
            area._responsive_guard = True
            try:
                window.dataset_dock.show()
            finally:
                area._responsive_guard = False

        self._settings.setValue("ui/version", SETTINGS_VERSION)
        self._settings.setValue("ui/main_window_geometry", window.saveGeometry())
        self._settings.setValue(
            "ui/main_window_state", window.saveState(SETTINGS_VERSION)
        )
        self._settings.sync()

        if responsive_hidden:
            area._responsive_guard = True
            try:
                window.dataset_dock.hide()
            finally:
                area._responsive_guard = False
            area._responsive_dataset_auto_hidden = True

        if focused:
            window.actions.focus_workspace.setChecked(True)

    def last_directory(self):
        return str(self._settings.value("files/last_directory", ""))

    def set_last_directory(self, path):
        self._settings.setValue("files/last_directory", str(path))
