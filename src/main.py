from __future__ import annotations

import logging
import sys
import traceback

from PySide6.QtWidgets import QApplication, QMessageBox

from gui.controllers import WorkstationController
from gui.icons import ICONS, icon
from gui.main_window import MainWindow
from gui.settings import UiSettings
from gui.widgets.numeric_field import configure_c_locale
from logging_setup import configure_logging
from version import __version__


logger = logging.getLogger(__name__)


def create_application(argv=None):
    configure_c_locale()
    QApplication.setOrganizationName("SAM3-AutoAnnotator")
    QApplication.setOrganizationDomain("sam3-auto-annotator.local")
    QApplication.setApplicationName("SAM3 AutoAnnotator")
    QApplication.setApplicationDisplayName("SAM3 AutoAnnotator")
    QApplication.setApplicationVersion(__version__)

    app = QApplication.instance() or QApplication(
        list(sys.argv if argv is None else argv)
    )
    app.setWindowIcon(icon(ICONS["app"]))
    log_path = configure_logging()

    settings = UiSettings()
    window = MainWindow()
    WorkstationController(window, settings)
    window.ui_settings = settings
    settings.restore_window(window)
    window.diagnostic_log_path = log_path
    _install_exception_hook(window, log_path)
    return app, window


def _install_exception_hook(window, log_path):
    def handle_exception(error_type, error, tb):
        detail = "".join(traceback.format_exception(error_type, error, tb))
        logger.critical("Unhandled GUI exception\n%s", detail)
        support_hint = (
            f"\n\nDiagnostic log: {log_path}" if log_path is not None else ""
        )
        QMessageBox.critical(
            window,
            "Unexpected Error",
            "The application hit an unexpected error. Your saved project was not "
            f"modified by this message. Please retry the last action.{support_hint}",
        )

    sys.excepthook = handle_exception


def run(argv=None):
    app, window = create_application(argv)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
