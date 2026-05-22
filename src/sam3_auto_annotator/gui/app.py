import sys

from PySide6.QtWidgets import QApplication

from sam3_auto_annotator.gui.fields import configure_c_locale
from sam3_auto_annotator.gui.main_window import MainWindow


def main():
    configure_c_locale()
    app = QApplication(sys.argv)
    app.setApplicationName("SAM3 AutoAnnotator")
    app.setOrganizationName("SAM3-AutoAnnotator")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
