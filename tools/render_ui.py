"""Render real Qt screens with manually annotated fixtures, without a SAM3 model.

Run from any directory:
    python tools/render_ui.py --output-dir /path/to/screenshots
Use QT_SCALE_FACTOR=1.25 or 1.5 for additional Qt scaling captures.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QSettings, Qt, qVersion
from PySide6.QtWidgets import QApplication
from gui.controllers import WorkstationController
from gui.main_window import MainWindow
from gui.settings import UiSettings
from services.project_service import create_project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    measurements = {
        "qt": qVersion(),
        "platform": app.platformName(),
        "scale_factor": os.environ.get("QT_SCALE_FACTOR", "1"),
        "screens": {},
    }
    with tempfile.TemporaryDirectory(prefix="sam3-ui-") as temp:
        temp = Path(temp)
        window = MainWindow()
        settings = UiSettings(
            QSettings(str(temp / "settings.ini"), QSettings.IniFormat)
        )
        controller = WorkstationController(window, settings)
        window.show()

        def capture(name, widget=window):
            for _ in range(3):
                QCoreApplication.processEvents()
            pixmap = widget.grab()
            if not pixmap.save(str(destination / f"{name}.png")):
                raise RuntimeError(f"Could not save {name}")
            measurements["screens"][name] = {
                "logical_size": list(widget.size().toTuple()),
                "pixels": list(pixmap.size().toTuple()),
                "canvas": list(window.canvas.size().toTuple()),
            }

        capture("welcome")
        project = create_project(
            ROOT / "tests/fixtures/images", prompts=["car", "person", "traffic light"]
        )
        project.project_name = "Vehicle dataset"
        controller.projects.load_project(project)
        window.setup.output_dir_edit.setText(str(temp / "vehicle-dataset"))
        image = controller.current_image
        controller.annotations.add_manual_box(
            (
                image.width * 0.06,
                image.height * 0.1,
                image.width * 0.96,
                image.height * 0.94,
            )
        )
        capture("workspace-1360")
        window.resize(960, 620)
        capture("workspace-960")
        window.annotation.show_coordinates()
        capture("coordinates", window.annotation.coordinates_dialog)
        window.annotation.coordinates_dialog.reject()
        window.actions.focus_workspace.setChecked(True)
        capture("focus-960")
        window.actions.focus_workspace.setChecked(False)
        window.show_setup()
        capture("setup", window.setup_dialog)
        window.setup.tabs.setCurrentIndex(1)
        capture("setup-files", window.setup_dialog)
        window.setup_dialog.reject()
        window.show_export_preflight()
        capture("export-preflight", window.results_dialog)
        # This writes only into the temporary demonstration project.
        window.actions.export.trigger()
        capture("export-complete", window.results_dialog)
        window.results.tabs.setCurrentIndex(1)
        capture("export-files", window.results_dialog)
        window.results_dialog.accept()
        window.show_shortcuts()
        capture("shortcuts", window.shortcuts_dialog)
        window.shortcuts_dialog.accept()
        window.dataset.search_edit.setText("no matching image")
        capture("no-matches")
        window.dataset.search_edit.clear()
        window.show_canvas_error("unreadable.png", "The image could not be decoded.")
        capture("image-error")
        window.controller = None
        window.close()
    (destination / "measurements.json").write_text(
        json.dumps(measurements, indent=2) + "\n"
    )
    print(json.dumps(measurements, indent=2))


if __name__ == "__main__":
    main()
