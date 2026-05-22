from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from sam3_auto_annotator.gui.app import main
except ImportError as exc:
    message = (
        "Could not start the SAM3 AutoAnnotator GUI.\n"
        "Install the GUI dependency first with:\n\n"
        "  pip install -e .[gui]\n\n"
        f"Original import error: {exc}"
    )
    raise SystemExit(message) from exc


if __name__ == "__main__":
    raise SystemExit(main())
