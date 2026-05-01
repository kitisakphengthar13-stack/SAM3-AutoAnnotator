from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sam3_auto_annotator.cli import main  # noqa: E402


if __name__ == "__main__":
    main()


# Example commands:
# python run_sam3_auto_annotator.py --input "path\to\image.jpg" --model "path\to\sam3.pt" --text "siamese cat"
# python run_sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text "siamese cat" dog car --project-name sample_run
