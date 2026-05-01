import argparse

from sam3_auto_annotator.config import resolve_run_config
from sam3_auto_annotator.paths import EXPORT_FORMATS
from sam3_auto_annotator.pipeline import run


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    known_args, _ = config_parser.parse_known_args()
    using_config = known_args.config is not None

    parser = argparse.ArgumentParser(
        description="Run Ultralytics SAM3 auto-annotation on one image or a folder of images."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file. Direct CLI arguments override config values.",
    )
    parser.add_argument(
        "--input",
        required=not using_config,
        default=argparse.SUPPRESS if using_config else None,
        help="Path to one image file or a folder of images.",
    )
    parser.add_argument(
        "--text",
        nargs="+",
        required=not using_config,
        default=argparse.SUPPRESS if using_config else None,
        help="Text prompt class names. Example: --text \"siamese cat\" dog car",
    )
    parser.add_argument(
        "--model",
        required=not using_config,
        default=argparse.SUPPRESS if using_config else None,
        help="Path to the SAM3 model file.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=argparse.SUPPRESS if using_config else 0.7,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS if using_config else True,
        help="Use fp16 inference when supported by the selected device/backend.",
    )
    parser.add_argument(
        "--project-name",
        default=argparse.SUPPRESS if using_config else None,
        help="Name of the auto-annotation project output folder.",
    )
    parser.add_argument(
        "--output-root",
        default=argparse.SUPPRESS if using_config else "outputs",
        help="Root folder where project output folders will be created.",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        default=argparse.SUPPRESS if using_config else False,
        help="Always append a timestamp to the project output folder name.",
    )
    parser.add_argument(
        "--save-predictions",
        "--save-annotated",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS if using_config else True,
        dest="save_predictions",
        help=(
            "Save SAM3 prediction visualization images into prediction_results/. "
            "--save-annotated is a legacy alias."
        ),
    )
    parser.add_argument(
        "--export-formats",
        nargs="+",
        choices=EXPORT_FORMATS,
        default=argparse.SUPPRESS if using_config else ["csv"],
        help="Output formats to create. Use one or more of: csv yolo. Default: csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=argparse.SUPPRESS if using_config else False,
        help="Allow writing into an existing project output folder.",
    )
    parser.add_argument(
        "--run-summary",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS if using_config else True,
        help="Save run_summary.json in the project output folder.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=argparse.SUPPRESS if using_config else False,
        help="Show prediction visualization images with matplotlib.",
    )
    args = parser.parse_args()
    return resolve_run_config(args, parser)


def main():
    args = parse_args()
    run(args)
