from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from sam3_auto_annotator.paths import EXPORT_FORMATS


@dataclass
class RunConfig:
    input: Optional[str] = None
    text: Optional[List[str]] = None
    model: Optional[str] = None
    conf: float = 0.7
    half: bool = True
    project_name: Optional[str] = None
    output_root: str = "outputs"
    timestamp: bool = False
    save_predictions: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["csv"])
    overwrite: bool = False
    run_summary: bool = True
    show: bool = False

    def to_namespace(self):
        return Namespace(
            input=self.input,
            text=self.text,
            model=self.model,
            conf=self.conf,
            half=self.half,
            project_name=self.project_name,
            output_root=self.output_root,
            timestamp=self.timestamp,
            save_predictions=self.save_predictions,
            export_formats=self.export_formats,
            overwrite=self.overwrite,
            run_summary=self.run_summary,
            show=self.show,
        )


def load_config(config_path):
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}

    config = RunConfig()

    input_config = data.get("input") or {}
    model_config = data.get("model") or {}
    output_config = data.get("output") or {}
    export_config = data.get("export") or {}

    if "path" in input_config:
        config.input = input_config["path"]

    if "path" in model_config:
        config.model = model_config["path"]
    if "confidence" in model_config:
        config.conf = model_config["confidence"]
    if "half" in model_config:
        config.half = model_config["half"]

    if "prompts" in data:
        config.text = data["prompts"]

    if "root" in output_config:
        config.output_root = output_config["root"]
    if "project_name" in output_config:
        config.project_name = output_config["project_name"]
    if "timestamp" in output_config:
        config.timestamp = output_config["timestamp"]
    if "overwrite" in output_config:
        config.overwrite = output_config["overwrite"]
    if "save_predictions" in output_config:
        config.save_predictions = output_config["save_predictions"]
    if "run_summary" in output_config:
        config.run_summary = output_config["run_summary"]
    if "show" in output_config:
        config.show = output_config["show"]

    if "formats" in export_config:
        config.export_formats = export_config["formats"]

    return config


def merge_cli_args(config, args):
    values = vars(args)

    mapping = {
        "input": "input",
        "text": "text",
        "model": "model",
        "conf": "conf",
        "half": "half",
        "project_name": "project_name",
        "output_root": "output_root",
        "timestamp": "timestamp",
        "save_predictions": "save_predictions",
        "export_formats": "export_formats",
        "overwrite": "overwrite",
        "run_summary": "run_summary",
        "show": "show",
    }

    for arg_name, config_name in mapping.items():
        if arg_name in values:
            setattr(config, config_name, values[arg_name])

    return config


def validate_config(config, parser):
    missing = []
    if not config.input:
        missing.append("--input or input.path")
    if not config.model:
        missing.append("--model or model.path")
    if not config.text:
        missing.append("--text or prompts")

    if missing:
        parser.error("Missing required values after config merge: " + ", ".join(missing))

    invalid_formats = [
        export_format
        for export_format in config.export_formats
        if export_format not in EXPORT_FORMATS
    ]
    if invalid_formats:
        parser.error(
            "Invalid export format(s): "
            + ", ".join(invalid_formats)
            + f". Choose from: {', '.join(EXPORT_FORMATS)}"
        )

    return config


def resolve_run_config(args, parser):
    if getattr(args, "config", None):
        config = load_config(args.config)
    else:
        config = RunConfig()

    config = merge_cli_args(config, args)
    return validate_config(config, parser).to_namespace()
