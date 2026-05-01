# Project Instructions

This is a Python computer vision project for SAM3 auto annotation using Ultralytics SAM3SemanticPredictor.

## Goals

Refactor the current single-file script into a clean package structure without changing the existing behavior.

## Hard rules

- Do not change the output behavior unless explicitly requested.
- Preserve current CLI options as much as possible.
- Keep CSV output fields compatible with the current script.
- Keep YOLO segmentation and detection label formats compatible.
- Keep run_summary.json compatible.
- Do not remove existing example commands from README unless replacing them with equivalent new commands.
- Do not introduce heavy dependencies unless necessary.
- Use pathlib for paths.
- Prefer dataclasses for structured config and annotation results.
- Keep code readable for a computer vision portfolio project.
- After every change, explain what changed and how to run it.

## Target structure

sam3-auto-annotator/
├── assets/
├── configs/
│   └── default.yaml
├── data/
│   └── images/
├── outputs/
├── src/
│   └── sam3_auto_annotator/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── pipeline.py
│       ├── predictor.py
│       ├── schemas.py
│       ├── paths.py
│       ├── visualization.py
│       ├── summary.py
│       └── exporters/
│           ├── __init__.py
│           ├── csv_exporter.py
│           └── yolo_exporter.py
├── scripts/
│   └── run_auto_annotator.py
├── requirements.txt
├── README.md
└── .gitignore

## Refactor stages

1. First inspect the current script and propose a migration plan.
2. Then create folders and move code into modules.
3. Then add config loading from YAML.
4. Then update CLI to support both direct arguments and --config.
5. Then update README.
6. Then run syntax checks.