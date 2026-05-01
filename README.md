# SAM3-AutoAnnotator

SAM3-AutoAnnotator is a Python CLI tool for SAM3 text-prompted auto-annotation using Ultralytics `SAM3SemanticPredictor`.

The tool runs SAM3 on one image or a folder of images, then exports practical annotation outputs for inspection, debugging, and downstream training workflows:

- polygon and bounding-box CSV files
- YOLO segmentation labels
- YOLO detection labels
- prediction visualization images
- `run_summary.json`

This project does not train or reimplement SAM3. It builds a reusable annotation workflow around Ultralytics SAM3 inference.

## Features

- One image or non-recursive image folder input
- Text prompt class names, such as `car`, `person`, or `"traffic light"`
- CSV polygon export with normalized mask points
- CSV bounding-box export with absolute and normalized box values
- YOLO segmentation TXT export
- YOLO detection TXT export
- Prediction visualization images saved by default
- Per-image and total class counts
- Confidence score logging
- `run_summary.json` for each run
- Optional YAML config file
- Direct CLI arguments override config values
- Project-based output folders under `outputs/`

## Project Structure

```text
SAM3-AutoAnnotator/
|-- configs/
|   `-- default.yaml
|-- data/
|-- models/
|-- outputs/
|-- scripts/
|   `-- run_auto_annotator.py
|-- src/
|   `-- sam3_auto_annotator/
|       |-- __main__.py
|       |-- cli.py
|       |-- config.py
|       |-- paths.py
|       |-- predictor.py
|       |-- pipeline.py
|       |-- summary.py
|       |-- visualization.py
|       `-- exporters/
|           |-- csv_exporter.py
|           `-- yolo_exporter.py
|-- run_sam3_auto_annotator.py
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Installation

Install the project in editable mode:

```powershell
pip install -e .
```

Verify the package entry points:

```powershell
sam3-auto-annotator --help
python -m sam3_auto_annotator --help
```

Local wrapper scripts are also available:

```powershell
python run_sam3_auto_annotator.py --help
python scripts/run_auto_annotator.py --help
```

## SAM3 Model Weights

This repository does not include SAM3 model weights.

Request access to the SAM3 weights/checkpoints, download the model file, and keep it outside Git tracking. A convenient local path is:

```text
models/sam3.pt
```

Then pass it with `--model` or configure it in `configs/default.yaml`.

## Quick Start With Direct CLI

Put images in `data/images`, place the model at `models/sam3.pt`, then run:

```powershell
sam3-auto-annotator --input data/images --model models/sam3.pt --text car --export-formats csv yolo --project-name demo_run
```

Single-image input is also supported:

```powershell
sam3-auto-annotator --input data/images/image_001.jpg --model models/sam3.pt --text car --project-name single_image_demo
```

Multiple prompts keep their order as class IDs:

```powershell
sam3-auto-annotator --input data/images --model models/sam3.pt --text person car dog --export-formats csv yolo
```

Multi-word prompts should be quoted:

```powershell
sam3-auto-annotator --input data/images --model models/sam3.pt --text "sports car" "traffic light"
```

## Config-Based Usage

Config files are optional. Direct CLI usage works without them.

`configs/default.yaml` follows this schema:

```yaml
input:
  path: data/images

model:
  path: models/sam3.pt
  confidence: 0.7
  half: true

prompts:
  - car

output:
  root: outputs
  project_name: null
  timestamp: false
  overwrite: false
  save_predictions: true
  run_summary: true
  show: false

export:
  formats:
    - csv
```

Run from config:

```powershell
sam3-auto-annotator --config configs/default.yaml
```

Direct CLI arguments override config values:

```powershell
sam3-auto-annotator --config configs/default.yaml --conf 0.5 --project-name demo_override
```

You can also override input, model, prompts, output root, export formats, and boolean flags:

```powershell
sam3-auto-annotator --config configs/default.yaml --input data/images --model models/sam3.pt --text car truck --export-formats csv yolo
```

## Output Structure

Each run writes to a project folder under `outputs/`.

```text
outputs/<project_name>/
|-- sam3_auto_annotation_xyn_outputs.csv
|-- sam3_auto_annotation_box_outputs.csv
|-- prediction_results/
|   `-- <image_stem>_predicted.png
|-- yolo_labels/
|   |-- segmentation/
|   |   `-- <image_stem>.txt
|   `-- detection/
|       `-- <image_stem>.txt
`-- run_summary.json
```

Some folders/files depend on selected options:

- CSV files are created when `csv` export is enabled.
- `yolo_labels/` is created when `yolo` export is enabled.
- `prediction_results/` is created when prediction saving is enabled.
- `run_summary.json` is created when run summary output is enabled.

## Output Formats

### XYN Polygon CSV

`sam3_auto_annotation_xyn_outputs.csv` stores normalized polygon data from SAM3 masks.

It includes:

- image path and image name
- image index
- object index
- class ID and class name
- per-image class count
- total class count
- polygon point count
- normalized polygon points
- YOLO segmentation line
- confidence score

The `polygon_xyn` field stores points as:

```text
x1,y1 x2,y2 x3,y3 ...
```

The `yolo_segmentation_line` field stores:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

### Bounding Box CSV

`sam3_auto_annotation_box_outputs.csv` stores bounding-box data from SAM3 box outputs.

It includes:

- image path and image name
- image index
- object index
- class ID and class name
- per-image class count
- total class count
- absolute `xyxy` box values
- box width and height
- box center
- normalized YOLO `xywh` values
- confidence score

### YOLO Segmentation Labels

YOLO segmentation labels are written to:

```text
yolo_labels/segmentation/
```

Format:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

Coordinates are normalized polygon points from `result.masks.xyn`.

### YOLO Detection Labels

YOLO detection labels are written to:

```text
yolo_labels/detection/
```

Format:

```text
class_id x_center y_center width height
```

Values are normalized. The tool uses `result.boxes.xywhn` when available, or computes normalized values from `result.boxes.xyxy` and the original image size.

If an image has no detections, empty YOLO `.txt` files are still created for that image when YOLO export is enabled.

### Prediction Visualizations

Prediction visualizations are saved by default:

```text
prediction_results/<image_stem>_predicted.png
```

Disable them with:

```powershell
--no-save-predictions
```

`--save-annotated` is kept as a legacy alias for `--save-predictions`.

### Run Summary

By default, each run writes:

```text
run_summary.json
```

It records:

- project name
- output folder
- selected export formats
- input path
- model path
- prompts
- confidence threshold
- number of processed images
- images with detections
- images with no detections
- total detections
- class counts
- generated output paths
- creation timestamp

Disable it with:

```powershell
--no-run-summary
```

## Example Workflow

1. Install the package:

   ```powershell
   pip install -e .
   ```

2. Place images in:

   ```text
   data/images
   ```

3. Place the SAM3 model at:

   ```text
   models/sam3.pt
   ```

4. Choose text prompt class names, for example:

   ```text
   car
   ```

5. Run annotation:

   ```powershell
   sam3-auto-annotator --input data/images --model models/sam3.pt --text car --export-formats csv yolo --project-name car_demo
   ```

6. Inspect prediction images:

   ```text
   outputs/car_demo/prediction_results
   ```

7. Review CSV and YOLO labels:

   ```text
   outputs/car_demo/sam3_auto_annotation_xyn_outputs.csv
   outputs/car_demo/sam3_auto_annotation_box_outputs.csv
   outputs/car_demo/yolo_labels
   ```

8. Check the run summary:

   ```text
   outputs/car_demo/run_summary.json
   ```

## CLI Options

| Option | Required | Description |
|---|---:|---|
| `--input` | Yes, unless provided by config | Path to one image file or a folder of images |
| `--model` | Yes, unless provided by config | Path to the local SAM3 model file |
| `--text` | Yes, unless provided by config | One or more text prompt class names |
| `--config` | No | Path to a YAML config file |
| `--conf` | No | Confidence threshold. Default: `0.7` |
| `--half` / `--no-half` | No | Enable or disable fp16 inference. Default: enabled |
| `--project-name` | No | Name of the output project folder |
| `--output-root` | No | Root folder for output projects. Default: `outputs` |
| `--export-formats` | No | One or more output formats: `csv`, `yolo`. Default: `csv` |
| `--timestamp` | No | Always append a timestamp to the output folder name |
| `--overwrite` | No | Allow writing into an existing project output folder |
| `--save-predictions` / `--no-save-predictions` | No | Enable or disable prediction visualization images. Default: enabled |
| `--save-annotated` | No | Legacy alias for `--save-predictions` |
| `--run-summary` / `--no-run-summary` | No | Enable or disable `run_summary.json`. Default: enabled |
| `--show` | No | Display prediction visualization images with matplotlib |

## Important Notes / Limitations

- Folder input is non-recursive. Only supported image files directly inside the folder are processed.
- Supported image extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, and `.webp`.
- Prompt order defines class ID order. For example, `--text person car dog` maps `person` to class `0`, `car` to class `1`, and `dog` to class `2`.
- Config files are optional.
- Direct CLI arguments override config values.
- Exact detections depend on the SAM3 model, image quality, prompt quality, and confidence threshold.
- Ultralytics may auto-adjust image size to match model stride. This is not necessarily an error.
- Prediction visualizations are for review and debugging.
- Auto annotations should be reviewed before use as final labels.
- YOLO export does not copy source images.
- YOLO export does not generate `data.yaml` or train/val/test splits yet.
- This repository does not include SAM3 model weights.

## Smoke Test Status

Runtime smoke tests passed with a real SAM3 model and real image inputs.

Tested:

- single-image input
- 222-image folder input
- CSV export
- YOLO segmentation and detection export
- `--no-save-predictions`
- `--no-run-summary`
- `--no-half`
- config mode with `--config`
- direct CLI override of config values
- package command: `sam3-auto-annotator`
- module command: `python -m sam3_auto_annotator`
- local wrapper commands

The exact detection counts are intentionally not treated as fixed test expectations because SAM3 output can vary by model, prompts, image content, confidence threshold, and environment.

## Attribution

SAM3 is developed by Meta. This project does not include SAM3 weights. Users are responsible for obtaining access to SAM3 weights/checkpoints and following the applicable license terms.

Ultralytics provides the SAM3 inference integration used by this package through `SAM3SemanticPredictor`.

