# SAM3-AutoAnnotator

SAM3-AutoAnnotator is a Python command-line auto-annotation tool built on top of the Ultralytics SAM3 pipeline.

It runs SAM3 text-prompted segmentation on either a single image or a folder of images, then saves project-based annotation outputs including polygon CSV data, bounding-box CSV data, prediction visualization images, and a run summary.

This project does not implement SAM3 from scratch. It provides a reusable workflow around Ultralytics SAM3 inference for practical auto-annotation tasks.

## What This Project Does

SAM3-AutoAnnotator helps convert SAM3 prediction results into structured annotation outputs.

Given an image or image folder and one or more text prompts, the tool can:

- run SAM3 text-prompted segmentation
- extract polygon mask coordinates
- extract bounding boxes
- track class names, class IDs, confidence scores, and object counts
- save prediction visualization images for quality inspection
- export results into CSV files
- organize every run into a separate project folder

## Key Features

- Single-image input support
- Folder-based image input support
- Text-prompted SAM3 segmentation
- Project-based output organization
- Automatic prediction visualization saving
- Polygon mask export to CSV
- Bounding box export to CSV
- Per-image and total class counting
- Confidence score logging
- Run summary generation
- Automatic timestamping to prevent accidental overwrites
- Windows-friendly path support

## Quick Start

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run auto-annotation on an image folder:

```powershell
python sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text car --project-name car_annotation
```

Run auto-annotation on a single image:

```powershell
python sam3_auto_annotator.py --input "path\to\image.jpg" --model "path\to\sam3.pt" --text car
```

By default, the tool saves CSV outputs, `run_summary.json`, and SAM3 prediction visualization images.

## Example Output Structure

Each run creates a project folder under `outputs/` by default.

```text
outputs/
└── car_annotation/
    ├── sam3_auto_annotation_xyn_outputs.csv
    ├── sam3_auto_annotation_box_outputs.csv
    ├── run_summary.json
    └── prediction_results/
        ├── image_001_predicted.png
        ├── image_002_predicted.png
        └── image_003_predicted.png
```

## Important Note: SAM3 Access

This repository does not include SAM3 model weights.

Before using this tool, you must request access to the SAM3 model weights/checkpoints from the official SAM3 model checkpoint page. After access is approved, download the SAM3 weight file, such as `sam3.pt`, and pass its local path with `--model`.

The SAM3 model file should not be committed to this repository.

Example:

```powershell
python sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text car
```

## Backend

SAM3-AutoAnnotator uses the Ultralytics SAM3 pipeline as its inference backend.

Internally, the script uses:

```python
from ultralytics.models.sam import SAM3SemanticPredictor
```

The tool forwards the SAM3 model path, confidence threshold, and text prompts to Ultralytics `SAM3SemanticPredictor`.

This project focuses on the auto-annotation workflow around SAM3, including input collection, prediction execution, mask extraction, bounding-box extraction, CSV export, prediction visualization saving, and project-based output management.

## Supported Inputs

Use `--input` with either:

- a single supported image file
- a folder containing supported image files

Supported image formats:

```text
.jpg
.jpeg
.png
.bmp
.tif
.tiff
.webp
```

Folder input is currently non-recursive. Images inside subfolders are not processed.

## Usage

### Folder Input

```powershell
python sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text "siamese cat" dog car --project-name sample_run
```

### Single Image Input

```powershell
python sam3_auto_annotator.py --input "path\to\image.jpg" --model "path\to\sam3.pt" --text "siamese cat"
```

### Multiple Text Prompts

```powershell
python sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text person car dog
```

Multi-word prompts should be quoted:

```powershell
python sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text "siamese cat" "sports car"
```

### Always Append a Timestamp

```powershell
python sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text person --project-name people_run --timestamp
```

### Use a Custom Output Root

```powershell
python sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text car --output-root "annotation_outputs" --project-name car_run
```

### Disable Prediction Images

Prediction visualization images are saved by default.

Disable them with:

```powershell
python sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text person car --no-save-predictions
```

## Project Output Behavior

Outputs are written under `outputs/` by default.

- Use `--project-name` to choose the project folder name.
- Without `--project-name`, the script derives a name from the input and prompts.
- If the project folder already exists, a timestamp suffix is added automatically.
- Use `--timestamp` to always append a timestamp.
- Use `--overwrite` to allow writing into an existing project folder.
- Use `--output-root` to save projects somewhere other than `outputs/`.

This design prevents annotation results from different runs from being mixed together.

## Prediction Results

By default, each project folder includes SAM3 prediction visualization images in:

```text
prediction_results/
```

Each saved image uses the original image stem plus `_predicted.png`.

Example:

```text
2-Tesla-Model-S_predicted.png
```

These images are intended for visual inspection only. They help users quickly check whether SAM3 predictions look reasonable before using the exported annotation data.

Disable prediction image output with:

```powershell
--no-save-predictions
```

`--save-annotated` is kept as a legacy alias for `--save-predictions`.

## CSV Outputs

Each project folder contains two CSV files.

```text
sam3_auto_annotation_xyn_outputs.csv
sam3_auto_annotation_box_outputs.csv
```

### Polygon CSV

`sam3_auto_annotation_xyn_outputs.csv` stores normalized polygon segmentation data extracted from SAM3 mask outputs.

It includes:

- image path
- image name
- image index
- object index
- class ID
- class name
- per-image class count
- total class count
- polygon point count
- normalized polygon coordinates
- YOLO-style segmentation line
- confidence score

### Bounding Box CSV

`sam3_auto_annotation_box_outputs.csv` stores bounding box data extracted from SAM3 box outputs.

It includes:

- image path
- image name
- image index
- object index
- class ID
- class name
- per-image class count
- total class count
- absolute `xyxy` box coordinates
- box width and height
- box center coordinates
- normalized YOLO box values
- confidence score

## Run Summary

By default, the script writes `run_summary.json` into the project folder.

It includes:

- project name
- output folder
- input path
- model path
- text prompts
- confidence threshold
- images processed
- images with detections
- images with no detections
- total detections
- class counts
- generated output file paths
- prediction results folder path
- creation timestamp

Disable it with:

```powershell
--no-run-summary
```

## CLI Options

| Option | Required | Description |
|---|---:|---|
| `--input` | Yes | Path to one image file or a folder of images |
| `--model` | Yes | Path to the local SAM3 model weight file, such as `sam3.pt` |
| `--text` | Yes | One or more text prompts/classes |
| `--conf` | No | Confidence threshold. Default: `0.7` |
| `--half` / `--no-half` | No | Enable or disable fp16 inference. Default: enabled |
| `--project-name` | No | Name of the output project folder |
| `--output-root` | No | Root folder for output projects. Default: `outputs` |
| `--timestamp` | No | Always append a timestamp to the output folder name |
| `--overwrite` | No | Allow writing into an existing project output folder |
| `--save-predictions` / `--no-save-predictions` | No | Enable or disable prediction visualization images in `prediction_results/`. Default: enabled |
| `--save-annotated` | No | Legacy alias for `--save-predictions` |
| `--run-summary` / `--no-run-summary` | No | Enable or disable `run_summary.json`. Default: enabled |
| `--show` | No | Display prediction visualization images with matplotlib |

## Requirements

Install dependencies with:

```powershell
pip install -r requirements.txt
```

SAM3 support depends on the Ultralytics version installed in your environment. Make sure your Ultralytics installation supports SAM3.

## Notes and Limitations

- This repository does not include SAM3 model weights.
- Users must request access to SAM3 weights/checkpoints separately.
- Users must provide the local SAM3 model path with `--model`.
- Folder input is currently non-recursive.
- Prediction visualization images are for visual inspection only.
- Auto annotations should be reviewed before being used as final training labels.
- Output quality depends on the SAM3 model, prompt quality, image quality, and confidence threshold.
- The confidence threshold is forwarded to Ultralytics SAM3 inference through `--conf`.
- FP16 behavior depends on the device and backend. Use `--no-half` if FP16 is not supported in your environment.
- This tool currently exports CSV outputs. YOLO dataset export may be added in a future version.

## License and Attribution

This project is an auto-annotation utility built around Ultralytics SAM3 inference.

SAM3 is developed by Meta. This repository does not include SAM3 model weights. Users are responsible for requesting access to SAM3 weights/checkpoints and complying with the SAM license terms.

Ultralytics is used as the SAM3 inference pipeline through `SAM3SemanticPredictor`.

If you use SAM3 outputs in research or publication, acknowledge the use of SAM Materials according to the SAM license.

## Portfolio Summary

SAM3-AutoAnnotator demonstrates a practical computer vision annotation workflow using Python, Ultralytics SAM3, text-prompted segmentation, CSV export, prediction visualization saving, and project-based output organization.

The project focuses on building a reusable tool around an existing foundation model rather than training a segmentation model from scratch.