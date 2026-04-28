# SAM3-AutoAnnotator

SAM3-AutoAnnotator is a command-line auto-annotation tool built on top of the Ultralytics SAM3 pipeline. It runs SAM3 text-prompted segmentation on either one image or a folder of images, then writes project-based outputs that include polygon segmentation CSV data, bounding-box CSV data, a run summary, and optional annotated preview images.

This project does not implement SAM3 from scratch. It provides a reusable Python CLI workflow around Ultralytics SAM3 inference for auto-annotation tasks.

## Key Features

- Run SAM3 text-prompted segmentation from the command line
- Support single-image input
- Support folder-based image input
- Extract normalized polygon mask coordinates from SAM3 outputs
- Extract bounding box coordinates from SAM3 outputs
- Export polygon annotation results to CSV
- Export bounding box annotation results to CSV
- Track class names, class IDs, confidence scores, and object counts
- Create project-based output folders for each annotation job
- Prevent accidental overwriting of previous runs
- Save `run_summary.json` by default
- Optionally save annotated preview images

## Important Note: SAM3 Access

This tool does not include SAM3 model weights.

Before using SAM3, you must request access to the SAM3 model weights/checkpoints from the official SAM3 model page. After your access is approved, download the SAM3 weight file, such as `sam3.pt`, and pass its local path to this tool with `--model`.

SAM3 weights are not automatically downloaded by this project. The model file should also not be committed to this repository.

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

This project focuses on the auto-annotation workflow around SAM3, including:

- collecting input images
- running SAM3 text-prompted segmentation
- extracting polygon masks
- extracting bounding boxes
- organizing outputs by project
- writing CSV files
- saving run summaries
- optionally saving annotated preview images

## Supported Inputs

Use `--input` with either:

- A single supported image file: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, or `.webp`
- A folder containing supported image files

Folder input processes only image files directly inside that folder.

Note: folder input is currently non-recursive. Images inside subfolders are not processed.

## Installation

Install the required Python packages with:

```powershell
pip install -r requirements.txt
```

SAM3 support depends on the Ultralytics version installed in your environment. Make sure your Ultralytics installation supports SAM3.

## SAM3 Model Path

This tool requires a local SAM3 model weight file, such as `sam3.pt`.

SAM3 weights are not included in this repository. You must request access to the SAM3 model weights first, download the model file after approval, and then provide the local model path with `--model`.

```powershell
python sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text "cat"
```

The script forwards this model path to Ultralytics `SAM3SemanticPredictor`.

## Text Prompts

Pass one or more text prompt class names with `--text`. Multi-word prompts should be quoted.

```powershell
python sam3_auto_annotator.py --input "path\to\image.jpg" --model "path\to\sam3.pt" --text "siamese cat" dog car
```

Each prompt is treated as a class name. Output rows include `class_id` and `class_name`.

## Output Folders

Outputs are written under `outputs/` by default. Each run creates a project folder.

- Use `--project-name` to choose the folder name.
- Without `--project-name`, the script derives a name from the input and prompts.
- If the folder already exists, a timestamp suffix is added unless `--overwrite` is used.
- Use `--timestamp` to always append a timestamp.
- Use `--output-root` to write projects somewhere other than `outputs/`.

## Output Structure

Example output structure:

```text
outputs/
└── sample_run/
    ├── sam3_auto_annotation_xyn_outputs.csv
    ├── sam3_auto_annotation_box_outputs.csv
    ├── run_summary.json
    └── annotated_images/        # only when --save-annotated is used
```

## CSV Outputs

Each project folder contains two CSV files.

### Polygon CSV

```text
sam3_auto_annotation_xyn_outputs.csv
```

This file stores normalized polygon segmentation data extracted from SAM3 mask outputs.

It includes fields such as:

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

```text
sam3_auto_annotation_box_outputs.csv
```

This file stores bounding box data extracted from SAM3 box outputs.

It includes fields such as:

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
| `--save-annotated` | No | Save annotated preview images |
| `--run-summary` / `--no-run-summary` | No | Enable or disable `run_summary.json`. Default: enabled |
| `--show` | No | Display annotated images with matplotlib |

## Example Commands

### Folder Input

```powershell
python sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text "siamese cat" dog car --project-name sample_run
```

### Single Image Input

```powershell
python sam3_auto_annotator.py --input "path\to\image.jpg" --model "path\to\sam3.pt" --text "siamese cat"
```

### Always Append a Timestamp

```powershell
python sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text person --project-name people_run --timestamp
```

### Save Annotated Images

```powershell
python sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text person car --save-annotated
```

### Use a Custom Output Root

```powershell
python sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text car --output-root "annotation_outputs" --project-name car_run
```

## Notes and Limitations

- This repository does not include SAM3 model weights.
- Users must request access to SAM3 weights/checkpoints separately.
- Users must provide the local SAM3 model path with `--model`.
- Folder input is currently non-recursive.
- Auto annotations should be reviewed before being used as final training labels.
- Output quality depends on the SAM3 model, prompt quality, image quality, and confidence threshold.
- This tool currently exports CSV outputs. YOLO dataset export may be added in a future version.

## License and Attribution

This project is an auto-annotation utility built around Ultralytics SAM3 inference.

SAM3 is developed by Meta. This repository does not include SAM3 model weights. Users are responsible for requesting access to SAM3 weights/checkpoints and complying with the SAM license terms.

Ultralytics is used as the SAM3 inference pipeline through `SAM3SemanticPredictor`.

If you use SAM3 outputs in research or publication, acknowledge the use of SAM Materials according to the SAM license.

## Portfolio Summary

SAM3-AutoAnnotator demonstrates a practical computer vision annotation workflow using Python, Ultralytics SAM3, text-prompted segmentation, CSV export, and project-based output organization.

The project focuses on building a reusable tool around an existing foundation model rather than training a segmentation model from scratch.