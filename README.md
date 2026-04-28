# SAM3-AutoAnnotator

SAM3-AutoAnnotator is a command-line auto-annotation tool for running Ultralytics SAM3 text-prompted segmentation on either one image or a folder of images. It writes project-based outputs that include polygon segmentation CSV data, bounding-box CSV data, an optional run summary, and optional annotated preview images.

## Supported Inputs

Use `--input` with either:

- A single supported image file: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, or `.webp`
- A folder containing supported image files

Folder input processes only image files directly inside that folder.

## SAM3 Model Path

Pass the SAM3 model weights with `--model`:

```powershell
python sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text "cat"
```

The script forwards this path to `SAM3SemanticPredictor` from Ultralytics.

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

## CSV Outputs

Each project folder contains:

- `sam3_auto_annotation_xyn_outputs.csv`: normalized polygon points and YOLO segmentation lines
- `sam3_auto_annotation_box_outputs.csv`: bounding boxes, normalized box values, class counts, and confidence scores

Both CSV files include image-level object indexes, class names, per-image class counts, and total class counts.

## Run Summary

By default, the script writes `run_summary.json` into the project folder. It includes:

- project name and output folder
- input path and model path
- prompts and confidence threshold
- images processed
- images with detections and no detections
- total detections and class counts
- generated output file paths

Disable it with `--no-run-summary`.

## Example Commands

Folder input:

```powershell
python sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text "siamese cat" dog car --project-name sample_run
```

Single image input:

```powershell
python sam3_auto_annotator.py --input "path\to\image.jpg" --model "path\to\sam3.pt" --text "siamese cat"
```

Always append a timestamp:

```powershell
python sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text person --project-name people_run --timestamp
```

Save annotated images:

```powershell
python sam3_auto_annotator.py --input "path\to\images" --model "path\to\sam3.pt" --text person car --save-annotated
```

## Requirements

Install dependencies with:

```powershell
pip install -r requirements.txt
```

