# User guide

## 1. Start the application

Launch the repository's desktop entry point directly:

```powershell
.\.venv\Scripts\python.exe main.py
```

If the virtual environment is activated, `python main.py` is equivalent. The
window restores its previous geometry and splitter sizes through Qt settings.
Project data is separate and is restored only by opening an
`annotation_state.json` file.

## 2. Open or resume work

Choose one of these starting actions:

- **Open Image** starts a one-image project.
- **Open Folder** loads supported images directly in one folder. It does not
  scan subfolders.
- **Open Project** resumes a saved `annotation_state.json`.

Supported image suffixes are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`,
`.tiff`, and `.webp`.

Every image in one folder must have a unique filename stem. For example,
`frame01.jpg` and `frame01.png` cannot coexist in one project because both would
map to `frame01.txt` in YOLO output.

## 3. Configure the project

The **Setup** tab owns project-level settings:

- **Output** is the directory for the state and exported artifacts.
- **Model** is a local SAM3 `.pt` checkpoint.
- **Classes** are SAM3 text prompts, entered one per line or comma-separated.
- **Confidence** accepts values from `0.01` through `1.00`.
- **fp16** requests 16-bit inference; disable it when the execution environment
  does not support that precision reliably.

Prompt order defines stable class IDs. If the prompts are `person`, `car`, and
`dog`, their class IDs are `0`, `1`, and `2` respectively. Prompt names must
be unique.

## 4. Create or import draft annotations

### SAM3 on one image

Select an image and choose **Run SAM3**. The task runs outside the main GUI
thread. Successful results replace the SAM3 drafts for that image and set its
status to `predicted`; an empty result sets `no_detection`.

### SAM3 on pending images

Choose **Run Pending (N)** to process only images with `not_predicted` or
`error` status. `N` is the number currently eligible; the action is disabled at
zero. It does not overwrite predicted, edited, reviewed, or no-detection images.
**Cancel Batch** appears only for this batch and stops after the current
inference call finishes.

### Existing YOLO detection labels

Choose **Import YOLO**, then select the directory containing `.txt` files in
this format:

```text
class_id x_center y_center width height
```

Coordinates must be normalized. Label files match image files by stem. Missing
files leave the image untouched; empty files mark it `no_detection`; malformed
rows are skipped and included in the import result. Unknown class IDs receive a
generated name such as `class_4`.

## 5. Review and edit

The workspace has three resizable regions:

- **Dataset**: search, status filter, image list, and review counts.
- **Canvas**: image, box/mask/polygon overlays, draw control, fit control, and
  task progress.
- **Inspector**: Setup, Review, and Export tabs.

On the canvas you can select a box, drag it, resize it with handles, or draw a
new box. The **Review** tab keeps image-level review separate from
selected-annotation controls and provides precise class and `x1, y1, x2, y2`
editing, re-segmentation, reset, and deletion.

Editing a SAM3 box or class marks the image `edited` and invalidates the old
polygon. This prevents a polygon derived from old geometry from being exported
as if it were still correct.

Apply actions are enabled only after their corresponding value changes.
Applying an unchanged class or unchanged coordinates is a no-op: it does not
mark the project dirty or invalidate a valid segmentation. **Reset to SAM3** is
available only when the selected annotation actually differs from its original
SAM3 snapshot, and **Mark Image Reviewed** disables once the image is reviewed.

Use these correction actions when appropriate:

- **Re-segment** asks SAM3 for a new polygon using the current box and
  class.
- **Reset to SAM3** restores the original SAM3 box, class, and available
  polygon snapshot.
- **Mark Image Reviewed** records that the current image has been checked.

Direct point-by-point polygon editing is not implemented.

## 6. Save and export

### Save Project

**Save Project** writes `annotation_state.json` with images, prompts, settings,
statuses, and editable annotations. Saving uses a temporary file and atomic
replacement so an interrupted write is less likely to corrupt the prior state.

Save after meaningful review work and after a partial or completed batch.

### Export Labels

**Export Labels** derives artifacts from the current editable state:

- `sam3_auto_annotation_box_outputs.csv` contains active boxes and normalized
  detection coordinates.
- `yolo_labels/detection/*.txt` contains current YOLO detection labels.
- `yolo_labels/segmentation/*.txt` contains only currently valid polygons.
- `segmentation_skipped_report.json` explains each omitted stale, missing, or
  invalid segmentation and appears only when needed.
- `run_summary.json` records counts, project settings, and generated paths.

**Save Preview** creates a reviewed image using the overlays currently enabled
in the canvas. Preview images are for inspection, not training labels.

## Layout and command placement

The top command bar contains only project-level open/save commands. Prediction
actions stay in the Setup footer, annotation actions stay in Review, canvas
actions stay beside the canvas, and final-output actions stay in Export. This
keeps one emphasized primary action per surface and prevents unrelated widgets
from stretching one another.

Setup and Export use an internal vertical scroll area with a pinned action
footer. Review scrolls only the selected-annotation editor while preserving
usable space for the annotation table. Long paths and status text are elided by
their owning widgets and remain available through tooltips.

The minimum supported window is **960 x 620**; **1360 x 840** is the reference
working size. Splitter handles remain user-adjustable at both sizes.

## Image statuses

| Status | Meaning |
|---|---|
| `not_predicted` | No imported or SAM3 result has been applied. |
| `predicted` | SAM3 produced one or more draft annotations. |
| `edited` | A person changed the current editable annotations. |
| `reviewed` | A person explicitly marked the image reviewed. |
| `no_detection` | Prediction/import completed but produced no active object. |
| `error` | The last automated operation failed for this image. |

## Keyboard shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+O` | Open image |
| `Ctrl+Shift+O` | Open folder |
| `Ctrl+Alt+O` | Open project |
| `Ctrl+I` | Import YOLO labels |
| `Ctrl+S` | Save project |
| `F5` | Run SAM3 on current image |
| `Shift+F5` | Run pending images |
| `B` | Toggle box drawing |
| `Delete` | Delete selected annotation |
| `Ctrl+Return` | Apply edited box coordinates |
| `Ctrl+R` | Re-segment selected box |
| `R` | Mark current image reviewed |
| `F` | Fit image to canvas |
| `Alt+Left` / `Alt+Right` | Previous / next visible image |
| `Ctrl+E` | Export labels |

Shortcuts are implemented as shared Qt actions, so menu entries, toolbar
buttons, inspector buttons, enabled state, tooltip, and shortcut refer to one
command definition.

## Recovery and diagnostics

- If an image cannot be decoded, the canvas is cleared and replaced by an
  inline error state. Use **Retry** after repairing the file, or select another
  image; the previous image is never left visible as if it were the failed one.
- If inference fails, keep the image at `error`, correct the
  model/prompt/runtime problem, and use **Run Pending (N)** to retry it.
- If a segmentation is skipped, inspect
  `segmentation_skipped_report.json`, correct the box, and use **Re-segment from
  Box**.
- If the application reports an unexpected error, use the diagnostic log path
  shown in the dialog. Saved project data is separate from that report.
- If the window layout becomes inconvenient, resize the splitter regions; Qt
  persists the resulting geometry for the next launch.
