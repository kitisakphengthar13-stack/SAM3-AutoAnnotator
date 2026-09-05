# User guide

## 1. Start the application

Launch from the repository root:

```powershell
.\.venv\Scripts\python.exe main.py
```

The window restores Qt geometry/dock state. Project data is separate and returns
only when you open a saved `annotation_state.json`.

## 2. Open or resume work

- **Open Image** starts a one-image project.
- **Open Folder** loads supported images directly in one folder; it is
  non-recursive.
- **Open Project** resumes `annotation_state.json`.

Supported image suffixes: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`,
`.webp`.

Images in one folder need unique filename stems because YOLO files are generated
from those stems.

## 3. Understand the workstation

The application is canvas-first:

- **Dataset dock**: image search/filter/list and navigation.
- **Canvas**: image, objects, overlays, Select/Pan/Box tools, active class, zoom,
  Fit, and task progress.
- **Objects dock**: object list and compact selected-object controls.
- **Setup**: transient configuration dialog.
- **Export**: transient preflight/result dialog.

Dataset and Objects can be closed, moved, floated, and restored from **View**.
**Focus Workspace** hides them temporarily. **F11** is actual window fullscreen;
**Fit** only changes image framing.

## 4. Configure the project

Open **Setup**. Its fields are staged drafts:

- **Output**: project/export destination.
- **Model**: local SAM3 `.pt` checkpoint.
- **Classes**: text prompts, one per line or comma-separated.
- **Confidence**: `0.01` through `1.00`.
- **fp16**: request 16-bit inference when supported.

Prompt order defines stable class IDs. Prompt names must be unique.

**Apply** validates then commits the configuration once. **Cancel** or closing the
dialog discards the draft. Removing a class currently used by annotations is
rejected before any other staged setting is committed.

Loading a saved project does not force Setup open.

## 5. Create or import draft annotations

### Run SAM3 on the current image

Select an image and use **Run SAM3**. The task runs outside the GUI thread. If the
image already contains active annotations, replacing them may require confirmation.

### Run pending images

Use **Run Pending (N)** for images whose status is `not_predicted` or `error`.
Predicted, edited, reviewed, and no-detection images are skipped. Batch cancel is
cooperative and takes effect after the current inference call.

### Import YOLO detections

Use **Import YOLO** and choose a label folder containing rows in this format:

```text
class_id x_center y_center width height
```

Coordinates are normalized. Files match images by stem. Missing files leave images
untouched; empty files mark no detection; malformed rows are skipped. Unknown class
IDs receive generated names such as `class_4`.

Import is a project operation, not a Setup control.

## 6. Navigate and edit the canvas

Tools are explicit and mutually exclusive:

- **Select (`Esc`)**: select, move, and resize objects.
- **Pan (`P`)**: pan without editing objects.
- **Box (`B`)**: draw a new box using the visible active class.
- Hold **Space** for temporary pan.

The active class beside the canvas is the class for the **next new box**. It is
independent of the class editor for the selected existing object.

Canvas objects display class and, when available, confidence. Use Zoom Out,
**100%**, Zoom In, Fit, or the mouse wheel for scale changes.

The Objects dock provides precise class and `x1, y1, x2, y2` editing plus
Re-segment, Reset, and Delete.

Changing a SAM3 box or class marks it edited and invalidates segmentation derived
from old geometry/class. Unchanged Apply operations are no-ops.

### Undo and delete

Completed add, move/resize, class change, coordinate Apply, Reset, and Delete
operations are undoable. Single-object Delete does not show a confirmation dialog;
Undo is the recovery path.

Inference is a separate mutation boundary. Starting model-generated replacement or
re-segmentation clears stale object-edit undo history so old snapshots cannot be
replayed into new model state.

### Re-segment and reset

- **Re-segment** asks SAM3 for a new polygon from the current selected box.
- **Reset to SAM3** restores the original SAM3 geometry/class snapshot when one
  exists.

Direct point-by-point polygon editing is not implemented.

## 7. Review and move forward

Use **Review & Next (`R`)** to mark the current image reviewed and advance to the
next visible image when one exists. Previous/Next remain independent navigation
commands. Dataset filtering changes which images are considered visible.

## 8. Save

**Save Project** writes `annotation_state.json` containing prompts, settings,
statuses, image records, and editable annotations. The save uses temporary-file
replacement to protect the previous state from interrupted writes.

Save after meaningful review work and after partial/completed batch inference.

## 9. Export

Press **Ctrl+E** or choose **Export**. This opens preflight; it does **not** write
files merely by opening.

Preflight summarizes:

- reviewed images;
- images still needing review;
- unpredicted/failed images;
- stale or missing segmentation.

Use **Export Now** when ready. When warnings remain the explicit action becomes
**Export Anyway**.

Generated artifacts include:

- `sam3_auto_annotation_box_outputs.csv`;
- `yolo_labels/detection/*.txt`;
- `yolo_labels/segmentation/*.txt` for currently valid polygons only;
- `segmentation_skipped_report.json` when segmentation is omitted;
- `run_summary.json`.

**Save Preview** creates an inspection image using current overlays; it is not a
training label.

## Image statuses

| Status | Meaning |
|---|---|
| `not_predicted` | No imported or SAM3 result has been applied. |
| `predicted` | SAM3 produced one or more draft annotations. |
| `edited` | A person changed editable annotations. |
| `reviewed` | A person explicitly reviewed the image. |
| `no_detection` | Prediction/import completed with no active object. |
| `error` | The last automated operation failed for this image. |

## Keyboard shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+O` | Open image |
| `Ctrl+Shift+O` | Open folder |
| `Ctrl+Alt+O` | Open project |
| `Ctrl+I` | Import YOLO labels |
| `Ctrl+S` | Save project |
| `Ctrl+,` | Setup |
| `F5` | Run SAM3 on current image |
| `Shift+F5` | Run pending images |
| `Esc` | Select tool |
| `P` | Pan tool |
| `B` | Box tool |
| `Space` | Temporary pan |
| `Delete` | Delete selected annotation |
| `Ctrl+Z` / `Ctrl+Y` | Undo / Redo (platform standard actions) |
| `Ctrl+Return` | Apply exact box coordinates |
| `Ctrl+R` | Re-segment selected box |
| `R` | Review & Next |
| `F` | Fit image |
| `Ctrl+0` | 100% image scale |
| `Ctrl+-` / `Ctrl++` | Zoom out / in |
| `Alt+Left` / `Alt+Right` | Previous / next visible image |
| `Ctrl+Shift+F` | Focus Workspace |
| `F11` | Fullscreen |
| `Ctrl+E` | Open Export preflight |

Shared Qt actions keep menu/toolbar/button enabled state, tooltip, and shortcut tied
to one command definition.

## Recovery and diagnostics

- Image decode failure clears stale canvas graphics and shows an inline Retry state.
- Inference failure marks the image `error`; fix the model/prompt/runtime issue and
  retry directly or through Run Pending.
- For skipped segmentation, inspect `segmentation_skipped_report.json`, correct the
  object, and Re-segment when appropriate.
- Unexpected errors include the diagnostic log path in the dialog.
- If dock placement becomes inconvenient, restore panels from View and reposition
  them; Qt persists `QMainWindow` state for the next launch.

## Verification limits

Automated offscreen tests validate workflow and architecture behavior but do not
replace visible Windows checks for native maximize/fullscreen, DPI scaling, dock
hit targets, text clipping, or toolbar behavior. Real SAM3/CUDA inference likewise
requires the target GPU/checkpoint.
