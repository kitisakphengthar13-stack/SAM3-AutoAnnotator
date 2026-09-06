# User guide

## 1. Start the application

Launch from the repository root:

```powershell
.\.venv\Scripts\python.exe src/main.py
```

For a reproducible environment matching CI's tested resolution, install with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints-tested.txt
```

The window restores Qt geometry/dock state. Project data is separate and returns
only when you open a saved `annotation_state.json`.

## 2. Open or resume work

Use the top **Open** menu or the File menu:

- **Open Image** starts a one-image project.
- **Open Folder** loads supported images directly in one folder; it is
  non-recursive.
- **Open Project** resumes `annotation_state.json`.

If a saved project has a newer `annotation_state.recovery.json` beside its manual
state, Open Project offers to restore the recovery snapshot. Restored changes remain
marked unsaved until **Save Project** is explicitly used. Declining recovery removes
the stale recovery file and opens the last manual state.

Supported image suffixes: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`,
`.webp`.

Images in one folder need unique filename stems because YOLO files are generated
from those stems.

When a project is created, the app stores source dimensions and a SHA-256
fingerprint for each source image. Loading an existing fingerprinted project rejects
a changed source even when the replacement has the same dimensions. Legacy project
states without fingerprints are upgraded in memory after their current source files
pass dimension validation.

## 3. Understand the workstation

The application is canvas-first:

- **Dataset dock**: numbered image cards, search/status filter, and review progress.
- **Canvas**: image, objects, overlays, Select/Pan/Box tools, active class, zoom,
  Fit, and task progress.
- **Objects dock**: object cards with class, confidence, source, and mask status;
  selected-object controls appear below the list.
- **Setup**: transient configuration dialog.
- **Export**: transient preflight/result dialog.

At narrow window widths the Dataset dock auto-collapses to preserve annotation
space while the Objects dock remains available. You can reopen Dataset from
**View**; a panel you deliberately reopen is not immediately hidden again. When a
Dataset panel was hidden only by the responsive layout, widening the window restores
it automatically. The responsive threshold adapts to the usable desktop width on
native displays rather than assuming that a requested window size always fits the
physical screen.

Dataset and Objects can also be closed, moved, floated, and restored from **View**.
**Focus Workspace** hides them temporarily. **F11** is actual window fullscreen;
**Fit** only changes image framing. **View → Reset Workspace Layout** restores the
default dock arrangement; **F1** opens the shortcut reference.

## 4. Configure the project

Open **Setup**. The **Annotation** tab contains the model, classes, confidence,
and precision. **Files and output** contains the destination. Fields are staged drafts:

- **Output**: project/export destination.
- **Model**: local SAM3 `.pt` checkpoint. Only load checkpoint files from sources
  you trust.
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

Open **…** beside Run SAM3 and choose **Run Pending (N)** for images whose status is `not_predicted` or `error`.
Predicted, edited, reviewed, and no-detection images are skipped. Batch cancel is
cooperative and takes effect after the current inference call.

### Import YOLO detections

Use **… → Import YOLO** (or **Ctrl+I**) and choose a label folder containing rows in this format:

```text
class_id x_center y_center width height
```

Coordinates are normalized. Files match images by stem. Missing files leave images
untouched; empty files mark no detection; malformed rows are skipped. Unknown class
IDs receive generated names such as `class_4`.

Import is transactional across the project: a later file-level failure does not
leave earlier images partially imported. Import is a project operation, not a Setup
control.

## 6. Navigate and edit the canvas

Tools are explicit and mutually exclusive:

- **Select (`Esc`)**: select, move, and resize objects.
- **Pan (`P`)**: pan without editing objects.
- **Box (`B`)**: draw a new box using the visible active class.
- Hold **Space** while the canvas has focus for temporary pan from Select or Box;
  releasing it restores the tool. **Esc** cancels an unfinished box.

The active class beside the canvas is the class for the **next new box**. It is
independent of the class editor for the selected existing object.

Canvas objects display class and, when available, confidence. Use Zoom Out,
**100%**, Zoom In, Fit, or the mouse wheel for scale changes. Wheel/button zoom
is bounded from 5% to 2000%; the live percentage is below the image. **Layers**
opens independent Boxes, Masks, and Polygons visibility controls.

The Objects dock provides a class selector and an Apply checkmark, plus
Re-segment, Reset, and Delete. **Edit coordinates…** opens `x1, y1, x2, y2`
in a separate dialog. **Apply Box** or **Ctrl+Return** commits a valid edit;
**Cancel**, Esc, or closing the dialog discards the coordinate draft. Invalid
coordinates keep the dialog open for correction.

Changing a SAM3 box or class marks it edited and invalidates segmentation derived
from old geometry/class. Unchanged Apply operations are no-ops.

### Undo and delete

Completed add, move/resize, class change, coordinate Apply, Reset, and Delete
operations are undoable. Undo/Redo also restore the appropriate object selection.
Single-object Delete does not show a confirmation dialog; Undo is the recovery path.

Undo history is intentionally limited to object edits. Non-undoable project/image
changes such as **Review & Next**, applied Setup changes, YOLO import, and inference
results/errors clear older object-edit history so an old snapshot cannot overwrite
newer state. Starting SAM3 or Re-segment clears history only when the task actually
starts; a validation/start failure does not discard valid Undo history.

The saved-state marker uses the undo stack clean index. If you save, make only
undoable object edits, and Undo exactly back to the saved point, the project returns
to clean state. Once a non-undoable mutation occurs, the project remains dirty until
Save Project establishes a new clean state.

### Re-segment and reset

- **Re-segment** asks SAM3 for a new polygon from the current selected box using the
  visual box-prompt path and spatial result matching.
- **Reset to SAM3** restores the original SAM3 geometry/class snapshot when one
  exists.

Direct point-by-point polygon editing is not implemented. Exportable polygons must
be finite and normalized, contain at least three distinct points, have non-zero
area, contain no zero-length edge, and not self-intersect.

## 7. Review and move forward

Use **Review & Next (`R`)** to mark the current image reviewed and advance to the
next visible image when one exists. Previous/Next remain independent navigation
commands. Dataset filtering changes which images are considered visible.

## 8. Save and crash recovery

**Save Project** writes the authoritative `annotation_state.json` containing prompts,
settings, statuses, source fingerprints, image records, and editable annotations.
The save uses temporary-file replacement to protect the previous state from
interrupted writes.

While the project has unsaved mutations, the app schedules a separate atomic
`annotation_state.recovery.json` snapshot after edits settle for five seconds.
Further edits restart the debounce interval, avoiding repeated full-state writes
while the user is actively editing. Undo/Redo, project mutations, and completed
inference/batch changes participate in the same recovery schedule.

A recovery snapshot never replaces the manual state. Successful Save Project clears
the recovery snapshot. Recovery write failures are logged and do not replace or
corrupt the manually saved state.

Save after meaningful review work and after partial/completed batch inference even
though crash recovery exists; recovery is a safety net, not a substitute for an
explicit save checkpoint.

## 9. Export

Press **Ctrl+E** or choose **Export**. This opens preflight; it does **not** write
files merely by opening.

Before Save/export the app verifies every source image against the stored dimensions
and SHA-256 fingerprint. If image contents changed, restore the original source or
open the changed images as a new project; old annotations are not silently exported
against replacement pixels.

Preflight summarizes:

- reviewed images;
- images still needing review;
- unpredicted/failed images;
- stale, missing, or invalid segmentation.

Use **Export Now** when ready. When warnings remain the explicit action becomes
**Export Anyway**.

Generated artifacts include:

- `sam3_auto_annotation_box_outputs.csv`;
- `yolo_labels/detection/*.txt`;
- `yolo_labels/segmentation/*.txt` for currently valid polygons only;
- `segmentation_skipped_report.json` when segmentation is omitted;
- `run_summary.json`.

CSV text fields that begin with spreadsheet formula markers such as `=`, `+`, `-`,
or `@` are prefixed with an apostrophe in the CSV output so opening exported data in
a spreadsheet does not interpret those cells as formulas. This export hardening does
not modify the annotation state or YOLO labels.

Managed export artifacts are staged before publication. If publication raises an
exception, the exporter restores the previous managed output instead of knowingly
leaving a mixed old/new label set.

After export, the same dialog shows the completion summary and **Open Output
Folder**. **Files and preview** shows the generated paths and an inspection image.
**Save Preview** creates an inspection image using current overlays; it is not a
training label. **Done** closes the result dialog.

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
| `F1` | Keyboard shortcut reference |

Shared Qt actions keep menu/toolbar/button enabled state, tooltip, and shortcut tied
to one command definition.

## Recovery and diagnostics

- Image decode/display failure is presentation-only: it clears stale canvas graphics
  and shows Retry without demoting reviewed/edited annotation state.
- Inference failure marks a pending/error image `error`; fix the model/prompt/runtime
  issue and retry directly or through Run Pending. Existing reviewed/edited content
  is not demoted solely because a later automated operation failed.
- For skipped segmentation, inspect `segmentation_skipped_report.json`, correct the
  object, and Re-segment when appropriate.
- Unexpected errors include the diagnostic log path in the dialog.
- If dock placement becomes inconvenient, use **View → Reset Workspace Layout**;
  Qt persists the layout for the next launch. Closing while Focus Workspace is
  active does not permanently hide both panels.

## Production runtime verification

Hosted CI does not have the user's trusted SAM3 checkpoint or CUDA workstation. On
the intended machine, verify the installed dependency stack and CUDA first:

```powershell
.\.venv\Scripts\python.exe tools/verify_runtime.py --require-cuda
```

Then, for a checkpoint you trust, exercise Ultralytics' real checkpoint-loading path:

```powershell
.\.venv\Scripts\python.exe tools/verify_runtime.py --require-cuda --checkpoint D:\path\to\sam3.pt
```

After that, perform an actual workstation acceptance pass: current-image prediction,
selected-box Re-segment, pending batch/cancel, Save Project, reload, and export.

## Verification limits

CI covers domain/GUI workflows on Linux/Windows, native Windows interaction checks,
tested dependency resolution, the Ultralytics SAM import contract, and rendered
screens at two Qt scales. These runs do not replace testing the actual
GPU/checkpoint or the physical monitor setup. See [verification](verification.md).
