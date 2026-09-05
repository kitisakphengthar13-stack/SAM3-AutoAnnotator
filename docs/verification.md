# Verification

This document separates historical main-branch evidence from checks required for
the canvas-workstation redesign.

## Current redesign branch

Branch: `redesign/canvas-workspace-v2`

The UI architecture changed materially: the three-way splitter/Inspector contract
was removed, Dataset and Objects became `QDockWidget` surfaces, Setup/Export became
transient dialogs, and canvas navigation gained explicit zoom/100%/fit, Space-pan,
focus-workspace, and fullscreen behavior.

Because this is a structural UI change, the previous `112 tests passed` statement
from `main` is **not** evidence that this branch passes. Do not copy that claim into
a handoff until the commands below have actually been executed against this branch.

The GitHub connector used for the first redesign slice can read/write repository
content but cannot execute the repository. The isolated execution environment also
cannot clone GitHub directly, so no runtime pass is claimed for this slice.

## Required local baseline

From the repository root on the redesign branch:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q main.py sam3_auto_annotator tests
```

Then run the offscreen suite:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
```

Record the exact Python/PySide6 versions, test count, and result. A command listed
in this file is not a passing result.

## Canvas-workstation acceptance

The redesigned `tests/test_gui_ui_audit.py` intentionally checks outcomes rather
than preserving the removed splitter hierarchy. It covers:

- canvas as the central work surface;
- independent Dataset and Objects docks;
- hide/restore Focus Workspace behavior;
- transient Setup and Export surfaces;
- visible active class for manual drawing;
- explicit Zoom In/Out, 100%, and Fit semantics;
- temporary Space-drag hand panning;
- separation of F11 Fullscreen from image Fit.

These automated tests are necessary but not sufficient.

## Visible Windows acceptance

Run:

```powershell
.\.venv\Scripts\python.exe main.py
```

Check at minimum:

1. Open `images_test` and confirm Dataset is docked left, Objects docked right, and
   the canvas occupies the central area.
2. Close, move, and float each dock. Restart and verify `QMainWindow` state restores
   the layout.
3. Toggle **Focus Workspace** and confirm both docks disappear without changing the
   current image/selection; toggle again and confirm prior visibility is restored.
4. Maximize the window, press `F11`, press `F11` again, and confirm the window
   returns to maximized rather than normal size.
5. Verify Fit changes only image framing and never window state.
6. Verify Zoom Out, 100%, Zoom In, mouse-wheel zoom, and Space-drag panning.
7. Configure at least three classes. Change the active class beside Draw Box, draw
   a box, and confirm the created annotation uses that visible class.
8. Select existing objects and confirm precise edits remain available in the
   Objects dock without changing the active drawing class unexpectedly.
9. Mark an image **Review & Next** and confirm it advances exactly once when another
   visible image exists.
10. Open Setup and Export and confirm closing either dialog does not shrink or
    restructure the central canvas.
11. Exercise 960×620, 1360×840, maximized, fullscreen, and Windows DPI scaling at
    125%/150%.
12. Run a pending-image task and confirm progress remains usable in both normal and
    Focus Workspace modes.

## Regression checks retained from the product

The redesign must still verify:

- corrupt image decode clears stale graphics and exposes recovery;
- unchanged box/class edits remain no-ops;
- canvas/table/controller selection remains synchronized;
- project replacement clears project-specific transient state;
- pending batch selection does not overwrite edited/reviewed images;
- save/reopen preserves editable project data;
- export detection/segmentation counts match project state;
- close with unsaved work and active inference remains safe.

## Real SAM3 GPU check

Real checkpoint inference remains hardware-dependent. Before claiming a GPU pass,
verify CUDA device selection, first inference, predictor reuse, GUI responsiveness,
saved-state round trip, and export counts with the intended local SAM3 checkpoint.
A fake-predictor or CPU-only test is not a real GPU pass.
