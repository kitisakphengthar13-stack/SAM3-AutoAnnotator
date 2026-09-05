# Verification

This file separates automated branch evidence from checks that still require the
target Windows/GPU workstation.

## Current redesign branch

Branch: `redesign/canvas-workspace-v2`

The redesign now includes:

- canvas-first `QMainWindow` with Dataset/Objects docks;
- explicit Select/Pan/Box tools and independent active drawing class;
- class/confidence labels, bounded zoom, 100%, Fit, Space-pan, Focus Workspace,
  and F11 fullscreen;
- Undo/Redo for completed object edits and immediate single-object Delete;
- transactional Setup and Export preflight;
- compact command surfaces;
- focused Project/Annotation/Inference/Export/Presentation controllers;
- no Inspector compatibility surface;
- no monolithic `AppController` implementation or inheritance.

`gui/controller.py` is only a compatibility import alias to `WorkstationController`.
The active application is constructed directly with `WorkstationController`.

## Automated branch evidence

GitHub Actions run `33960250564` executed against code head
`be2f52e2d91f3f20f551b1df5b33252ff23217ae` on 2026-09-05 and completed
successfully.

Recorded environment/checks:

- Ubuntu 24.04;
- CPython 3.12.14;
- PySide6 6.11.2;
- Pillow 12.3.0;
- Linux EGL runtime for Qt offscreen execution;
- production `requirements.txt` dependency resolution with `pip --dry-run` passed;
- `python -m compileall -q main.py sam3_auto_annotator tests` passed;
- `python -m unittest discover -s tests -v` ran **129 tests**;
- result: **129 passed, 0 failures, 0 errors**.

This run includes the controller decomposition after retiring the monolithic
`AppController` and `ControllerSurfaceAdapter`. CI uses branch-level concurrency so
obsolete runs are cancelled when a newer branch head is pushed.

The workflow uses `QT_QPA_PLATFORM=offscreen` and intentionally does not load a
real SAM3 checkpoint. Production dependencies are resolved, while the heavy
Torch/CUDA stack is not installed solely for GUI/domain tests.

## Automated acceptance covered

The suite covers, among other product and architecture invariants:

- central canvas and independent recoverable docks;
- Focus Workspace and fullscreen/Fit separation;
- exclusive Select/Pan/Box tools and Space-pan;
- active drawing class independence and on-canvas labels;
- zoom/100%/Fit behavior;
- Setup Apply/Cancel and all-or-nothing validation;
- Export preflight before disk writes;
- Undo/Redo snapshot semantics;
- image decode recovery and selection synchronization;
- prediction/re-segmentation/task boundaries;
- save/export/project integration;
- focused controller composition with no retired Inspector dependency;
- `AppController` compatibility name resolving to `WorkstationController` rather
  than a second implementation;
- `WorkstationController` remaining composition/delegation rather than importing
  project-service or inference implementation dependencies.

## Required local Windows baseline

From the repository root:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q main.py sam3_auto_annotator tests
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
```

A local run remains useful because the deployed workstation may use a different
Python/PySide6/GPU environment.

## Visible Windows acceptance

Run:

```powershell
.\.venv\Scripts\python.exe main.py
```

Check at minimum:

1. Open `images_test`; Dataset is docked left, Objects right, canvas central.
2. Close/move/float docks and restore them from View; restart and verify saved
   `QMainWindow` state.
3. Toggle Focus Workspace without losing image or selection.
4. Maximize, press F11 twice, and verify return to maximized state.
5. Verify Fit changes image framing only, never native window state.
6. Verify Zoom Out, 100%, Zoom In, wheel zoom, Pan, and temporary Space-pan.
7. Switch Select/Pan/Box repeatedly; Esc must return Select with unambiguous state.
8. Change the active next-box class while another object is selected; the selected
   object must not be reclassified.
9. Verify class/confidence labels do not block object selection.
10. Add, move, resize, reclassify, reset, and delete objects; Undo/Redo each.
11. Start inference after local edits and verify stale undo history cannot cross
    the inference mutation boundary.
12. Verify Review & Next advances exactly once when another visible image exists.
13. Verify Setup Cancel/X discards drafts and valid Apply commits once.
14. Remove an in-use class while changing another setting; the whole Apply must be
    rejected without partial mutation.
15. Press Ctrl+E and verify preflight alone writes nothing; export only after the
    explicit Export Now/Export Anyway action.
16. Check normal annotation flow at 960x620 and 1360x840 without depending on an
    overflow chevron.
17. Repeat maximized/fullscreen at Windows DPI 125% and 150%; inspect native hit
    targets, dock/title-bar behavior, text clipping, and shortcuts.
18. Run pending inference and confirm progress/cancel remain usable in normal and
    Focus Workspace modes.

## Regression requirements

The redesign must continue to preserve:

- corrupt image decode clears stale graphics and exposes recovery;
- unchanged class/box edits remain no-ops;
- canvas/table/controller selection remains synchronized;
- project replacement clears project-specific transient state;
- pending batch prediction does not overwrite edited/reviewed images;
- save/reopen preserves editable project data;
- export counts match project state;
- close with unsaved work and active inference remains safe.

## Real SAM3 GPU check

Real checkpoint inference remains hardware-dependent. Before claiming a GPU pass,
verify CUDA selection, first inference, predictor reuse, GUI responsiveness,
saved-state round trip, and export counts with the intended SAM3 checkpoint.
An offscreen CI pass is not a real GPU pass.
