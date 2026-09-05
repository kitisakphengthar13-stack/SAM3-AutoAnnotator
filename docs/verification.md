# Verification

This file separates automated branch evidence from checks that still require the
target Windows/GPU workstation.

## Current redesign branch

Branch: `redesign/canvas-workspace-v2`

The redesign now includes:

- flat repository source tree under `src/` with no project-name wrapper package;
- one executable composition root at `src/main.py` and no root forwarding script;
- `domain`, `services`, `sam3`, `storage`, and `gui` directly under `src/`;
- integration images under `tests/fixtures/images/`;
- runtime models, projects, and logs under the OS user-data home rather than the repository;
- canvas-first `QMainWindow` with Dataset/Objects docks;
- explicit Select/Pan/Box tools and independent active drawing class;
- class/confidence labels, bounded zoom, 100%, Fit, Space-pan, Focus Workspace,
  and F11 fullscreen;
- Undo/Redo for completed object edits and immediate single-object Delete;
- transactional Setup and service-backed Export preflight;
- focused Project/Annotation/Inference/Export/Presentation controllers;
- no Inspector compatibility surface, `AppController`, controller forwarding API,
  or `gui/controller.py` shim.

## Automated branch evidence

GitHub Actions run `33964762882` executed against code head
`a1240366c758b24f9e9d7803061ad59a870d6cfc` on 2026-09-05 and completed
successfully.

Recorded checks:

- Ubuntu 24.04;
- CPython 3.12.14;
- PySide6 6.11.2;
- Pillow 12.3.0;
- Linux EGL runtime for Qt offscreen execution;
- production `requirements.txt` dependency resolution with `pip --dry-run` passed;
- `PYTHONPATH=src`;
- `python -m compileall -q src tests` passed;
- `python -m unittest discover -s tests -v` ran **147 tests**;
- result: **147 passed, 0 failures, 0 errors, 0 skipped**.

The suite includes guards for repository layout, removed namespace/controller shims,
absence of Workstation use-case forwarding methods, focused controller routing,
Review & Next behavior in All/Needs Review filters, staged Setup, single export
warning acknowledgement, app-home paths, and undo clean/external-dirty behavior.

## Required local Windows baseline

From the repository root:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q src tests
$env:PYTHONPATH = "src"
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
```

## Visible Windows acceptance

Launch:

```powershell
.\.venv\Scripts\python.exe src\main.py
```

Check at minimum:

1. Open `tests/fixtures/images`; Dataset is docked left, Objects right, canvas central.
2. Close/move/float docks and restore them from View; restart and verify saved state.
3. Toggle Focus Workspace without losing image or selection.
4. Maximize, press F11 twice, and verify return to maximized state.
5. Verify Fit changes image framing only, never native window state.
6. Verify Zoom Out, 100%, Zoom In, wheel zoom, Pan, and temporary Space-pan.
7. Switch Select/Pan/Box repeatedly; Esc must return Select.
8. Change the active next-box class while another object is selected; the selected
   object must not be reclassified.
9. Verify class/confidence labels do not block object selection.
10. Add, move, resize, reclassify, reset, and delete objects; Undo/Redo each.
11. Save after an edit, edit again, Undo to the saved point, and verify the window
    returns to clean state; then verify an external change remains dirty across Undo.
12. Start inference after local edits and verify stale undo history cannot cross the
    inference mutation boundary.
13. Verify Review & Next advances exactly once in All and Needs Review filters.
14. Verify Setup Cancel/X discards drafts and valid Apply commits once.
15. Remove an in-use class while changing another setting; the whole Apply must be
    rejected without partial mutation.
16. Press Ctrl+E and verify preflight alone writes nothing; export only after the
    explicit Export Now/Export Anyway action.
17. Check normal annotation flow at 960x620 and 1360x840.
18. Repeat maximized/fullscreen at Windows DPI 125% and 150%; inspect native hit
    targets, dock/title-bar behavior, text clipping, and shortcuts.
19. Run pending inference and confirm progress/cancel remain usable in normal and
    Focus Workspace modes.
20. Verify the default project/model/log locations resolve under the Windows user-data
    directory rather than creating `models/` or `outputs/` in the checkout.

## Real SAM3 GPU check

Real checkpoint inference remains hardware-dependent. Before claiming a GPU pass,
verify CUDA selection, first inference, predictor reuse, GUI responsiveness,
saved-state round trip, and export counts with the intended SAM3 checkpoint.
An offscreen CI pass is not a real GPU pass.
