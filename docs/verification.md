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
- generated `outputs/` excluded from Git;
- canvas-first `QMainWindow` with Dataset/Objects docks;
- explicit Select/Pan/Box tools and independent active drawing class;
- class/confidence labels, bounded zoom, 100%, Fit, Space-pan, Focus Workspace,
  and F11 fullscreen;
- Undo/Redo for completed object edits and immediate single-object Delete;
- transactional Setup and service-backed Export preflight;
- focused Project/Annotation/Inference/Export/Presentation controllers;
- no Inspector compatibility surface, `AppController`, or `gui/controller.py` shim.

## Automated branch evidence

GitHub Actions run `33963467478` executed against code head
`6f286289e4cc2513056dadd3f0d5e20e3004ada6` on 2026-09-05 and completed
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
- `python -m unittest discover -s tests -v` ran **136 tests**;
- result: **136 passed, 0 failures, 0 errors, 0 skipped**.

The suite includes repository-layout guards that reject the removed wrapper package,
root forwarding entrypoint, retired controller shim, old fixture directory, removed
namespace imports, and one-shot migration workflows.

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
11. Start inference after local edits and verify stale undo history cannot cross the
    inference mutation boundary.
12. Verify Review & Next advances exactly once when another visible image exists.
13. Verify Setup Cancel/X discards drafts and valid Apply commits once.
14. Remove an in-use class while changing another setting; the whole Apply must be
    rejected without partial mutation.
15. Press Ctrl+E and verify preflight alone writes nothing; export only after the
    explicit Export Now/Export Anyway action.
16. Check normal annotation flow at 960x620 and 1360x840.
17. Repeat maximized/fullscreen at Windows DPI 125% and 150%; inspect native hit
    targets, dock/title-bar behavior, text clipping, and shortcuts.
18. Run pending inference and confirm progress/cancel remain usable in normal and
    Focus Workspace modes.

## Real SAM3 GPU check

Real checkpoint inference remains hardware-dependent. Before claiming a GPU pass,
verify CUDA selection, first inference, predictor reuse, GUI responsiveness,
saved-state round trip, and export counts with the intended SAM3 checkpoint.
An offscreen CI pass is not a real GPU pass.
