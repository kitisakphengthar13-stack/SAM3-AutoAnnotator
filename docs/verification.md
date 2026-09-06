# Verification

## Current implementation

This document describes verification of the current source tree. It deliberately
does not claim that an unmerged branch is already present on `main`, and it avoids a
hard-coded workflow-run number that becomes stale after every verification commit.
The branch/commit being evaluated by GitHub Actions is the source of truth for a
particular run.

The [visual walkthrough](v3-review.md) documents the running UI and its reference
screenshots.

## Automated verification

`.github/workflows/ci.yml` runs on pushes, pull requests, and manual dispatch. For
Python 3.12 it currently verifies:

- `requirements.txt` resolves against `constraints-tested.txt`, the known-tested
  production dependency resolution;
- the installed Ultralytics distribution still exposes the SAM and
  `SAM3SemanticPredictor` import contract used by production code;
- `src`, `tests`, and `tools` compile;
- the complete domain/offscreen GUI suite passes on Linux and Windows;
- the interaction suite passes again with the native Windows Qt platform;
- actual workstation screens render at 100% and 150% Qt scaling on both runners;
- rendered evidence is uploaded as `workstation-ui-Linux` and
  `workstation-ui-Windows` for 14 days.

A green CI result is required for the exact branch HEAD being considered. Do not use
a green run from an earlier commit as evidence for a newer HEAD.

## Integrity and regression coverage

The suite covers, among other paths:

- pointer drawing, moving, resizing, Pan-over-object behavior, Esc draft
  cancellation, temporary Space pan, wheel bounds, and independent next-box class;
- incremental canvas item reuse and incremental Dataset refresh so ordinary edits do
  not rebuild every graphics item or rescan the full dataset;
- responsive Dataset auto-collapse/restore at narrow widths, including native
  desktop-size behavior rather than assuming requested window geometry was granted;
- object edit history, selection restoration, clean saved-state markers, and
  history barriers for non-undoable project/inference operations;
- atomic manual project saves and separate atomic crash-recovery snapshots, including
  newer-recovery detection and cleanup without replacing the manual state;
- Save/reload/export with actual fixture files, CSV/YOLO output, preflight opening
  without writes, stale-output replacement, and rollback protection for managed
  export artifacts;
- spreadsheet-formula neutralization for CSV text fields without mutating project or
  YOLO state;
- YOLO import validation and project-level rollback, non-finite geometry rejection,
  class-id bounds, and missing/malformed label behavior;
- source-image dimension and SHA-256 fingerprint validation, including replacement
  content with unchanged pixel dimensions;
- annotation-ID/class/prompt invariants, strict persisted booleans, persisted polygon
  validation, and reset-to-SAM3 provenance edge cases;
- segmentation rejection for out-of-range/non-finite points, fewer than three
  distinct points, zero-length edges, zero-area polygons, and self-intersection;
- selected-box Re-segmentation through the visual SAM box-prompt path, with spatial
  result matching so a higher-confidence remote instance is not selected merely
  because it shares the same semantic concept;
- display/decode errors remaining presentation-only rather than rewriting review or
  annotation workflow state;
- whole-button Open/assistance menus, confidence arrow hit targets, coordinate
  Apply/Cancel/keyboard behavior, Layers, dock close/float/restore, Focus Workspace
  persistence, and maximized/fullscreen restoration;
- staged Setup, checkpoint trust warning, image review/navigation/filtering,
  background inference task lifecycle/cancellation with fake services, and
  repository architecture boundaries.

## Run the repository checks

From a prepared Windows checkout:

```powershell
$env:PYTHONPATH = "src"
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m compileall -q src tests tools
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
$env:QT_QPA_PLATFORM = "windows"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_gui_v3_interactions.py -v
.\.venv\Scripts\python.exe tools/render_ui.py --output-dir ui-captures
Remove-Item Env:QT_QPA_PLATFORM
Remove-Item Env:PYTHONPATH
```

See [the capture instructions](v3-review.md#reproduce-the-captures) for 150% scaling.
The renderer records logical and physical sizes rather than assuming the CI
machine's desktop resolution.

## Verify a production workstation

Hosted CI intentionally does not claim a real SAM3/CUDA pass because it has neither
the user's trusted production checkpoint nor the intended GPU. First install the
known-tested dependency resolution:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints-tested.txt
```

Then verify imports and CUDA:

```powershell
.\.venv\Scripts\python.exe tools/verify_runtime.py --require-cuda
```

For a checkpoint you trust, exercise Ultralytics' real checkpoint load path:

```powershell
.\.venv\Scripts\python.exe tools/verify_runtime.py --require-cuda --checkpoint D:\path\to\trusted\sam3.pt
```

Finally run a manual acceptance sequence on the intended workstation: first
prediction, predictor reuse, selected-box Re-segment, pending batch/cancel, Save,
reload, and export. Only that sequence can support a claim about the actual
checkpoint/GPU combination.

## Verification limits

Physical multi-monitor/DPI transitions, native window chrome, and GPU behavior on a
specific workstation remain distinct from hosted-runner checks. The supported
minimum client area is 960 × 620 logical pixels. Direct polygon-point editing
remains outside this app's feature set; use a corrected box plus Re-segment.
