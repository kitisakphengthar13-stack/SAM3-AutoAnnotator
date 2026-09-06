# Verification

## Current implementation

`main` is the canonical implementation. This document describes verification of the
current workstation source tree and does not depend on historical development
branches. The [visual walkthrough](v3-review.md) documents the running UI and its
reference screenshots.

## Recorded evidence

- [GitHub Actions run 34020065817](https://github.com/kitisakphengthar13-stack/SAM3-AutoAnnotator/actions/runs/34020065817)
  verified the audited workstation changes on Linux and Windows. The complete
  domain/offscreen GUI suite passed on both runners, the native Windows interaction
  suite passed, and both 100% and 150% Qt renders completed successfully.
- CI resolves the production dependency set, compiles `src`, `tests`, and `tools`,
  runs the full suite on Linux and Windows, then runs the interaction suite again
  with the native Windows Qt platform.
- Rendered UI evidence is uploaded as `workstation-ui-Linux` and
  `workstation-ui-Windows`. Artifacts expire after 14 days; selected reference
  captures remain under `docs/screenshots/`.

## What is covered

- Pointer drawing, moving, resizing, pan starting over an existing box, Esc draft
  cancellation, temporary Space pan, wheel bounds, and independent next-box class.
- Object edit history, selection restoration, clean saved-state markers, and
  history barriers for non-undoable project/inference operations.
- Save/reload/export with actual fixture files, CSV/YOLO output, segmentation
  omission rules, preflight opening without writes, stale-output replacement, and
  rollback protection for managed export artifacts.
- YOLO import validation and rollback, non-finite geometry rejection, source-image
  dimension mismatch detection, annotation-ID validation, and reset-to-SAM3
  provenance edge cases.
- Selected-box re-segmentation through the visual SAM box-prompt path, with spatial
  result matching so a higher-confidence remote instance is not selected merely
  because it shares the same semantic concept.
- Display/decode errors remaining presentation-only rather than rewriting review or
  annotation workflow state.
- Whole-button Open/assistance menus, confidence arrow hit targets, coordinate
  Apply/Cancel/keyboard behavior, Layers, dock close/float/restore, Focus Workspace
  persistence, and maximized/fullscreen restoration.
- Staged Setup, image review/navigation/filtering, background inference task
  lifecycle/cancellation with fake services, and repository architecture boundaries.

## Run the checks

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
machine's desktop resolution. It explicitly sizes reference windows, which may
extend beyond a small runner desktop while still rendering the complete widget.

## Verification limits

CI does not certify real SAM3 checkpoint execution or CUDA behavior because no
production checkpoint/GPU is supplied to the workflow. Before claiming a GPU pass,
exercise first prediction, predictor reuse, selected-box re-segmentation, pending
batch/cancel, and saved/exported output using the intended model and GPU.

Physical multi-monitor/DPI transitions, native window chrome, and pointer behavior
on the target workstation remain distinct from runner checks. The current minimum
client area is 960 × 620 logical pixels. Direct polygon-point editing remains
outside this app's feature set; use a corrected box plus Re-segment.
