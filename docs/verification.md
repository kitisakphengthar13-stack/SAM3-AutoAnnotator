# Verification

## Branch and implementation

`redesign/canvas-workspace-v3` descends from v2 commit
`9fc70dd72de75e35a5de150665775f99dac6e91e`. Main and v2 are unchanged.
The [visual walkthrough](v3-review.md) documents the UI and its actual screenshots.

## Recorded evidence

- [GitHub Actions run 34006331642](https://github.com/kitisakphengthar13-stack/SAM3-AutoAnnotator/actions/runs/34006331642)
  at `85777a7b68d03fae83ab9898cc24a33a24f2e4da`: Linux and Windows passed all
  169 tests; Windows passed 13 additional native Qt interaction tests. Both runners
  resolved production dependencies, compiled the project, and rendered the app at
  100% and 150% Qt scaling.
- The first Windows run exposed toolbar overflow at 960 × 620. The corrected bar
  was then verified in the successful run above. Native Windows screenshots also
  exposed a narrow confidence field; v3 widens it and tests its text rectangle.
- Local final implementation: **172 tests passed**, no failures/errors/skips, on
  PySide6 6.11.2 with Qt offscreen. The added cases cover coordinate keyboard Apply
  and invalid-draft recovery, pointer-driven Layers visibility, and destination
  visibility in a short export dialog.

The CI workflow runs the full suite on Linux and Windows and the 16 interaction
checks again using the Windows native Qt platform. It uploads real app captures
at both scales as `workstation-ui-Linux` and `workstation-ui-Windows` artifacts.
Artifacts expire after 14 days; selected reference captures are committed in
`docs/screenshots/`.

## What is covered

- Pointer drawing, moving, resizing, pan starting over an existing box, Esc draft
  cancellation, temporary Space pan, wheel bounds, and independent next-box class.
- Object edit history, selection restoration, clean saved-state markers, and
  history barriers for non-undoable project/inference operations.
- Save/reload/export with actual fixture files, CSV/YOLO output, segmentation
  omission rules, and preflight opening without writes.
- Whole-button Open/assistance menus, confidence arrow hit targets, coordinate
  Apply/Cancel/keyboard behavior, Layers, dock close/float/restore, Focus Workspace
  persistence, and maximized/fullscreen restoration.
- Staged Setup, image review/navigation/filtering, image decode/error recovery,
  background inference task lifecycle/cancellation with fake services, and domain
  and repository architecture boundaries inherited from v2.

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

## Target workstation checks

Real SAM3 inference is not available in this environment: no checkpoint or CUDA
GPU was supplied. Before claiming a GPU pass, run first prediction, predictor
reuse, corrected-box re-segmentation, pending batch/cancel, and saved/exported
output using the intended model and GPU. UI tests use fake inference services;
reference screenshots use a manual annotation.

Physical multi-monitor/DPI transitions, native window chrome, and pointer behavior
on the user's own hardware remain distinct from the Windows runner checks. The
minimum supported client area is 960 × 620 logical pixels. Direct polygon-point
editing remains outside this app's feature set; use a corrected box + Re-segment.
