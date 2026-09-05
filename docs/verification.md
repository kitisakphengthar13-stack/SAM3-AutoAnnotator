# Verification

This file distinguishes implementation evidence from tests that still need to be
executed on the target repository checkout.

## Current redesign branch

Branch: `redesign/canvas-workspace-v2`

The branch now changes the interaction model materially:

- canvas-first `QMainWindow` with Dataset/Objects docks;
- explicit Select / Pan / Box tools and visible active drawing class;
- class/confidence labels on canvas objects;
- bounded zoom, 100%, Fit, Space-pan, Focus Workspace, and F11 fullscreen;
- Undo/Redo for completed annotation edits and immediate single-object Delete;
- Objects-list-first right dock instead of a fixed-height editor form;
- transactional Setup with Apply/Cancel;
- Export preflight before disk writes;
- compact command bar rather than relying on overflow navigation.

The historical `112 tests passed` result from `main` is **not** a result for this
branch.

At the latest connector check, GitHub reported no status checks and no workflow
runs for the branch head. The GitHub connector can read/write repository content
but cannot execute the checkout. Therefore this document makes **no runtime pass
claim** for the redesign branch.

## Required local baseline

From the repository root on the redesign branch:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q main.py sam3_auto_annotator tests
```

Then run:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
```

Record exact Python/PySide6 versions, test count, failures/errors, and exit code. A
command listed here is not a passing result.

## Automated acceptance added by the redesign

`tests/test_gui_ui_audit.py` checks the workstation contract, including:

- canvas centrality and independent docks;
- View-menu recovery for closed docks;
- Focus Workspace hide/restore;
- Setup/Export as transient dialog surfaces;
- Export entry opening preflight rather than the write action;
- independent active drawing class;
- exclusive Select/Pan/Box modes and Esc -> Select;
- canvas class/confidence labels;
- Zoom In/Out, 100%, Fit, Space-pan, and fullscreen/Fit separation.

`tests/test_gui_undo.py` checks snapshot command semantics for completed delete/add
edits, including the required first-redo no-op when pushed to `QUndoStack`.

`tests/test_gui_setup_dialog.py` checks:

- Cancel restoring the values present when Setup opened;
- Apply emitting one settings commit;
- closing the dialog discarding draft values;
- validation failure preventing the commit signal entirely.

`tests/test_gui_controller.py` now requires prompt edits to remain drafts until
Apply and requires in-use class removal to be rejected on Apply.

These tests are source-level acceptance definitions until they are actually run.

## Visible Windows acceptance

Run:

```powershell
.\.venv\Scripts\python.exe main.py
```

Check at minimum:

1. Open `images_test`; Dataset is docked left, Objects right, canvas central.
2. Close/move/float each dock; restore closed docks from View; restart and verify
   saved `QMainWindow` state.
3. Toggle Focus Workspace without losing current image/selection.
4. Maximize, press F11 twice, and verify return to maximized state.
5. Verify Fit changes image framing only, never native window state.
6. Verify Zoom Out, 100%, Zoom In, wheel zoom, Pan mode, and temporary Space-pan.
7. Switch Select/Pan/Box repeatedly; Esc always returns Select and checked state is
   visually unambiguous.
8. Configure at least three classes. Change the active next-box class while a
   different object is selected; confirm the selected object is not reclassified.
9. Draw objects and verify canvas labels show the correct class/confidence without
   blocking pointer selection of the box.
10. Add, move, resize, reclassify, reset, and delete annotations; Undo/Redo each
    operation. Delete must not show a per-object confirmation dialog.
11. Start inference after local edits and verify stale pre-inference undo history
    cannot be replayed across model-generated state.
12. Mark Review & Next and verify exactly one advance when another visible image
    exists.
13. Open Setup, edit model/classes/confidence/output, then Cancel/X; verify project
    state is unchanged. Reopen and Apply valid changes; verify one project update.
14. Attempt to remove an in-use class while also changing another setting; Apply
    must reject the whole transaction rather than partially committing settings.
15. Press Ctrl+E. Confirm no files are written merely by opening preflight. Review
    warning counts, then explicitly choose Export Now/Export Anyway.
16. At 960x620 confirm the command bar does not depend on an overflow chevron for
    the normal annotation loop. Repeat at 1360x840.
17. Repeat maximized/fullscreen and at Windows DPI 125%/150%; inspect real pointer
    hit targets, dock/title-bar behavior, and text clipping.
18. Run a pending-image task and confirm progress/cancel remain usable in normal
    and Focus Workspace modes.

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
