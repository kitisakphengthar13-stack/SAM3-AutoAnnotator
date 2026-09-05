# Architecture and design

## Product direction

SAM3 AutoAnnotator is a desktop annotation workstation, not a form application and
not a GUI wrapper around a CLI. The high-frequency loop is:

```text
Open -> Configure -> Predict/Import -> Inspect/Edit -> Review & Next -> Save -> Export
```

Configuration and export are transient transactions. The canvas and object review
workflow define the persistent workstation.

## Top-level boundaries

```text
src/
|-- main.py                         # executable composition root
|-- app_paths.py                    # OS user-data locations
|-- logging_setup.py
|-- version.py
|-- domain/
|-- services/
|-- sam3/
|-- storage/
`-- gui/
    |-- actions.py
    |-- controllers/
    |-- coordinators/
    |-- main_window.py
    |-- settings.py
    |-- theme.py
    |-- undo.py
    |-- models/
    |-- rendering/
    |-- tasks/
    |-- views/
    `-- widgets/
```

`domain`, `services`, `sam3`, and `storage` remain UI-independent. The repository
has no project-name wrapper package and no forwarding entrypoint.

## Application controller composition

`src/main.py` is the executable composition root and constructs
`WorkstationController`. The workstation object owns shared state and composes:

```text
WorkstationController
|-- ProjectController
|-- AnnotationController
|-- InferenceController
|-- ExportController
`-- PresentationController
```

Responsibilities are explicit:

- `ProjectController`: open/create/load/save projects, staged project settings,
  model/output browsing, path memory, and YOLO import.
- `AnnotationController`: object selection, rendering, manual boxes, class/box edits,
  delete/reset, overlays, and Review & Next.
- `InferenceController`: inference settings, run-current/run-pending/re-segmentation,
  task start/cancel/finish, and application of prediction/batch results.
- `ExportController`: export, preview, output paths, and filesystem-facing export
  operations.
- `PresentationController`: image loading, dataset-filter presentation, dirty/status
  context, action enablement policy, and GUI error reporting.

Active QAction, view-signal, and task-signal routing targets the focused owner
directly. `WorkstationController` exposes shared state/composition concerns only;
use-case forwarding methods, both public and private, are absent and guarded by
automated architecture tests.

`WorkstationController` does not inherit a monolithic controller. The retired
`gui/controller.py`, `AppController`, Inspector API, and `ControllerSurfaceAdapter`
do not exist in the active tree.

## Main window and coordinators

`MainWindow` is a Qt composition shell. It owns window-only concerns such as docks,
dialog containers, fullscreen/focus behavior, file pickers, native messages, and
status-bar surfaces. Project mutation algorithms do not belong there.

Cross-widget transactions that genuinely need window surfaces live in
`gui/coordinators/`:

- `AnnotationHistoryCoordinator`: undo/redo capture, clean-index, and external-dirty
  tracking;
- `SetupDialogCoordinator`: staged Apply/Cancel setup transaction;
- `ExportDialogCoordinator`: export preflight/result presentation.

Review & Next is an annotation use case in `AnnotationController`, not a timer-based
window coordinator. Setup captures its draft snapshot synchronously when the user
opens the dialog. Export preflight is the single warning acknowledgement; the write
operation does not ask the same question again through title-string bypass logic.

Export-readiness rules live in `services/export_service.py`; the export dialog only
presents those results. Setup/history coordinators call the focused controller that
owns each responsibility rather than private methods on the workstation facade.

## Canvas-first window composition

```text
QMainWindow
|-- menu / compact command bar
|-- left QDockWidget: Dataset
|-- central widget: CanvasWorkspace
|-- right QDockWidget: Objects
|-- window-modal Setup QDialog
|-- window-modal Export QDialog
`-- status bar
```

There is no three-way Dataset/Canvas/Inspector splitter and no persistent
Setup/Review/Export tab stack.

### Dataset dock

Owns image search/filter/list presentation and navigation. It may be closed, moved,
floated, and restored from View.

### Objects dock

The object table is primary. Compact selected-object controls provide class,
coordinates, re-segmentation, reset, and delete. Class identity remains visible on
the canvas when this dock is hidden.

### Canvas workspace

Owns image rendering/editable boxes, class/confidence labels, an independent active
class for the next box, Select/Pan/Box tools, Space-pan, zoom/100%/Fit, overlays,
and inference progress.

The active drawing class is not the selected-object class editor. Manual box
creation reads the visible active class directly.

## Setup and export transactions

Setup stages model path, classes/prompts, confidence, FP16, and output location.
Typing does not mutate the project. Apply validates before one commit; Cancel or
window close discards the draft. Removing an in-use class rejects the transaction
without partial mutation.

Loading an existing project does not automatically open Setup.

`Ctrl+E` opens Export preflight rather than writing files. Preflight reports review
completion, failed/unpredicted images, and segmentation readiness. Disk writes only
occur after the explicit Export Now/Export Anyway action.

## Editing safety and undo

Routine completed object edits are captured through `QUndoStack` using
`ImageSnapshotCommand`. Add, move/resize, class change, exact-coordinate edit,
reset, and delete are reversible. Single-object Delete is immediate rather than
modal because Undo is the recovery path.

The stack clean index represents the last saved/exported annotation state. Undoing
back to that index clears the dirty marker. Mutations outside object-edit history
(settings, review state, inference, imports, image metadata) are tracked separately
as external dirty state, so Undo cannot falsely mark an externally changed project
clean.

Inference clears object-edit history before model-generated state replaces or
re-segments annotations, preventing stale snapshots from crossing that mutation
boundary.

## Commands and window semantics

- Fit changes image framing only.
- 100% restores a 1:1 canvas transform.
- Zoom changes canvas scale only.
- Focus Workspace hides/restores side docks.
- F11 changes actual main-window fullscreen state.
- Select (`Esc`), Pan (`P`), and Box (`B`) are exclusive editing modes.
- Undo/Redo use standard platform shortcuts.

Native window maximize/fullscreen is never represented by the image Fit command.

## Models and background work

Images use `QAbstractListModel` plus `QSortFilterProxyModel`; annotations use
`QAbstractTableModel`. Domain records remain project-owned.

SAM3 work remains off the GUI thread through worker `QObject` instances in
`QThread`. Cancellation is cooperative; blocking model work must not move to the
main thread.

## Runtime data

Source checkout and runtime data are separate. `src/app_paths.py` resolves a
writable OS user-data home:

- Windows: `%LOCALAPPDATA%\SAM3-AutoAnnotator`;
- macOS: `~/Library/Application Support/SAM3-AutoAnnotator`;
- Linux: `$XDG_DATA_HOME/sam3-autoannotator` or `~/.local/share/sam3-autoannotator`.

`SAM3_AUTOANNOTATOR_HOME` is an explicit override. Automatic model discovery uses
`<app-home>/models`; the default project destination uses `<app-home>/projects`.
Neither path depends on the repository location.

## Persistent state and safety invariants

`QSettings` stores window geometry and `QMainWindow.saveState()` data for docks and
toolbars. Project content remains in `annotation_state.json`.

Required invariants include:

- project saves use atomic replacement;
- pending prediction does not overwrite edited/reviewed images;
- unchanged edits are no-ops;
- geometry/class changes invalidate stale segmentation;
- failed image loading clears stale canvas graphics;
- invalid/stale polygons are reported rather than exported as valid;
- unsaved work is protected on close;
- active inference is cancelled cooperatively rather than force-terminated.

## Repository invariants

Automated architecture tests reject restoration of the retired repository shape:

- no root `main.py` forwarding script;
- no `sam3_auto_annotator/` project-name wrapper at root or under `src/`;
- no root `images_test/`;
- no `src/gui/controller.py` compatibility shim;
- no imports from the removed namespace;
- no one-shot migration workflows left in the repository.

Integration images live under `tests/fixtures/images/`. Runtime model/project data
lives outside the repository by default.

## Verification philosophy

Tests assert user-visible behavior and architecture boundaries rather than the old
widget tree. CI uses `PYTHONPATH=src`, dependency-resolves production requirements,
compiles `src` and `tests`, and runs the complete offscreen suite on each branch
head.

Offscreen Qt tests are necessary but not sufficient. Visible Windows verification
is still required for native title-bar behavior, maximized/fullscreen restoration,
dock interaction, high-DPI scaling, pointer hit targets, toolbar overflow, and
keyboard shortcuts. Real SAM3/CUDA validation remains a separate hardware check.

The concrete interaction contract lives in [UI audit](ui-audit.md).
