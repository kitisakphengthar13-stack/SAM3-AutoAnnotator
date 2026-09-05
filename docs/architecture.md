# Architecture and design

## Product direction

SAM3 AutoAnnotator is a desktop annotation workstation, not a form application and
not a GUI wrapper around a CLI. Architecture is organized around a high-frequency
human review loop while preserving independent domain, inference, and storage
boundaries.

```text
Open -> Configure -> Predict/Import -> Inspect/Edit -> Review & Next -> Save -> Export
```

Configuration and export are secondary transactions. They do not define the
persistent geometry of the annotation workspace.

## Top-level boundaries

```text
main.py
sam3_auto_annotator/
|-- application.py
|-- app_paths.py
|-- logging_setup.py
|-- core/
|-- services/
|-- sam3/
|-- storage/
`-- gui/
    |-- actions.py
    |-- controller.py              # legacy host being decomposed
    |-- controllers/               # active use-case controllers
    |-- coordinators/              # cross-widget UI transactions
    |-- main_window.py
    |-- settings.py
    |-- theme.py
    |-- undo.py
    |-- models/
    |-- rendering/
    |-- resources/
    |-- tasks/
    |-- views/
    `-- widgets/
```

`core`, `services`, `sam3`, and `storage` remain UI-independent. GUI structure may
be replaced without changing annotation/export domain rules.

### Core

Pure project concepts and invariants: annotations, image/project state, geometry,
segmentation validity, serialization-ready records. No Qt or Ultralytics imports.

### Services

Application use cases: create/load/save projects, import labels, apply annotation
edits, run prediction/re-segmentation, select pending targets, and build exports.
Services do not own desktop widgets.

### SAM3 boundary

Lazy Ultralytics predictor construction, cache/reuse, precision configuration, and
mapping third-party results into editable annotations.

### Storage

Filesystem behavior: image discovery, atomic project persistence, YOLO import, CSV
and YOLO export, summaries, and skipped-segmentation reports.

## GUI responsibility rule

`MainWindow` is a composition shell. It creates the persistent Qt surfaces, owns
window-specific behavior such as fullscreen/dock visibility, and exposes dialogs
and file pickers. It must not grow project mutation algorithms.

Workflow state that belongs to GUI interaction but not to a widget is isolated in
`gui/coordinators/`:

- `AnnotationHistoryCoordinator` owns undo/redo capture and replay boundaries;
- `AnnotationInteractionCoordinator` owns Review & Next cross-view follow-up;
- `SetupDialogCoordinator` owns the staged Apply/Cancel configuration transaction;
- `ExportDialogCoordinator` owns export readiness/preflight presentation.

This split is behavioral, not cosmetic: private undo snapshots, setup transaction
snapshots, and export-readiness calculations are no longer `MainWindow` state.
Tests explicitly prevent those responsibilities from drifting back into the shell.

Application use cases are moving through a strangler facade in `gui/controllers/`.
`application.py` constructs `WorkstationController`, not the old `AppController`
directly. The facade currently delegates active runtime behavior to:

```text
ProjectController     project activation and YOLO import
AnnotationController  selection/edit/review/manual boxes
InferenceController   prediction/re-segmentation/batch result application
ExportController      export/preview/output workflows
```

The legacy `AppController` temporarily supplies common state, settings validation,
task start/finish orchestration, persistence, action-state calculation, and shared
error/context helpers. New code must not add new use cases to it. Migration is done
only when those remaining responsibilities have been moved or deliberately reduced
and the inherited class can be deleted.

`ControllerSurfaceAdapter` remains only for legacy tests/code paths that instantiate
`AppController` directly. The active Project/Annotation/Inference/Export controllers
contain no Inspector references. The adapter is not an accepted workstation API.

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

There is no architectural requirement for a three-way Dataset/Canvas/Inspector
splitter and no persistent Setup/Review/Export tab stack.

### Dataset dock

Owns search, status filtering, image counts, list selection, and previous/next
navigation. It can be closed, moved, floated, and restored through View.

### Objects dock

The object table is the primary content. Compact selected-object controls below it
provide class, exact coordinates, re-segmentation, reset, and delete. The dock can
be hidden without making class identity unknowable because objects are labeled on
the canvas.

### Canvas workspace

Owns image rendering/editable boxes, class/confidence labels, the independent active
class for the next new box, exclusive Select/Pan/Box tools, temporary Space-pan,
Zoom Out/100%/Zoom In/Fit, overlay visibility, inference progress, and focus mode.

The active drawing class is not the selected-object class editor. Manual annotation
creation reads the visible `active_class_combo` directly; no compatibility bridge
changes the selected-object class behind the user's back.

### Setup transaction

Setup stages model path, prompts/classes, confidence, FP16, and output location.
Typing does not mutate `ProjectState`. Apply emits one validated settings commit;
Cancel or window close restores the snapshot captured when the dialog opened.
Invalid class removal leaves the dialog open with validation feedback.

Loading an existing project does not automatically open Setup. That behavior came
from the retired Inspector tab model and is intentionally not preserved.

### Export transaction

`Ctrl+E` opens preflight rather than writing files. The dialog summarizes review
completion, failed/unpredicted images, and stale/missing segmentation. A separate
primary action inside the dialog performs the write and becomes **Export Anyway**
when warnings are present. The same transient surface presents output paths and
preview after completion.

## Editing safety and undo

Routine object edits use `QUndoStack`. The current migration layer stores completed
`ImageRecord` before/after snapshots in `ImageSnapshotCommand` so add, move/resize,
class change, exact-coordinate edits, reset, and delete are reversible.

`QUndoStack.push()` calls `redo()` immediately, so a snapshot command deliberately
skips its first redo: the controller has already completed the mutation before the
command is recorded. Subsequent undo/redo restores a fresh `ImageRecord` from the
serialized snapshot and republishes models/canvas state.

Inference clears the object-edit undo stack. Model-generated replacement or
re-segmentation is a different mutation boundary and must not be mixed with stale
pre-inference snapshots.

## Commands and window semantics

- Fit fits the image viewport.
- 100% restores a 1:1 transform.
- Zoom changes only canvas scale.
- Focus Workspace hides/restores side docks.
- Fullscreen changes actual main-window state (`F11`).
- Select (`Esc`), Pan (`P`), and Box (`B`) are exclusive editing modes.
- Undo/Redo use standard platform shortcuts.

Fit never substitutes for window maximize/fullscreen.

The global command bar is intentionally small. Dense canvas tools stay in the
canvas bar. Navigation and undo/redo use compact icon-only toolbar controls to
avoid creating an overflow-navigation dependency at the minimum window size.

## Models and background work

Images use `QAbstractListModel` plus `QSortFilterProxyModel`; annotations use
`QAbstractTableModel`. Domain records remain project-owned and Qt models publish
state to views.

Inference remains off the GUI thread through worker `QObject` instances in
`QThread`. Cancellation is cooperative. No redesign may move blocking SAM3 work
onto the main thread.

## Persistent UI and project state

`QSettings` stores window geometry and `QMainWindow.saveState()` data, including
toolbar/dock placement. Retired splitter-state persistence is not retained.

Project content remains exclusively in `annotation_state.json`; window layout must
never mutate annotation data.

## Data safety invariants

- project saves use atomic replacement;
- pending prediction does not overwrite edited/reviewed images;
- unchanged annotation edits are no-ops;
- changing geometry or class invalidates stale segmentation;
- failed image loading clears stale canvas graphics;
- invalid/stale polygons are reported instead of exported as valid;
- duplicate image stems are rejected before YOLO overwrite can occur;
- unsaved project work is protected on close;
- active inference is not force-terminated.

## Verification philosophy

Tests assert user-visible outcomes rather than obsolete widget trees. Current UI
acceptance coverage targets central canvas/docks, tool modes, drawing-class
independence, labels, zoom/pan, fullscreen semantics, undo snapshots, transactional
Setup, Export preflight, compact command surfaces, coordinator boundaries, and
active controller routing.

GitHub Actions runs compile and the offscreen suite for every pushed branch commit.
The test job installs the Linux EGL runtime required by PySide6. It intentionally
does not load a real SAM3 model; Ultralytics imports are lazy and predictor behavior
is tested through fakes. Production requirements are still dependency-resolved in
CI, while real checkpoint/CUDA verification remains a separate hardware step.

Offscreen Qt tests are necessary but not sufficient. Visible Windows verification
is required for native title bar behavior, maximize/fullscreen restoration, dock
interaction, high-DPI scaling, real pointer hit targets, toolbar overflow behavior,
and keyboard shortcuts.

The concrete user-visible contract lives in [UI audit](ui-audit.md).
