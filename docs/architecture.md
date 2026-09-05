# Architecture and design

## Product direction

SAM3 AutoAnnotator is a desktop annotation workstation, not a form application and
not a GUI wrapper around a CLI. Architecture is organized around a high-frequency
human review loop while preserving independent domain, inference, and storage
boundaries.

The repeated product path is:

```text
Open -> Configure -> Predict/Import -> Inspect/Edit -> Review & Next -> Save -> Export
```

Configuration and export are secondary workflows. They must not dictate the
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
    |-- controller.py
    |-- main_window.py
    |-- settings.py
    |-- theme.py
    |-- models/
    |-- rendering/
    |-- resources/
    |-- tasks/
    |-- views/
    `-- widgets/
```

`core`, `services`, `sam3`, and `storage` remain UI-independent. The GUI may be
redesigned without changing annotation rules or export semantics.

### Core

Pure project concepts and invariants: annotations, image/project state, geometry,
segmentation validity, serialization-ready records. No Qt or Ultralytics imports.

### Services

Application use cases: create/load/save projects, import labels, apply annotation
edits, run prediction/re-segmentation, select pending targets, and build export
results. Services do not own desktop widgets.

### SAM3 boundary

Lazy Ultralytics predictor construction, cache/reuse, precision configuration, and
mapping third-party results into editable application annotations.

### Storage

Filesystem behavior: image discovery, atomic project persistence, YOLO import, CSV
and YOLO export, summaries, and skipped-segmentation reports.

### GUI

Qt presentation and interaction. The current redesign uses Qt desktop mechanisms
rather than recreating a custom window framework.

## Canvas-first window composition

`QMainWindow` owns the persistent application shell.

```text
QMainWindow
|-- menu / command bar
|-- left QDockWidget: Dataset
|-- central widget: CanvasWorkspace
|-- right QDockWidget: Objects / selected annotation
|-- modeless Setup QDialog
|-- modeless Export QDialog
`-- status bar
```

The central widget is always the image work surface. There is no architectural
requirement for a three-way Dataset/Canvas/Inspector splitter and no persistent
Setup/Review/Export tab stack.

### Dataset dock

Owns image search, status filtering, image counts, list selection, and previous /
next navigation. It can be closed, moved, or floated. Closing it must not disable
canvas editing.

### Objects dock

Owns the current-image object table and precise selected-annotation editing. It can
be closed, moved, or floated. Selecting an object from the canvas may reveal this
dock, but the dock is not the source of canvas geometry.

### Canvas workspace

Owns:

- image rendering and editable bounding boxes;
- active class for the next manually drawn object;
- Draw mode;
- Zoom Out / 100% / Zoom In / Fit;
- Space-drag hand panning;
- box/mask/polygon visibility;
- inference progress and cancellation state;
- focus-workspace control.

The active drawing class is intentionally next to Draw. Manual object creation
must never read a class choice that is only visible in another hidden surface.

### Setup dialog

Owns model path, prompts/classes, confidence, fp16, output location, import, and
prediction commands during the transition. It is transient configuration, not a
permanent third column.

### Export dialog

Owns export results, generated paths, preview, and output-folder actions. Export
preflight can evolve here without consuming canvas width during annotation.

## Window and view commands

Qt semantics are kept distinct:

- **Fit** fits the image to the current canvas viewport.
- **100%** restores a 1:1 image/screen transform.
- **Zoom In/Out** change canvas scale.
- **Focus Workspace** hides side docks and later restores their prior visibility.
- **Fullscreen** changes the actual main-window fullscreen state (`F11`).

Window maximize/fullscreen must not be represented by the Fit action.

## Commands and action state

Each user command remains one `QAction`. Menu, command bar, canvas buttons, and
panel buttons reuse those actions where appropriate so shortcut, tooltip,
checked state, and enabled state stay synchronized.

`AppController` currently remains the coordination boundary for existing project
workflows. The redesign must not grow it further; subsequent slices should split
project, annotation, inference, and export coordination when doing so reduces
coupling. A large controller is not preserved merely because it already exists.

## Collection models

Images use `QAbstractListModel` plus `QSortFilterProxyModel`; annotations use
`QAbstractTableModel`. Domain records remain owned by the project and Qt models
publish their state to views.

## Background work

Inference remains off the GUI thread through worker `QObject` instances in
`QThread`. Cancellation is cooperative. No redesign may move blocking SAM3 work
onto the main thread.

## Persistent UI state

`QSettings` stores window geometry and `QMainWindow.saveState()` data. The latter
already includes toolbar and dock placement, so the retired splitter-state format
is no longer persisted. The UI settings version is bumped when this contract
changes.

Project content remains exclusively in `annotation_state.json`; window layout
must never mutate annotation data.

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

Tests should assert outcomes and interactions, not preserve obsolete widget trees.
The UI acceptance suite therefore checks canvas centrality, dock behavior, focus
workspace, setup/export transient surfaces, visible drawing class, zoom/fit/100%,
Space-pan, and distinct fullscreen semantics.

Offscreen Qt tests are necessary but not sufficient. Visible Windows verification
is required for native title bar behavior, maximize/fullscreen, dock interaction,
high-DPI scaling, mouse hit targets, and keyboard shortcuts.

## Upstream references

- Qt for Python / Qt Widgets documentation
- `QMainWindow`, `QDockWidget`, `QAction`, `QGraphicsView`, `QSettings`
- Ultralytics SAM 3 guide and predictor API

The concrete user-visible contract lives in [UI audit](ui-audit.md).
