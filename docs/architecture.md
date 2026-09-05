# Architecture and design

## Goals

The redesign treats SAM3 AutoAnnotator as a desktop annotation product, not a
pip package or a CLI wrapper. The structure therefore optimizes for a clear GUI
composition root, independently testable annotation rules, explicit file I/O,
and safe background inference.

The design favors existing Python, Qt, and Ultralytics mechanisms. Custom code
exists only where the product needs its own domain behavior: annotation state,
SAM3 result mapping, export policy, canvas interaction, and workflow
coordination.

## Top-level layout

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

`main.py` is deliberately small and conventional: it calls the application
composition root. It is not a second implementation of application behavior.

## Boundaries

### `core`

Pure project concepts and invariants:

- annotation geometry and normalized polygon rules;
- annotation source, original SAM3 snapshot, and segmentation validity;
- image status and project state;
- serialization-ready data models.

This layer contains no Qt widgets, dialogs, predictor construction, or file
picker behavior. Its rules can be exercised without starting a GUI.

### `services`

Application workflows that coordinate the core with external capabilities:

- create/load/save a project;
- import existing labels;
- apply annotation edits and review transitions;
- run one prediction or one box re-segmentation;
- select pending batch targets;
- build corrected export results.

The synchronous prediction service is intentionally independent of Qt threads.
The GUI decides how to schedule it, while tests can call it with a fake
predictor cache.

### `sam3`

The boundary around Ultralytics:

- lazily imports `SAM3SemanticPredictor` only when inference starts;
- constructs the predictor from model, confidence, and precision settings;
- caches a predictor for identical settings;
- maps Ultralytics result objects into application annotations.

The project does not reimplement SAM3. Ultralytics remains the source of model
execution; custom mapping code only converts third-party results into the
product's editable representation.

### `storage`

Filesystem-specific behavior:

- image discovery and dimensions;
- atomic JSON project persistence;
- YOLO detection import;
- CSV, YOLO, summary, and skipped-segmentation output.

Filename-stem collisions are rejected before YOLO export because YOLO label
files are stem-based. Export rows are associated with resolved image paths, not
only list position.

### `gui`

Qt-specific presentation and interaction:

- `controller.py` coordinates commands, services, selection, and view state;
- `main_window.py` is the `QMainWindow` shell and owns dialogs/status surfaces;
- `actions.py` is the single command catalog;
- `views/` assembles Dataset, Canvas, Setup, Review, and Export regions;
- `widgets/` contains small reusable controls and the annotation canvas;
- `models/` adapts project collections to Qt Model/View;
- `tasks/` owns worker objects, `QThread` lifecycle, progress, and cancellation;
- `rendering/` draws a saved preview without reading mutable widget state;
- `settings.py` persists UI geometry separately from project JSON;
- `theme.py` defines a restrained visual system through semantic object names.

`application.py` composes the main window, controller, services, settings, and
diagnostics. Widgets expose signals and render state instead of performing
prediction, persistence, or export themselves.

## Dependency direction

```text
main.py
  -> application.py (composition)
       -> gui (interaction and presentation)
       -> services (use cases)
            -> core (rules/state)
            -> sam3 (model gateway)
            -> storage (file gateway)

sam3 -> core
storage -> core
core -> Python standard library only
```

The practical rule is that `core` never imports Qt or Ultralytics, and SAM3/file
objects do not leak into widgets. Cross-boundary values are project records,
annotations, paths, or small service result objects.

## Qt mechanisms used

### Main window and layout

The application uses Qt's native desktop structure:

- `QMainWindow` owns the menu bar, toolbar, status bar, and central workspace.
- `QSplitter` gives the Dataset, Canvas, and Inspector user-resizable widths.
- standard layout managers (`QVBoxLayout`, `QHBoxLayout`, `QFormLayout`) own
  widget geometry; fixed pixel positioning is avoided.
- `QStackedWidget` switches among the empty state, recoverable image-load error,
  and active canvas, so a failed decode cannot leave a stale image visible.
- Setup and Export use internal scroll areas with pinned action footers. Review
  gives the selected-annotation editor its own bounded scroll area and reserves
  the remaining height for the annotation table.
- size-hint ownership is local: expandable content receives stretch, compact
  actions keep their natural size, and long single-line values use elision plus
  a tooltip instead of changing splitter geometry.

The minimum acceptance size is `960 x 620`; `1360 x 840` is the reference
working size. Both are checked with empty, ready, busy, error, and long-content
states rather than treating one screenshot as proof of layout behavior.

This follows
[QMainWindow](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QMainWindow.html)
and [QSplitter](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QSplitter.html)
rather than creating a custom window/layout framework.

### Commands

Each user command is one `QAction`. The same action is reused by menu, toolbar,
and panel buttons, keeping text, shortcut, icon, tooltip, checked state, and
enabled state consistent. See
[QAction](https://doc.qt.io/qtforpython-6/PySide6/QtGui/QAction.html) and
[QToolBar](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QToolBar.html).

The command bar contains only global project lifecycle actions. Context actions
are rendered by the panel that owns their data: prediction in Setup, annotation
editing in Review, overlays in the canvas bar, and artifact actions in Export.
Each surface has one primary action; secondary and destructive actions retain
their own state without competing for hierarchy.

Enabled state is derived from the current project, visible image, selection,
form edits, and task mode. Known no-ops are disabled: boundary navigation,
zero-count **Run Pending (N)**, already-reviewed images, unchanged box/class
forms, and reset when the SAM3 snapshot is already current. Domain operations
also preserve this no-op invariant so direct calls cannot dirty data or
invalidate segmentation accidentally.

### Collection widgets

Images use `QAbstractListModel` with `QSortFilterProxyModel`; annotations use
`QAbstractTableModel`. Views receive stable roles and model change signals
instead of being repopulated with ad-hoc item widgets. This follows Qt's
[Model/View Programming](https://doc.qt.io/qtforpython-6/overviews/qtwidgets-model-view-programming.html)
guidance for dynamic list and table data.

### Background work

Inference is a blocking model operation, so worker `QObject` instances run in a
`QThread`. Results, errors, progress, and completion cross the boundary through
signals. The task manager owns cleanup and allows one inference task at a time.
Batch cancellation is cooperative; it does not terminate a thread while a model
call is executing. This follows Qt's documented worker-object approach in
[QThread](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html).

### Icons and styling

Views request semantic icon names rather than filenames or provider-specific
identifiers. The resolver first uses Qt `QIcon.fromTheme()` with a native
`QStyle.StandardPixmap` fallback. Small bundled semantic SVGs are used only for
domain concepts without a clear cross-platform Qt icon: draw box, Setup,
Review, and Export. The text label remains visible for primary commands;
meaning never depends on color or icon alone.

The stylesheet targets semantic object names and states. Native widgets retain
their keyboard, focus, accessibility, and platform behavior. Spacing, grouping,
contrast, and command priority establish hierarchy without a custom component
framework.

### Persistent UI preferences

`QSettings` stores window geometry, main-window state, splitter state, and last
directory. Editable annotation data remains in `annotation_state.json`; UI
preferences cannot overwrite project content. See
[QSettings](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QSettings.html).

## UX rules

- The primary path is visible in order: Open, Configure, Predict/Import,
  Review/Edit, Save, Export.
- The Inspector names that workflow directly: Setup, Review, Export.
- Each surface exposes one primary action and keeps context actions with the
  content they affect.
- Destructive actions are visually distinct and require a selected target.
- Commands are disabled when prerequisites are absent, an operation would be a
  no-op, or a conflicting task is running.
- Progress appears next to the canvas where the result will be reviewed.
- Image status and unsaved state remain visible without opening a dialog.
- Errors state what failed and the next corrective action; technical detail is
  available in diagnostics.
- Automated annotations are drafts. Human review is explicit and export reads
  the editable project state.

## Data safety

- State save writes a temporary file, flushes it, then atomically replaces the
  destination.
- Pending-image prediction does not overwrite edited or reviewed images.
- Unchanged box/class submissions do not mutate source, dirty state, or
  segmentation validity.
- A failed image load clears all image-owned graphics before presenting a
  recoverable error state.
- Editing geometry/class invalidates the old segmentation.
- Invalid/stale segmentation is reported rather than exported.
- Duplicate image stems are rejected before they can overwrite YOLO files.
- The application asks before closing with unsaved work and does not force-kill
  active inference.

## Upstream references

- [Qt for Python documentation](https://doc.qt.io/qtforpython-6/)
- [Ultralytics SAM 3 guide](https://docs.ultralytics.com/models/sam-3)
- [Ultralytics SAM predictor reference](https://docs.ultralytics.com/reference/models/sam/predict/)

The concrete UI acceptance criteria and resolved audit findings are recorded in
[UI audit](ui-audit.md). Runtime test evidence belongs in
[Verification](verification.md).
