# UI audit

## Product contract

SAM3 AutoAnnotator is an annotation workstation. The dominant repeated task is:

```text
inspect image -> select/create object -> correct -> review -> next image
```

The canvas owns the workstation. Project configuration and export are transient
workflows and must not permanently consume canvas width.

## Layout contract

- The image canvas is the `QMainWindow` central work surface.
- Dataset navigation is a left `QDockWidget`; Objects is a right `QDockWidget`.
- Both docks can be closed, moved, floated, and restored from the View menu.
- Setup is a window-modal configuration dialog, not a permanent inspector tab.
- Export is a window-modal preflight/result dialog, not a permanent inspector tab.
- **Focus Workspace** hides both side docks and restores their prior visibility.
- **Fullscreen** (`F11`) is a window command and is independent of image **Fit**.
- Qt `QMainWindow.saveState()` owns persisted dock layout.

The retired `Dataset | Canvas | Inspector` splitter is not an acceptance
requirement and must not be recreated for compatibility with historical code.

## Canvas interaction contract

The editor has explicit, mutually exclusive tools:

- **Select** (`Esc`) selects, moves, and resizes objects.
- **Pan** (`P`) pans without changing annotation data.
- **Box** (`B`) draws a new box using the visible active class.
- Space temporarily pans while Select is active.

The next-box class and selected-object class are independent selections. Choosing
the class for a future box must not silently reclassify the selected object.

Every visible box carries a click-through canvas label containing its class and,
when available, confidence. The user must not need the Objects dock merely to know
which class a box represents.

Zoom Out, 100%, Zoom In, and Fit are separate commands. Fit never means maximize
or fullscreen. Scale changes are bounded so repeated wheel/button zoom cannot make
the scene effectively disappear.

## Editing safety contract

Routine annotation editing must optimize throughput without removing recovery:

- Add, move/resize, class change, coordinate Apply, Reset, and Delete are undoable.
- Undo/Redo use the standard platform shortcuts and are visible in Edit/top command UI.
- Single-object Delete is immediate; it does not interrupt every deletion with a
  confirmation dialog because Undo is the recovery path.
- Inference boundaries clear annotation undo history so an old snapshot cannot be
  replayed across a model-generated replacement or re-segmentation.
- Project-level destructive operations such as replacing all annotations with a
  new SAM3 prediction may still require confirmation.

## Review and navigation contract

- **Review & Next** marks the current image reviewed and advances when another
  visible image exists.
- Previous/Next remain available independently of review state.
- Dataset search/status filters may change which image is considered next.
- The Objects list is the primary content of the right dock; selected-object fields
  are compact contextual controls below it, not a large fixed-height form that
  displaces the object list.

## Setup transaction contract

Setup fields are drafts while the dialog is open:

- typing does not mutate `ProjectState`;
- **Apply** validates and commits the staged configuration once;
- invalid class removal keeps the dialog open and presents validation beside the
  offending configuration;
- **Cancel** or closing the dialog discards the draft and restores the values that
  were present when Setup opened;
- Run SAM3 belongs to the annotation workflow, not to a configuration-form footer.

This prevents half-typed model paths, prompt lists, and output destinations from
silently dirtying a project.

## Export transaction contract

`Ctrl+E` opens Export preflight; it does not write files immediately. Preflight
summarizes at least:

- reviewed images;
- images still needing review;
- unpredicted/failed images;
- annotations with stale or missing segmentation.

The primary button inside the dialog performs the disk write. When warnings exist,
its wording is **Export Anyway** so accepting the warning is explicit. Export
results and output paths remain in the same transient dialog after completion.

## Command-bar contract

The global command bar stays intentionally small. Dense editing tools live beside
the canvas. Previous/Next and Undo/Redo use compact icon-only toolbar buttons so
the minimum `960 x 620` window does not require a toolbar overflow chevron merely
because every command was given full text.

## State and data integrity

The redesign does not weaken domain rules:

- failed image decode clears image-owned graphics before recovery UI;
- unchanged box/class submissions remain no-ops;
- edited geometry/class invalidates stale segmentation;
- project replacement clears project-specific selection and transient results;
- pending batch prediction does not overwrite edited/reviewed work;
- saved project state remains the editable source of truth for export.

## Acceptance sizes and verification

The minimum supported window is `960 x 620`; `1360 x 840` is the reference working
size. Acceptance is outcome based rather than panel-width based:

- canvas remains usable at both sizes;
- Dataset and Objects can be independently hidden and restored;
- Focus Workspace produces a canvas-dominant layout;
- Setup and Export do not permanently narrow the canvas;
- command bar remains usable without depending on overflow navigation;
- maximize, fullscreen, Fit, 100%, and zoom remain semantically distinct;
- tool checked state, selected object, disabled actions, and running tasks remain
  visible beyond color alone.

Automated Qt tests cover the application contract but do not replace visible
Windows verification. Native title-bar behavior, dock interaction, high-DPI
scaling, fullscreen/maximize restoration, real pointer hit targets, and real SAM3
inference still require execution on the target environment.
