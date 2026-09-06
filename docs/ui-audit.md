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
- At narrow widths Dataset may auto-collapse to preserve the canvas while Objects
  remains available. A Dataset panel deliberately reopened by the user is not
  immediately hidden again; an auto-hidden Dataset is restored when the workspace
  becomes wide enough.
- Native responsive decisions use the usable desktop as a ceiling rather than
  assuming Qt granted a requested window width larger than the physical screen.
- Setup is a window-modal configuration dialog, not a permanent inspector tab.
- Export is a window-modal preflight/result dialog, not a permanent inspector tab.
- **Focus Workspace** hides both side docks and restores their prior visibility.
- **Fullscreen** (`F11`) is a window command and is independent of image **Fit**.
- Qt `QMainWindow.saveState()` owns user dock layout; responsive auto-hide must not
  be persisted as though the user deliberately closed Dataset.

The retired `Dataset | Canvas | Inspector` splitter is not an acceptance
requirement and must not be recreated for compatibility with historical code.

## Canvas interaction and performance contract

The editor has explicit, mutually exclusive tools:

- **Select** (`Esc`) selects, moves, and resizes objects.
- **Pan** (`P`) pans without changing annotation data.
- **Box** (`B`) draws a new box using the visible active class.
- Space temporarily pans from Select or Box and restores the tool on release or
  loss of focus. Esc cancels an unfinished box.

The next-box class and selected-object class are independent selections. Choosing
the class for a future box must not silently reclassify the selected object.

Every visible box carries a click-through canvas label containing its class and,
when available, confidence. The user must not need the Objects dock merely to know
which class a box represents.

Ordinary annotation refreshes are incremental: unchanged graphics items and labels
are reused, and a normal mouse release must not broadcast change events for every
annotation. Dataset single-image refresh uses cached lookup/status accounting rather
than rescanning the complete project after every batch result.

Zoom Out, 100%, Zoom In, and Fit are separate commands. Fit never means maximize
or fullscreen. Scale changes are bounded so repeated wheel/button zoom cannot make
the scene effectively disappear.

## Editing safety contract

Routine annotation editing must optimize throughput without removing recovery:

- Add, move/resize, class change, coordinate Apply, Reset, and Delete are undoable.
- Undo/Redo use the standard platform shortcuts and are visible in Edit and the canvas tool rail.
- Single-object Delete is immediate; it does not interrupt every deletion with a
  confirmation dialog because Undo is the recovery path.
- Inference boundaries clear annotation undo history so an old snapshot cannot be
  replayed across a model-generated replacement or re-segmentation.
- Project-level destructive operations such as replacing all annotations with a
  new SAM3 prediction may still require confirmation.
- Unsaved mutations schedule a separate debounced atomic recovery snapshot. Recovery
  never replaces the manually saved project state and a recovered project remains
  dirty until Save Project is used.

## Review and navigation contract

- **Review & Next** marks the current image reviewed and advances when another
  visible image exists.
- Previous/Next remain available independently of review state.
- Dataset search/status filters may change which image is considered next.
- The Objects list is the primary content of the right dock; selected-object fields
  are compact contextual controls below it, not a large fixed-height form that
  displaces the object list. Exact coordinates use a separate Apply/Cancel dialog.

## Setup transaction and trust contract

Setup fields are drafts while the dialog is open:

- typing does not mutate `ProjectState`;
- **Apply** validates and commits the staged configuration once;
- invalid class removal keeps the dialog open and presents validation beside the
  offending configuration;
- **Cancel** or closing the dialog discards the draft and restores the values that
  were present when Setup opened;
- Run SAM3 belongs to the annotation workflow, not to a configuration-form footer;
- checkpoint selection visibly warns that `.pt` files must come from trusted sources.

This prevents half-typed model paths, prompt lists, and output destinations from
silently dirtying a project while making the checkpoint trust boundary explicit.

## Export transaction contract

`Ctrl+E` opens Export preflight; it does not write files immediately. Preflight
summarizes at least:

- reviewed images;
- images still needing review;
- unpredicted/failed images;
- annotations with stale, missing, or invalid segmentation.

The primary button inside the dialog performs the disk write. When warnings exist,
its wording is **Export Anyway** so accepting the warning is explicit. Managed
artifacts are staged before publication and restored from backup if publication
raises an exception. CSV text that spreadsheet software could interpret as a
formula is neutralized without mutating project/YOLO state. Export results and
output paths remain in the same transient dialog after completion.

## Command-bar contract

The global command bar contains Open, Save, Run SAM3, its assistance menu, Setup,
and Export. A flexible project label gives way before a command can be hidden.
Previous/Next and Review & Next sit below the image. Select/Pan/Box, Undo/Redo,
and zoom/100%/Fit occupy the canvas tool rail. All project commands must remain
usable at `960 x 620` with native Windows font metrics.

## State and data integrity

The workstation must not weaken domain rules:

- failed image decode clears image-owned graphics but does not rewrite reviewed or
  edited annotation workflow state;
- unchanged box/class submissions remain no-ops;
- edited geometry/class invalidates stale segmentation;
- segmentation export rejects non-finite/out-of-range, too-short, zero-length-edge,
  zero-area, and self-intersecting polygons;
- selected-box Re-segment uses the visual SAM box-prompt path and spatial result
  matching, not semantic exemplar matching by confidence alone;
- project replacement clears project-specific selection and transient results;
- pending batch prediction does not overwrite edited/reviewed work;
- YOLO import and managed export have rollback boundaries rather than leaving known
  partial state after a later failure;
- source dimensions and SHA-256 fingerprints are verified before load/save/export so
  same-size replacement pixels cannot silently inherit old annotations;
- `annotation_state.json` remains the manually saved editable source of truth;
  `annotation_state.recovery.json` is only a temporary unsaved recovery snapshot.

## Acceptance sizes and verification

The minimum supported window is `960 x 620`; `1360 x 840` is the reference working
size. Acceptance is outcome based rather than panel-width based:

- canvas remains usable at both sizes;
- Dataset and Objects can be independently hidden and restored;
- narrow responsive mode gives canvas priority without treating auto-hide as a user
  preference;
- Focus Workspace produces a canvas-dominant layout;
- Setup and Export do not permanently narrow the canvas;
- command bar remains usable without depending on overflow navigation;
- maximize, fullscreen, Fit, 100%, and zoom remain semantically distinct;
- tool checked state, selected object, disabled actions, and running tasks remain
  visible beyond color alone.

Evidence combines automated Qt workflows, native Windows pointer/keyboard checks,
actual application captures at 100% and 150% Qt scaling, integrity regression tests,
and tested dependency resolution. Font, spacing, contrast, clipping, and control
visibility must be inspected in captures. Physical multi-monitor behavior and real
SAM3/CUDA inference still require the target environment and the explicit runtime
verification path documented in [verification](verification.md). See also the
[visual walkthrough](v3-review.md).
