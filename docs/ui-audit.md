# UI audit

## Product contract

SAM3 AutoAnnotator is an annotation workstation. The dominant repeated task is:

```text
inspect image -> select/create object -> correct annotation -> review -> next image
```

The canvas is therefore the primary work surface. Project configuration and export
are transient workflows and must not permanently consume canvas width.

## Layout contract

- The image canvas is the `QMainWindow` central widget.
- Dataset navigation is a left `QDockWidget` that can be hidden, moved, or floated.
- Object review/editing is a right `QDockWidget` that can be hidden, moved, or floated.
- Setup is a separate project-configuration dialog, not a permanent inspector tab.
- Export/result details are a separate dialog, not a permanent inspector tab.
- **Focus Workspace** hides both side docks and restores their prior visibility.
- **Fullscreen** is a real window command (`F11`) and is independent of image **Fit**.
- Window state persists through `QMainWindow.saveState()` so dock layout is restored by Qt.

The retired three-way `Dataset | Canvas | Inspector` splitter is not an acceptance
requirement and must not be recreated merely to preserve historical structure.

## Annotation throughput contract

- The class assigned to the next manually drawn box is visible beside **Draw Box**.
  Drawing must never depend on a class selector hidden in another surface.
- Canvas navigation exposes Zoom Out, 100%, Zoom In, and Fit as separate commands.
- Holding Space temporarily enables hand panning without changing annotation data.
- **Review & Next** marks the image reviewed and advances when another visible image
  is available.
- Side panels may be closed without making the canvas unusable.
- Overlay visibility remains local to the canvas.
- Prediction progress remains adjacent to the canvas where results are reviewed.

## State and data integrity

The redesign does not weaken domain rules:

- a failed image decode clears image-owned graphics before showing recovery UI;
- unchanged box/class submissions remain no-ops;
- edited geometry/class invalidates stale segmentation;
- project replacement clears project-specific selection and transient results;
- pending batch prediction does not overwrite edited/reviewed work;
- saved project state remains the editable source of truth for export.

## Interaction semantics

- Fit means fit the image to the current canvas viewport; it must not use a maximize
  window semantic.
- Fullscreen means fullscreen application workspace.
- Draw mode, selected object, disabled actions, and running tasks must have visible
  state beyond color alone.
- Configuration errors belong with Setup. Image-load failures belong on the canvas.
  Destructive project-level decisions may use modal confirmation.

## Acceptance sizes

The minimum supported window remains `960 x 620`; `1360 x 840` is the reference
working size. Acceptance is outcome based rather than panel-width based:

- the canvas remains usable at both sizes;
- Dataset and Objects can be independently hidden and restored;
- Focus Workspace produces a canvas-dominant layout;
- Setup and Export do not force the central canvas narrower;
- long paths/status text do not increase the main-window minimum width;
- maximize, fullscreen, Fit, 100%, and zoom commands remain semantically distinct.

Automated UI tests must verify these behaviors without asserting the retired
splitter hierarchy. Visible Windows testing is still required for native title-bar,
dock, DPI, and fullscreen behavior.
