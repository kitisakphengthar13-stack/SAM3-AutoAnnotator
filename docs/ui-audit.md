# UI audit

## Scope

This audit covers the PySide6 GUI launched by `main.py`: information hierarchy,
widget ownership, action state, resizing, overflow, error presentation, icons,
and keyboard-visible state. Runtime test results are recorded separately in
[Verification](verification.md).

## Resolved findings

### P0 — state and data integrity

- A failed image decode clears image-owned graphics and shows a recoverable
  inline error with **Retry**; a previous image cannot remain as stale content.
- Applying an unchanged box or class is a no-op and cannot invalidate a valid
  polygon or mark the project dirty.
- Controller, canvas, and annotation-table selection are synchronized, and
  project replacement clears project-specific preview, progress, filters,
  selection, and draw state.

### P1 — workflow and layout

- The Inspector follows the task vocabulary: **Setup**, **Review**, **Export**.
- The command bar holds global open/save actions. Context actions live with the
  content they affect: prediction in Setup, editing in Review, overlays at the
  canvas, and artifacts in Export.
- Each surface has at most one emphasized primary action.
- **Run Pending (N)** communicates its exact scope and is disabled at zero.
  Navigation, review, reset, and apply actions are likewise disabled when their
  result is already known to be a no-op.
- Setup and Export pin their action footers while their content scrolls. Review
  scrolls its editor independently so the annotation table retains useful
  height.
- Long paths, project/image names, and task/status messages elide inside their
  owning widget and expose the full value through a tooltip. They do not resize
  sibling splitter panels.

### P2 — visual language

- Icons are semantic and native-first: Qt theme icons, then Qt style fallbacks,
  with bundled SVGs only for draw-box and workflow concepts that Qt does not
  represent consistently.
- Text, icon, color, enabled state, hover/pressed/checked state, and keyboard
  focus work together; icon or color alone never carries essential meaning.
- Success stays in persistent status/results surfaces; blocking decisions and
  recoverable failures receive explicit actions.

## Acceptance matrix

The supported window sizes are:

- `960 x 620` — minimum usable layout;
- `1360 x 840` — reference working layout.

At both sizes, acceptance covers empty, configured, busy, completed, corrupt
image, long-content, and filtered-dataset states. Required invariants are:

- no horizontal overflow in inspector content;
- pinned Setup and Export footers remain visible;
- the Review table retains usable height;
- long content does not alter splitter proportions;
- the current primary action and recovery path remain visible;
- focus, disabled, selection, and checked states remain distinguishable.

Screenshots support this review but are not treated as the sole evidence of
behavior. Repeatable test commands and actual results remain in
[Verification](verification.md).
