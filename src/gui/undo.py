from __future__ import annotations

from copy import deepcopy

from PySide6.QtGui import QUndoCommand

from domain import ImageRecord


class ImageSnapshotCommand(QUndoCommand):
    """Undo one completed edit by restoring a stable ImageRecord snapshot."""

    def __init__(
        self,
        image,
        before,
        after,
        on_applied,
        *,
        text,
        selected_annotation_id=None,
    ):
        super().__init__(text)
        self._image = image
        self._before = deepcopy(before)
        self._after = deepcopy(after)
        self._on_applied = on_applied
        self._selected_annotation_id = selected_annotation_id
        self._first_redo = True

    def redo(self):
        # QUndoStack.push() invokes redo immediately. The controller has already
        # performed that first edit, so the first invocation must not duplicate it.
        if self._first_redo:
            self._first_redo = False
            return
        self._restore(self._after)

    def undo(self):
        self._restore(self._before)

    def _restore(self, snapshot):
        restored = ImageRecord.from_dict(deepcopy(snapshot))
        self._image.image_path = restored.image_path
        self._image.image_name = restored.image_name
        self._image.image_index = restored.image_index
        self._image.width = restored.width
        self._image.height = restored.height
        self._image.status = restored.status
        self._image.annotations = restored.annotations
        self._image.error_message = restored.error_message
        self._on_applied(
            self._image.image_index,
            self._selected_annotation_id,
        )
