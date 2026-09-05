from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from PySide6.QtGui import QUndoCommand

from domain import Annotation, ImageStatus


@dataclass(frozen=True)
class AnnotationStateSnapshot:
    status: str
    annotations: tuple[dict, ...]
    error_message: str | None

    @classmethod
    def capture(cls, image):
        return cls(
            status=image.status.value,
            annotations=tuple(deepcopy(item.to_dict()) for item in image.annotations),
            error_message=image.error_message,
        )

    def restore(self, image):
        image.status = ImageStatus(self.status)
        image.annotations = [
            Annotation.from_dict(deepcopy(item))
            for item in self.annotations
        ]
        image.error_message = self.error_message


class AnnotationSnapshotCommand(QUndoCommand):
    """Undo one completed annotation edit without restoring unrelated image metadata."""

    def __init__(
        self,
        image,
        before,
        after,
        on_applied,
        *,
        text,
        selected_annotation_id=None,
        before_selected_annotation_id=None,
        after_selected_annotation_id=None,
    ):
        super().__init__(text)
        self._image = image
        self._before = before
        self._after = after
        self._on_applied = on_applied
        self._before_selected_annotation_id = (
            selected_annotation_id
            if before_selected_annotation_id is None
            else before_selected_annotation_id
        )
        self._after_selected_annotation_id = (
            selected_annotation_id
            if after_selected_annotation_id is None
            else after_selected_annotation_id
        )
        self._first_redo = True

    def redo(self):
        # QUndoStack.push() invokes redo immediately. The controller has already
        # performed that first edit, so the first invocation must not duplicate it.
        if self._first_redo:
            self._first_redo = False
            return
        self._restore(self._after, self._after_selected_annotation_id)

    def undo(self):
        self._restore(self._before, self._before_selected_annotation_id)

    def _restore(self, snapshot, selected_annotation_id):
        snapshot.restore(self._image)
        self._on_applied(
            self._image.image_index,
            selected_annotation_id,
        )
