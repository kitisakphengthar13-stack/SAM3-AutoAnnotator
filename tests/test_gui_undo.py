import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QUndoStack

from domain import Annotation, AnnotationSource, ImageRecord, ImageStatus
from gui.undo import ImageSnapshotCommand


class ImageSnapshotUndoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_push_does_not_repeat_completed_edit_then_undo_and_redo_restore_state(self):
        image = ImageRecord(
            "image.png",
            0,
            width=100,
            height=80,
            status=ImageStatus.PREDICTED,
            annotations=[
                Annotation(
                    0,
                    "car",
                    (10, 10, 40, 40),
                    id="ann-1",
                    source=AnnotationSource.SAM3,
                    polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
                )
            ],
        )
        before = image.to_dict()
        image.annotations[0].mark_deleted()
        image.mark_edited()
        after = image.to_dict()
        callbacks = []
        stack = QUndoStack()

        stack.push(
            ImageSnapshotCommand(
                image,
                before,
                after,
                lambda image_index, selected_id: callbacks.append(
                    (image_index, selected_id)
                ),
                text="Delete annotation",
                selected_annotation_id="ann-1",
            )
        )

        self.assertTrue(image.annotations[0].deleted)
        self.assertEqual(image.status, ImageStatus.EDITED)
        self.assertEqual(callbacks, [])

        stack.undo()
        self.assertFalse(image.annotations[0].deleted)
        self.assertEqual(image.status, ImageStatus.PREDICTED)
        self.assertEqual(callbacks[-1], (0, "ann-1"))

        stack.redo()
        self.assertTrue(image.annotations[0].deleted)
        self.assertEqual(image.status, ImageStatus.EDITED)
        self.assertEqual(callbacks[-1], (0, "ann-1"))

    def test_snapshot_restores_added_annotation_list(self):
        image = ImageRecord("image.png", 3, width=100, height=80)
        before = image.to_dict()
        image.add_manual_annotation(0, "person", (2, 3, 20, 30))
        after = image.to_dict()
        stack = QUndoStack()
        stack.push(
            ImageSnapshotCommand(
                image,
                before,
                after,
                lambda *_args: None,
                text="Add annotation",
            )
        )

        self.assertEqual(len(image.active_annotations), 1)
        stack.undo()
        self.assertEqual(image.active_annotations, [])
        self.assertEqual(image.status, ImageStatus.NOT_PREDICTED)
        stack.redo()
        self.assertEqual(len(image.active_annotations), 1)
        self.assertEqual(image.active_annotations[0].class_name, "person")


if __name__ == "__main__":
    unittest.main()
