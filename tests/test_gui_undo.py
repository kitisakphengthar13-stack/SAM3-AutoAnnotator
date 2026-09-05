import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QImage, QUndoStack
from PySide6.QtWidgets import QApplication

from domain import Annotation, AnnotationSource, ImageRecord, ImageStatus
from gui.controllers import WorkstationController
from gui.main_window import MainWindow
from gui.undo import ImageSnapshotCommand
from services.project_service import create_project


class MemorySettings:
    def last_directory(self):
        return ""

    def set_last_directory(self, _path):
        pass

    def save_window(self, _window):
        pass


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

    def test_undo_to_clean_index_clears_dirty_but_external_change_survives(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.png"
            qimage = QImage(100, 80, QImage.Format_RGB32)
            qimage.fill(Qt.white)
            self.assertTrue(qimage.save(str(image_path)))

            window = MainWindow()
            controller = WorkstationController(window, MemorySettings())
            project = create_project(image_path, ["car"], half=False)
            controller.projects.load_project(project)
            QCoreApplication.processEvents()
            window.canvas_area.active_class_combo.setCurrentIndex(0)

            window.canvas.box_drawn.emit((10, 10, 40, 40))
            QCoreApplication.processEvents()
            QCoreApplication.processEvents()
            self.assertEqual(window.history.stack.count(), 1)
            self.assertTrue(controller.dirty)

            window.history.mark_clean()
            QCoreApplication.processEvents()
            self.assertFalse(controller.dirty)

            annotation = controller.current_image.active_annotations[0]
            original_box = annotation.box_xyxy
            window.canvas.annotation_changed.emit(annotation.id, (12, 12, 45, 45))
            QCoreApplication.processEvents()
            QCoreApplication.processEvents()
            self.assertTrue(controller.dirty)

            window.actions.undo.trigger()
            QCoreApplication.processEvents()
            QCoreApplication.processEvents()
            self.assertFalse(controller.dirty)
            self.assertEqual(controller.current_image.active_annotations[0].box_xyxy, original_box)

            controller.presentation.mark_dirty(refresh=False)
            self.assertTrue(controller.dirty)
            annotation = controller.current_image.active_annotations[0]
            window.canvas.annotation_changed.emit(annotation.id, (14, 14, 48, 48))
            QCoreApplication.processEvents()
            QCoreApplication.processEvents()
            window.actions.undo.trigger()
            QCoreApplication.processEvents()
            QCoreApplication.processEvents()
            self.assertTrue(controller.dirty)

            window.controller = None
            window.close()
            window.deleteLater()
            QCoreApplication.processEvents()


if __name__ == "__main__":
    unittest.main()
