import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from domain import Annotation
from gui.main_window import MainWindow


class WorkCanvasIncrementalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.window = MainWindow()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        QApplication.processEvents()
        self.temp_dir.cleanup()

    def create_image(self):
        path = self.root / "image.png"
        image = QImage(100, 80, QImage.Format_RGB32)
        image.fill(QColor("#2563eb"))
        self.assertTrue(image.save(str(path)))
        return path

    def test_annotation_label_is_reused_and_updated_in_place(self):
        canvas = self.window.canvas
        canvas.load_image(self.create_image())
        annotation = Annotation(
            0,
            "car",
            (10, 10, 30, 30),
            id="ann-1",
            confidence=0.50,
        )
        canvas.set_annotations([annotation])
        box = canvas._items_by_id[annotation.id]
        label = canvas._labels_by_id[annotation.id]

        annotation.box_xyxy = (12.0, 12.0, 32.0, 32.0)
        annotation.confidence = 0.75
        canvas.set_annotations([annotation])

        self.assertIs(canvas._items_by_id[annotation.id], box)
        self.assertIs(canvas._labels_by_id[annotation.id], label)
        self.assertIs(label.parentItem(), box)
        self.assertEqual(label.text(), "car  0.75")
        self.assertEqual(len(canvas._labels_by_id), 1)

    def test_deleted_annotation_drops_label_registry_entry(self):
        canvas = self.window.canvas
        canvas.load_image(self.create_image())
        annotation = Annotation(0, "car", (10, 10, 30, 30), id="ann-1")
        canvas.set_annotations([annotation])
        self.assertIn(annotation.id, canvas._labels_by_id)

        annotation.deleted = True
        canvas.set_annotations([annotation])

        self.assertNotIn(annotation.id, canvas._labels_by_id)
        self.assertNotIn(annotation.id, canvas._items_by_id)


if __name__ == "__main__":
    unittest.main()
