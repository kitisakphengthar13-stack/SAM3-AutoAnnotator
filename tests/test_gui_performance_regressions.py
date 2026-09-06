import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGraphicsView

from domain import Annotation, AnnotationSource, ImageRecord, ImageStatus
from gui.models import ANNOTATION_COUNT_ROLE, ImageListModel
from gui.views.dataset_panel import DatasetPanel
from gui.widgets.image_canvas import ImageCanvas


class GuiPerformanceRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.widgets = []

    def tearDown(self):
        for widget in self.widgets:
            widget.close()
            widget.deleteLater()
        QApplication.processEvents()
        self.temp_dir.cleanup()

    def create_image(self, name="image.png", size=(100, 80)):
        path = self.root / name
        image = QImage(size[0], size[1], QImage.Format_RGB32)
        image.fill(QColor("#2563eb"))
        self.assertTrue(image.save(str(path)))
        return path

    def test_canvas_uses_incremental_viewport_updates(self):
        canvas = ImageCanvas()
        self.widgets.append(canvas)
        self.assertEqual(
            canvas.viewportUpdateMode(),
            QGraphicsView.MinimalViewportUpdate,
        )

    def test_plain_canvas_click_does_not_emit_changes_for_every_annotation(self):
        canvas = ImageCanvas()
        self.widgets.append(canvas)
        canvas.resize(500, 400)
        canvas.show()
        canvas.load_image(self.create_image())
        annotation = Annotation(0, "car", (10, 10, 30, 30), id="ann-1")
        canvas.set_annotations([annotation])
        events = []
        canvas.annotation_changed.connect(lambda *args: events.append(args))

        point = canvas.mapFromScene(QPointF(75, 60))
        QTest.mouseClick(
            canvas.viewport(),
            Qt.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            point,
        )
        QApplication.processEvents()

        self.assertEqual(events, [])

    def test_canvas_reuses_unchanged_graphics_instead_of_rebuilding_scene(self):
        canvas = ImageCanvas()
        self.widgets.append(canvas)
        canvas.load_image(self.create_image())
        first = Annotation(
            0,
            "car",
            (10, 10, 30, 30),
            id="first",
            source=AnnotationSource.SAM3,
            polygon_xyn=[[0.1, 0.1], [0.3, 0.1], [0.3, 0.3]],
        )
        second = Annotation(
            0,
            "car",
            (50, 20, 80, 60),
            id="second",
            source=AnnotationSource.SAM3,
            polygon_xyn=[[0.5, 0.2], [0.8, 0.2], [0.8, 0.6]],
        )
        canvas.set_annotations([first, second])
        first_item = canvas._items_by_id[first.id]
        second_item = canvas._items_by_id[second.id]
        second_mask = canvas._mask_items_by_id[second.id]

        first.box_xyxy = (12.0, 12.0, 32.0, 32.0)
        canvas.set_annotations([first, second])

        self.assertIs(canvas._items_by_id[first.id], first_item)
        self.assertIs(canvas._items_by_id[second.id], second_item)
        self.assertIs(canvas._mask_items_by_id[second.id], second_mask)
        rect = canvas._items_by_id[first.id]._scene_rect()
        self.assertEqual(
            (rect.left(), rect.top(), rect.right(), rect.bottom()),
            first.box_xyxy,
        )

    def test_dataset_single_image_refresh_does_not_scan_all_images(self):
        panel = DatasetPanel(None)
        self.widgets.append(panel)
        images = [
            ImageRecord(f"dataset/{index}.jpg", index)
            for index in range(100)
        ]
        panel.set_images(images, "dataset")
        images[50].mark_reviewed()

        with patch.object(
            type(panel.image_model),
            "images",
            new_callable=PropertyMock,
            side_effect=AssertionError("full dataset scan"),
        ):
            self.assertTrue(panel.refresh(50))

        self.assertEqual(panel.stat_strip.reviewed.text(), "1 reviewed")
        self.assertEqual(panel.stat_strip.pending.text(), "99 pending")

    def test_image_model_refresh_updates_cached_annotation_count(self):
        image = ImageRecord("dataset/a.jpg", 7)
        model = ImageListModel([image])
        self.assertEqual(model.index(0, 0).data(ANNOTATION_COUNT_ROLE), 0)
        image.annotations.append(Annotation(0, "car", (1, 1, 10, 10)))

        self.assertTrue(model.refresh(7))

        self.assertEqual(model.index(0, 0).data(ANNOTATION_COUNT_ROLE), 1)
        self.assertEqual(model.row_for_image_index(7), 0)
        self.assertEqual(model.row_for_image_index(999), -1)


if __name__ == "__main__":
    unittest.main()
