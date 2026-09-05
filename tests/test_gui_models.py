import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush
from PySide6.QtTest import QAbstractItemModelTester, QSignalSpy
from PySide6.QtWidgets import QApplication

from domain import Annotation, AnnotationSource, ImageRecord, ImageStatus
from gui.models import (
    ANNOTATION_COUNT_ROLE,
    ANNOTATION_ID_ROLE,
    CLASS_ID_ROLE,
    CONFIDENCE_ROLE,
    IMAGE_INDEX_ROLE,
    IMAGE_NAME_ROLE,
    IMAGE_PATH_ROLE,
    IMAGE_STATUS_ROLE,
    SEGMENTATION_STATUS_ROLE,
    AnnotationTableModel,
    ImageFilterProxyModel,
    ImageListModel,
)


class QtModelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])


class ImageListModelTests(QtModelTestCase):
    def make_images(self):
        active = Annotation(0, "car", (1, 2, 20, 30), id="active")
        deleted = Annotation(0, "car", (2, 3, 21, 31), id="deleted", deleted=True)
        return [
            ImageRecord(
                "dataset/city/Car_001.jpg",
                10,
                width=1280,
                height=720,
                status=ImageStatus.EDITED,
                annotations=[active, deleted],
            ),
            ImageRecord(
                "dataset/forest/person.png",
                20,
                status=ImageStatus.REVIEWED,
            ),
            ImageRecord(
                "dataset/failed.jpg",
                30,
                status=ImageStatus.ERROR,
                error_message="Image file is unreadable.",
            ),
        ]

    def test_model_exposes_semantic_roles_and_presentation_data(self):
        model = ImageListModel(self.make_images())
        self._tester = QAbstractItemModelTester(
            model, QAbstractItemModelTester.FailureReportingMode.Warning
        )
        index = model.index(0, 0)

        self.assertEqual(model.rowCount(), 3)
        self.assertEqual(index.data(IMAGE_INDEX_ROLE), 10)
        self.assertEqual(index.data(IMAGE_PATH_ROLE), "dataset/city/Car_001.jpg")
        self.assertEqual(index.data(IMAGE_NAME_ROLE), "Car_001.jpg")
        self.assertEqual(index.data(IMAGE_STATUS_ROLE), "edited")
        self.assertEqual(index.data(ANNOTATION_COUNT_ROLE), 1)
        self.assertIn("Car_001.jpg", index.data(Qt.ItemDataRole.DisplayRole))
        self.assertIn("Annotations: 1", index.data(Qt.ItemDataRole.ToolTipRole))
        self.assertIn("Size: 1280 x 720", index.data(Qt.ItemDataRole.ToolTipRole))
        self.assertIsInstance(index.data(Qt.ItemDataRole.ForegroundRole), QBrush)
        self.assertIsInstance(index.data(Qt.ItemDataRole.BackgroundRole), QBrush)
        self.assertEqual(bytes(model.roleNames()[IMAGE_PATH_ROLE]), b"imagePath")

    def test_set_images_uses_reset_and_refresh_emits_data_changed(self):
        images = self.make_images()
        model = ImageListModel()
        reset_spy = QSignalSpy(model.modelReset)

        model.set_images(images)

        self.assertEqual(reset_spy.count(), 1)
        changed_spy = QSignalSpy(model.dataChanged)
        images[0].mark_reviewed()
        self.assertTrue(model.refresh(10))
        self.assertEqual(changed_spy.count(), 1)
        self.assertEqual(model.index(0, 0).data(IMAGE_STATUS_ROLE), "reviewed")
        self.assertFalse(model.refresh(999))
        self.assertEqual(model.index_for_image_index(20).row(), 1)
        self.assertFalse(model.index_for_image_index(999).isValid())

    def test_proxy_filters_case_insensitive_name_path_and_status(self):
        model = ImageListModel(self.make_images())
        proxy = ImageFilterProxyModel()
        proxy.setSourceModel(model)
        self._tester = QAbstractItemModelTester(
            proxy, QAbstractItemModelTester.FailureReportingMode.Warning
        )

        proxy.set_search_text("CITY")
        self.assertEqual(proxy.rowCount(), 1)
        self.assertEqual(proxy.image_at(0).image_index, 10)

        proxy.set_search_text("")
        proxy.set_status_filter("needs_review")
        self.assertEqual(proxy.rowCount(), 2)
        self.assertEqual(
            {proxy.image_at(row).status for row in range(proxy.rowCount())},
            {ImageStatus.EDITED, ImageStatus.ERROR},
        )

        proxy.set_status_filter(ImageStatus.ERROR)
        self.assertEqual(proxy.rowCount(), 1)
        self.assertEqual(proxy.image_at(0).error_message, "Image file is unreadable.")
        self.assertTrue(proxy.index_for_image_index(30).isValid())
        self.assertFalse(proxy.index_for_image_index(20).isValid())

        with self.assertRaises(ValueError):
            proxy.set_status_filter("unknown")


class AnnotationTableModelTests(QtModelTestCase):
    def make_annotations(self):
        return [
            Annotation(
                0,
                "car",
                (10, 20, 50, 80),
                id="ann-car",
                source=AnnotationSource.SAM3,
                confidence=0.91234,
                polygon_xyn=[[0.1, 0.2], [0.5, 0.2], [0.5, 0.8]],
            ),
            Annotation(
                1,
                "person",
                (60, 10, 90, 40),
                id="ann-person",
                source=AnnotationSource.MANUAL,
            ),
        ]

    def test_table_columns_roles_and_lookup_support_selection(self):
        annotations = self.make_annotations()
        model = AnnotationTableModel(annotations)
        self._tester = QAbstractItemModelTester(
            model, QAbstractItemModelTester.FailureReportingMode.Warning
        )

        self.assertEqual(model.rowCount(), 2)
        self.assertEqual(model.columnCount(), 4)
        self.assertEqual(
            [
                model.headerData(column, Qt.Orientation.Horizontal)
                for column in range(model.columnCount())
            ],
            ["Class", "Source", "Segmentation", "Confidence"],
        )
        self.assertEqual(model.index(0, 0).data(), "car")
        self.assertEqual(model.index(0, 1).data(), "sam3")
        self.assertEqual(model.index(0, 2).data(), "valid")
        self.assertEqual(model.index(0, 3).data(), "0.912")
        self.assertEqual(model.index(1, 3).data(), "-")
        self.assertEqual(model.index(0, 0).data(ANNOTATION_ID_ROLE), "ann-car")
        self.assertEqual(model.index(0, 0).data(CLASS_ID_ROLE), 0)
        self.assertEqual(model.index(0, 0).data(SEGMENTATION_STATUS_ROLE), "valid")
        self.assertAlmostEqual(model.index(0, 0).data(CONFIDENCE_ROLE), 0.91234)
        self.assertEqual(model.row_for_id("ann-person"), 1)
        self.assertEqual(model.index_for_id("ann-person", 2).row(), 1)
        self.assertIs(model.annotation_by_id("ann-car"), annotations[0])
        self.assertIs(model.annotation_at(model.index(1, 0)), annotations[1])
        self.assertFalse(model.index_for_id("missing").isValid())

    def test_set_items_resets_and_refresh_preserves_identity(self):
        annotations = self.make_annotations()
        model = AnnotationTableModel()
        reset_spy = QSignalSpy(model.modelReset)

        model.set_items(annotations)

        self.assertEqual(reset_spy.count(), 1)
        changed_spy = QSignalSpy(model.dataChanged)
        annotations[0].change_class(1, "vehicle")
        self.assertTrue(model.refresh("ann-car"))
        self.assertEqual(changed_spy.count(), 1)
        self.assertEqual(model.index_for_id("ann-car").data(), "vehicle")
        self.assertFalse(model.refresh("missing"))

    def test_models_reject_non_domain_items(self):
        with self.assertRaises(TypeError):
            ImageListModel([object()])
        with self.assertRaises(TypeError):
            AnnotationTableModel([object()])


if __name__ == "__main__":
    unittest.main()
