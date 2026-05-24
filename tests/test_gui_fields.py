import os
import tempfile
import unittest
from pathlib import Path


try:
    from PySide6.QtCore import QPointF, QRectF
    from PySide6.QtGui import QColor, QImage
    from PySide6.QtWidgets import QApplication

    from sam3_auto_annotator.annotation.models import Annotation
    from sam3_auto_annotator.gui.fields import NumericLineEdit, configure_c_locale
    from sam3_auto_annotator.gui.icons import ICONS, icon
    from sam3_auto_annotator.gui.image_canvas import AnnotationRectItem
    from sam3_auto_annotator.gui.main_window import MainWindow
except ImportError:  # pragma: no cover - optional GUI dependency
    QApplication = None
    QColor = None
    QImage = None
    QPointF = None
    QRectF = None
    Annotation = None
    NumericLineEdit = None
    configure_c_locale = None
    ICONS = None
    icon = None
    AnnotationRectItem = None
    MainWindow = None


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class GuiFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        configure_c_locale()
        cls.app = QApplication.instance() or QApplication([])

    def test_numeric_line_edit_uses_c_locale_decimal_format(self):
        field = NumericLineEdit(value=0.5, decimals=2, minimum=0.0, maximum=1.0)

        self.assertEqual(field.text(), "0.50")
        self.assertEqual(field.value(), 0.5)

        field.set_value(1920)
        self.assertEqual(field.text(), "1920.00")

    def test_icon_helper_returns_qicon_without_crashing(self):
        self.assertIsNotNone(icon(ICONS["image"]))

    def test_preview_thumbnail_uses_qpixmap_and_handles_invalid_images(self):
        window = MainWindow()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "preview.png"
            image = QImage(24, 16, QImage.Format_RGB32)
            image.fill(QColor("#2563eb"))
            self.assertTrue(image.save(str(image_path)))

            self.assertTrue(window._set_preview_thumbnail(image_path))
            self.assertFalse(window.preview_thumb.pixmap().isNull())
            self.assertFalse(window._set_preview_thumbnail(Path(temp_dir) / "missing.png"))

    def test_annotation_rect_resize_clips_to_image_bounds(self):
        annotation = Annotation(0, "object", (10, 10, 50, 50))
        item = AnnotationRectItem(annotation, QRectF(0, 0, 100, 100))
        item._active_handle = "top_left"
        item._press_scene_rect = QRectF(QPointF(10, 10), QPointF(50, 50))

        item._resize_from_handle(QPointF(-20, -20))
        rect = item._scene_rect()

        self.assertEqual((rect.left(), rect.top(), rect.right(), rect.bottom()), (0.0, 0.0, 50.0, 50.0))

    def test_annotation_rect_resize_enforces_minimum_size(self):
        annotation = Annotation(0, "object", (10, 10, 50, 50))
        item = AnnotationRectItem(annotation, QRectF(0, 0, 100, 100))
        item._active_handle = "right"
        item._press_scene_rect = QRectF(QPointF(10, 10), QPointF(50, 50))

        item._resize_from_handle(QPointF(10.5, 30))
        rect = item._scene_rect()

        self.assertEqual(rect.width(), 2.0)


if __name__ == "__main__":
    unittest.main()
