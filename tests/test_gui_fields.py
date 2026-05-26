import os
import tempfile
import unittest
from pathlib import Path


try:
    from PySide6.QtCore import QPointF, QRectF
    from PySide6.QtGui import QColor, QImage
    from PySide6.QtWidgets import QApplication

    from sam3_auto_annotator.annotation.models import Annotation, ImageRecord, ImageStatus, ProjectState
    from sam3_auto_annotator.gui.fields import NumericLineEdit, configure_c_locale
    from sam3_auto_annotator.gui.icons import ICONS, icon
    from sam3_auto_annotator.gui.image_canvas import AnnotationRectItem, ImageCanvas
    from sam3_auto_annotator.gui.main_window import MainWindow
    from sam3_auto_annotator.gui.project_ops import parse_prompts
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
    ImageCanvas = None
    ImageRecord = None
    ImageStatus = None
    ProjectState = None
    MainWindow = None
    parse_prompts = None


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

    def test_classes_field_is_empty_by_default_with_placeholder(self):
        window = MainWindow()

        self.assertEqual(window.prompts_edit.toPlainText(), "")
        self.assertEqual(
            window.prompts_edit.placeholderText(),
            "One class per line, or comma-separated",
        )
        self.assertEqual(parse_prompts(window.prompts_edit.toPlainText()), [])
        self.assertEqual(window.class_combo.count(), 0)

    def test_empty_classes_block_sam3_with_clear_message(self):
        window = MainWindow()
        image = ImageRecord("traffic.jpg", 0, width=800, height=600)
        window.project_state = ProjectState("images", [], [image])
        window.current_image_index = 0
        messages = []
        window._show_error = lambda title, message: messages.append((title, message))

        window.run_sam3_current()

        self.assertEqual(messages, [("Missing classes", "Enter at least one class prompt before running SAM3.")])
        self.assertEqual(parse_prompts(window.prompts_edit.toPlainText()), [])

    def test_annotation_action_buttons_keep_expected_labels_and_tooltips(self):
        window = MainWindow()

        self.assertEqual(window.apply_box_button.text(), "Apply Box")
        self.assertEqual(window.delete_button.text(), "Delete")
        self.assertEqual(window.resegment_button.text(), "Re-segment from Box")
        self.assertEqual(window.reset_sam3_button.text(), "Reset to SAM3")
        self.assertEqual(window.apply_box_button.toolTip(), "Save the edited bounding box.")
        self.assertEqual(
            window.resegment_button.toolTip(),
            "Generate a new mask/polygon from the selected bounding box.",
        )
        self.assertEqual(window.reset_sam3_button.toolTip(), "Restore the original SAM3 annotation.")
        self.assertEqual(window.delete_button.toolTip(), "Delete the selected annotation.")

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

    def test_canvas_selection_changed_prunes_deleted_qgraphics_items(self):
        canvas = ImageCanvas()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.png"
            image = QImage(100, 80, QImage.Format_RGB32)
            image.fill(QColor("#ffffff"))
            self.assertTrue(image.save(str(image_path)))

            canvas.load_image(image_path)
            annotation = Annotation(0, "object", (10, 10, 40, 40), id="ann-1")
            canvas.set_annotations([annotation])
            self.assertIn("ann-1", canvas._items_by_id)

            canvas._scene.clear()
            canvas._on_selection_changed()

            self.assertEqual(canvas._items_by_id, {})
            self.assertIsNone(canvas.selected_annotation_id())

    def test_canvas_preserves_selection_by_annotation_id_after_redraw(self):
        canvas = ImageCanvas()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.png"
            image = QImage(100, 80, QImage.Format_RGB32)
            image.fill(QColor("#ffffff"))
            self.assertTrue(image.save(str(image_path)))

            canvas.load_image(image_path)
            annotation = Annotation(0, "object", (10, 10, 40, 40), id="ann-1")
            canvas.set_annotations([annotation])
            canvas.select_annotation(annotation.id)

            canvas.set_annotations([annotation])

            self.assertEqual(canvas.selected_annotation_id(), annotation.id)

    def test_main_window_selected_annotation_survives_canvas_refresh(self):
        window = MainWindow()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.png"
            image_file = QImage(100, 80, QImage.Format_RGB32)
            image_file.fill(QColor("#ffffff"))
            self.assertTrue(image_file.save(str(image_path)))

            annotation = Annotation(0, "object", (10, 10, 40, 40), id="ann-1")
            image = ImageRecord(str(image_path), 0, width=100, height=80, annotations=[annotation])
            window.project_state = ProjectState(str(temp_dir), ["object"], [image])
            window.current_image_index = 0
            window.canvas.load_image(image_path)
            window.canvas.set_annotations(image.active_annotations)
            window.canvas.select_annotation(annotation.id)

            window.canvas.set_annotations(image.active_annotations)

            self.assertEqual(window.selected_annotation(), annotation)

    def test_resegment_syncs_pending_detail_box_before_worker_start(self):
        window = MainWindow()
        image = ImageRecord("traffic.jpg", 0, width=800, height=600)
        annotation = Annotation(0, "traffic sign", (10, 20, 30, 40), id="ann-1")
        image.annotations.append(annotation)
        window.current_image = lambda: image

        window.x1_edit.set_value(351.0)
        window.y1_edit.set_value(182.75)
        window.x2_edit.set_value(378.82)
        window.y2_edit.set_value(214.08)

        before_box, after_box = window._sync_selected_box_details_for_resegment(annotation, image)

        self.assertEqual(before_box, (10.0, 20.0, 30.0, 40.0))
        self.assertEqual(after_box, (351.0, 182.75, 378.82, 214.08))
        self.assertEqual(annotation.box_xyxy, after_box)
        self.assertEqual(annotation.source.value, "edited")
        self.assertFalse(annotation.segmentation_valid)
        self.assertEqual(image.status, ImageStatus.EDITED)

    def test_resegment_syncs_subpixel_detail_box_edits(self):
        window = MainWindow()
        image = ImageRecord("traffic.jpg", 0, width=800, height=600)
        annotation = Annotation(0, "traffic sign", (351.0, 182.75, 378.82, 214.08), id="ann-1")
        image.annotations.append(annotation)
        window.current_image = lambda: image

        window.x1_edit.set_value(351.25)
        window.y1_edit.set_value(182.95)
        window.x2_edit.set_value(378.95)
        window.y2_edit.set_value(214.25)

        _, after_box = window._sync_selected_box_details_for_resegment(annotation, image)

        self.assertEqual(after_box, (351.25, 182.95, 378.95, 214.25))
        self.assertEqual(annotation.box_xyxy, after_box)

    def test_resegment_invalid_detail_box_leaves_annotation_unchanged(self):
        window = MainWindow()
        image = ImageRecord("traffic.jpg", 0, width=800, height=600)
        annotation = Annotation(0, "traffic sign", (351.0, 182.75, 378.82, 214.08), id="ann-1")
        image.annotations.append(annotation)
        window.current_image = lambda: image
        before = annotation.to_dict()

        window.x1_edit.set_value(378.82)
        window.y1_edit.set_value(182.75)
        window.x2_edit.set_value(351.0)
        window.y2_edit.set_value(214.08)

        with self.assertRaises(ValueError):
            window._sync_selected_box_details_for_resegment(annotation, image)

        self.assertEqual(annotation.to_dict(), before)

    def test_resegment_detail_box_is_clamped_to_image_bounds_before_worker_start(self):
        window = MainWindow()
        image = ImageRecord("traffic.jpg", 0, width=800, height=600)
        annotation = Annotation(0, "traffic sign", (351.0, 182.75, 378.82, 214.08), id="ann-1")
        image.annotations.append(annotation)
        window.current_image = lambda: image

        window.x1_edit.set_value(0.0)
        window.y1_edit.set_value(0.0)
        window.x2_edit.set_value(900.0)
        window.y2_edit.set_value(700.0)

        _, after_box = window._sync_selected_box_details_for_resegment(annotation, image)

        self.assertEqual(after_box, (0.0, 0.0, 800.0, 600.0))
        self.assertEqual(annotation.box_xyxy, after_box)

    def test_resegment_worker_result_is_ignored_if_bbox_changed_while_running(self):
        window = MainWindow()
        annotation = Annotation(
            0,
            "traffic sign",
            (351.0, 182.75, 378.82, 214.08),
            id="ann-1",
            polygon_xyn=[[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]],
            segmentation_valid=False,
        )
        image = ImageRecord("traffic.jpg", 0, width=800, height=600, annotations=[annotation])
        window.project_state = ProjectState("images", ["traffic sign"], [image])
        annotation.edit_box((360.0, 190.0, 390.0, 225.0), image.width, image.height)
        before = annotation.to_dict()

        window._box_prompt_finished(
            image.image_index,
            annotation.id,
            (351.0, 182.75, 378.82, 214.08),
            [[0.4, 0.4], [0.5, 0.4], [0.5, 0.5]],
            0.9,
        )

        self.assertEqual(annotation.to_dict(), before)
        self.assertFalse(window.unsaved)
        self.assertIn("bbox changed", window.statusBar().currentMessage())

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
