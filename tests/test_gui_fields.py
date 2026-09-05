import os
import tempfile
import unittest
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QCoreApplication, QEvent, QObject, QPointF, QRectF, Signal
    from PySide6.QtGui import QColor, QImage
    from PySide6.QtWidgets import QApplication, QGraphicsItem

    from domain import (
        Annotation,
        AnnotationSource,
        ImageStatus,
    )
    from gui.controllers import WorkstationController
    from gui.controllers.state import UiMode
    from gui.icons import ICONS, icon
    from gui.main_window import MainWindow
    from gui.tasks.inference_task_manager import TaskKind
    from gui.widgets.image_canvas import (
        AnnotationRectItem,
        ImageCanvas,
    )
    from gui.widgets.numeric_field import (
        NumericLineEdit,
        configure_c_locale,
    )
    from services.prediction_service import (
        BoxSegmentation,
        ImagePrediction,
    )
    from services.project_service import parse_prompts
except ImportError:  # pragma: no cover - optional GUI dependency
    QApplication = None


if QApplication is not None:

    class FakeTaskManager(QObject):
        task_started = Signal(str)
        status = Signal(str)
        progress = Signal(int, int, str)
        prediction_ready = Signal(int, object)
        prediction_failed = Signal(int, str)
        segmentation_ready = Signal(int, str, object)
        segmentation_failed = Signal(int, str, str)
        batch_image_ready = Signal(int, object)
        batch_image_failed = Signal(int, str)
        batch_completed = Signal(dict)
        batch_cancelled = Signal(dict)
        task_failed = Signal(str)
        task_finished = Signal(str)

        def __init__(self):
            super().__init__()
            self.is_running = False
            self.kind = None
            self.prediction_calls = []
            self.segmentation_calls = []
            self.batch_calls = []
            self.cancel_requested = False

        def _start(self, kind):
            if self.is_running:
                raise RuntimeError("A fake task is already running.")
            self.is_running = True
            self.kind = kind
            self.task_started.emit(kind.value)

        def start_prediction(self, image_index, **settings):
            self._start(TaskKind.PREDICTION)
            self.prediction_calls.append((image_index, dict(settings)))

        def start_segmentation(self, image_index, annotation_id, **settings):
            self._start(TaskKind.SEGMENTATION)
            self.segmentation_calls.append(
                (image_index, annotation_id, dict(settings))
            )

        def start_batch(self, items, **settings):
            self._start(TaskKind.BATCH)
            self.batch_calls.append((list(items), dict(settings)))

        def request_cancel(self):
            if not self.is_running:
                return False
            self.cancel_requested = True
            return True

        def finish(self):
            kind = self.kind.value if self.kind is not None else "unknown"
            self.is_running = False
            self.kind = None
            self.task_finished.emit(kind)


    class FakeSettings:
        def __init__(self):
            self.directory = ""
            self.saved_window = None

        def last_directory(self):
            return self.directory

        def set_last_directory(self, path):
            self.directory = str(path)

        def save_window(self, window):
            self.saved_window = window


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class GuiFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_c_locale()
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.windows = []
        self.widgets = []
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        for window in self.windows:
            window.controller = None
            window.close()
            window.deleteLater()
        for widget in self.widgets:
            widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QCoreApplication.processEvents()
        self.temp_dir.cleanup()

    def make_controller(self):
        window = MainWindow()
        tasks = FakeTaskManager()
        settings = FakeSettings()
        errors = []
        infos = []
        confirmations = []
        window.show_error = lambda title, message, **kwargs: errors.append(
            (title, message, kwargs)
        )
        window.show_info = lambda title, message: infos.append((title, message))

        def confirm(title, message, **kwargs):
            confirmations.append((title, message, kwargs))
            return True

        window.confirm = confirm
        controller = WorkstationController(window, settings, task_manager=tasks)
        self.windows.append(window)
        return window, controller, tasks, errors, infos, confirmations

    def create_image(self, name="image.png", size=(100, 80)):
        path = self.root / name
        image = QImage(size[0], size[1], QImage.Format_RGB32)
        image.fill(QColor("#2563eb"))
        self.assertTrue(image.save(str(path)))
        return path

    def open_image_project(
        self,
        window,
        controller,
        *,
        prompts=("car",),
        image_path=None,
        model_path=None,
    ):
        image_path = image_path or self.create_image()
        model_path = model_path or self.root / "unused-model.pt"
        window.setup.prompts_edit.setPlainText("\n".join(prompts))
        window.setup.model_path_edit.setText(str(model_path))
        window.setup.output_dir_edit.setText(str(self.root / "output"))
        window.choose_image = lambda _start_directory="": str(image_path)
        controller.projects.open_image()
        QCoreApplication.processEvents()
        self.assertIsNotNone(controller.project)
        self.assertIsNotNone(controller.current_image)
        return controller.current_image

    @staticmethod
    def model_text(panel, annotation_id, column):
        index = panel.annotation_model.index_for_id(annotation_id, column)
        return index.data()

    def test_numeric_line_edit_uses_c_locale_decimal_format(self):
        field = NumericLineEdit(value=0.5, decimals=2, minimum=0.0, maximum=2000.0)
        self.widgets.append(field)

        self.assertEqual(field.text(), "0.50")
        self.assertEqual(field.value(), 0.5)
        field.set_value(1920)
        self.assertEqual(field.text(), "1920.00")
        field.setText("not-a-number")
        with self.assertRaisesRegex(ValueError, "valid decimal"):
            field.value()

    def test_icon_helper_returns_platform_icon_without_crashing(self):
        self.assertFalse(icon(ICONS["image"]).isNull())

    def test_canvas_keeps_fit_mode_when_the_window_resizes(self):
        canvas = ImageCanvas()
        self.widgets.append(canvas)
        canvas.resize(800, 600)
        canvas.show()
        QCoreApplication.processEvents()
        canvas.load_image(self.create_image(size=(640, 480)))

        canvas.resize(360, 260)
        QCoreApplication.processEvents()

        mapped = canvas.mapFromScene(canvas.sceneRect()).boundingRect()
        viewport = canvas.viewport().rect()
        self.assertLessEqual(mapped.width(), viewport.width() + 2)
        self.assertLessEqual(mapped.height(), viewport.height() + 2)

    def test_initial_shell_exposes_empty_setup_and_disables_project_actions(self):
        window, controller, _tasks, _errors, _infos, _confirmations = (
            self.make_controller()
        )

        self.assertIs(window.controller, controller)
        self.assertEqual(window.setup.prompts_text(), "")
        self.assertEqual(
            window.setup.prompts_edit.placeholderText(),
            "One class per line, or comma-separated",
        )
        self.assertEqual(parse_prompts(window.setup.prompts_text()), [])
        self.assertEqual(window.annotation.class_combo.count(), 0)
        self.assertTrue(window.actions.open_image.isEnabled())
        self.assertFalse(window.actions.run_current.isEnabled())
        self.assertFalse(window.actions.export.isEnabled())
        self.assertIs(
            window.canvas_area.workspace_stack.currentWidget(),
            window.canvas_area.empty_state,
        )

    def test_empty_classes_block_inference_with_recovery_guidance(self):
        window, controller, tasks, errors, _infos, _confirmations = (
            self.make_controller()
        )
        self.open_image_project(window, controller, prompts=())

        controller.inference.run_current()

        self.assertEqual(tasks.prediction_calls, [])
        self.assertEqual(controller.mode, UiMode.READY)
        self.assertEqual(len(errors), 1)
        title, message, detail = errors[0]
        self.assertEqual(title, "Could Not Start SAM3")
        self.assertIn("could not be started", message)
        self.assertIn("enter at least one class", detail["next_action"].lower())
        self.assertIn("Enter at least one class prompt", detail["details"])

    def test_annotation_buttons_are_backed_by_shared_actions(self):
        window, _controller, _tasks, _errors, _infos, _confirmations = (
            self.make_controller()
        )
        panel = window.annotation

        expected = (
            (panel.apply_box_button, window.actions.apply_box, "Apply Box"),
            (panel.delete_button, window.actions.delete_annotation, "Delete"),
            (panel.resegment_button, window.actions.resegment, "Re-segment"),
            (panel.reset_sam3_button, window.actions.reset_sam3, "Reset to SAM3"),
        )
        for button, action, label in expected:
            with self.subTest(label=label):
                self.assertIs(button.defaultAction(), action)
                self.assertEqual(button.text(), label)
                self.assertEqual(button.toolTip(), action.toolTip())

        self.assertEqual(
            window.actions.resegment.toolTip(),
            "Generate a new mask/polygon from the selected bounding box.",
        )

    def test_annotation_panel_uses_model_view_for_segmentation_status(self):
        window = MainWindow()
        self.windows.append(window)
        valid = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            id="valid",
            source=AnnotationSource.SAM3,
            polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
        )
        none = Annotation(
            0,
            "car",
            (50, 50, 80, 80),
            id="none",
            source=AnnotationSource.MANUAL,
        )
        invalid = Annotation(
            0,
            "car",
            (5, 5, 20, 20),
            id="invalid",
            source=AnnotationSource.SAM3,
            polygon_xyn=[[0.05, 0.05], [0.2, 0.2]],
        )

        window.annotation.set_classes(["car"])
        window.annotation.set_annotations([valid, none, invalid])
        window.annotation.show_details(valid)

        self.assertEqual(self.model_text(window.annotation, "valid", 2), "valid")
        self.assertEqual(self.model_text(window.annotation, "none", 2), "none")
        self.assertEqual(self.model_text(window.annotation, "invalid", 2), "invalid")
        self.assertEqual(window.annotation.segmentation_label.text(), "Segmentation: valid")

    def test_results_preview_uses_pixmap_and_rejects_invalid_image(self):
        window = MainWindow()
        self.windows.append(window)
        image_path = self.create_image("preview.png", (24, 16))

        self.assertTrue(window.results.set_preview(image_path))
        self.assertFalse(window.results.preview_thumb.pixmap().isNull())
        self.assertFalse(window.results.set_preview(self.root / "missing.png"))
        self.assertTrue(window.results.preview_thumb._source.isNull())
        self.assertEqual(window.results.preview_label.text(), "-")

    def test_canvas_selection_changed_prunes_deleted_graphics_items(self):
        canvas = ImageCanvas()
        self.widgets.append(canvas)
        image_path = self.create_image()
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
        self.widgets.append(canvas)
        image_path = self.create_image()
        canvas.load_image(image_path)
        annotation = Annotation(0, "object", (10, 10, 40, 40), id="ann-1")
        canvas.set_annotations([annotation])
        canvas.select_annotation(annotation.id)

        canvas.set_annotations([annotation])

        self.assertEqual(canvas.selected_annotation_id(), annotation.id)

    def test_controller_box_edit_invalidates_segmentation_and_refreshes_view(self):
        window, controller, _tasks, _errors, _infos, _confirmations = (
            self.make_controller()
        )
        image = self.open_image_project(window, controller)
        annotation = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            id="ann-1",
            source=AnnotationSource.SAM3,
            polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
        )
        image.replace_sam3_drafts([annotation])
        controller.presentation.load_current_image()
        controller.annotations.select_annotation(annotation.id)

        controller.annotations.canvas_box_changed(annotation.id, (12, 12, 42, 42))

        self.assertFalse(annotation.segmentation_valid)
        self.assertEqual(self.model_text(window.annotation, annotation.id, 2), "stale")
        self.assertIn("stale", window.annotation.segmentation_label.text())
        self.assertIn("Re-segment", window.statusBar().currentMessage())
        self.assertTrue(controller.dirty)

    def test_controller_class_edit_invalidates_segmentation_and_refreshes_view(self):
        window, controller, _tasks, _errors, _infos, _confirmations = (
            self.make_controller()
        )
        image = self.open_image_project(window, controller, prompts=("car", "truck"))
        annotation = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            id="ann-1",
            source=AnnotationSource.SAM3,
            polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
        )
        image.replace_sam3_drafts([annotation])
        controller.presentation.load_current_image()
        controller.annotations.select_annotation(annotation.id)
        window.annotation.class_combo.setCurrentText("truck")

        controller.annotations.apply_selected_class()

        self.assertEqual((annotation.class_id, annotation.class_name), (1, "truck"))
        self.assertFalse(annotation.segmentation_valid)
        self.assertEqual(self.model_text(window.annotation, annotation.id, 2), "stale")
        self.assertIn("stale", window.annotation.segmentation_label.text())
        self.assertIn("Re-segment", window.statusBar().currentMessage())

    def test_controller_runs_prediction_through_task_boundary_without_model_load(self):
        window, controller, tasks, errors, _infos, _confirmations = (
            self.make_controller()
        )
        model_path = self.root / "fake-model.pt"
        model_path.write_bytes(b"not real weights")
        image = self.open_image_project(
            window,
            controller,
            model_path=model_path,
        )

        controller.inference.run_current()

        self.assertEqual(len(tasks.prediction_calls), 1)
        image_index, settings = tasks.prediction_calls[0]
        self.assertEqual(image_index, image.image_index)
        self.assertEqual(settings["image_path"], image.image_path)
        self.assertEqual(settings["model_path"], str(model_path))
        self.assertEqual(settings["prompts"], ["car"])
        self.assertEqual(controller.mode, UiMode.PREDICTING)
        self.assertFalse(window.canvas.isEnabled())
        self.assertFalse(window.actions.run_current.isEnabled())

        predicted = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            id="predicted",
            source=AnnotationSource.SAM3,
            confidence=0.9,
            polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
        )
        tasks.prediction_ready.emit(
            image.image_index,
            ImagePrediction(
                Path(image.image_path), [predicted], 100, 80, False
            ),
        )
        tasks.finish()

        self.assertEqual(errors, [])
        self.assertEqual(image.status, ImageStatus.PREDICTED)
        self.assertEqual(window.annotation.annotation_model.rowCount(), 1)
        self.assertEqual(controller.mode, UiMode.READY)
        self.assertTrue(window.canvas.isEnabled())
        self.assertIn("SAM3 complete", window.task_progress.status_label.text())

    def test_controller_presents_prediction_failure_and_marks_image_error(self):
        window, controller, tasks, errors, _infos, _confirmations = (
            self.make_controller()
        )
        model_path = self.root / "fake-model.pt"
        model_path.write_bytes(b"not real weights")
        image = self.open_image_project(
            window,
            controller,
            model_path=model_path,
        )
        controller.inference.run_current()

        tasks.prediction_failed.emit(image.image_index, "synthetic out of memory")
        tasks.finish()

        self.assertEqual(image.status, ImageStatus.ERROR)
        self.assertEqual(image.error_message, "synthetic out of memory")
        self.assertEqual(controller.mode, UiMode.READY)
        self.assertEqual(len(errors), 1)
        title, message, detail = errors[0]
        self.assertEqual(title, "SAM3 Error")
        self.assertIn("could not complete", message)
        self.assertIn("available memory", detail["next_action"])
        self.assertEqual(detail["details"], "synthetic out of memory")

    def test_apply_box_then_resegment_uses_clipped_current_coordinates(self):
        window, controller, tasks, errors, _infos, _confirmations = (
            self.make_controller()
        )
        model_path = self.root / "fake-model.pt"
        model_path.write_bytes(b"not real weights")
        image = self.open_image_project(
            window,
            controller,
            model_path=model_path,
        )
        annotation = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            id="ann-1",
            source=AnnotationSource.SAM3,
            polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
        )
        image.replace_sam3_drafts([annotation])
        controller.presentation.load_current_image()
        controller.annotations.select_annotation(annotation.id)
        for field, value in zip(
            (
                window.annotation.x1_edit,
                window.annotation.y1_edit,
                window.annotation.x2_edit,
                window.annotation.y2_edit,
            ),
            (0.0, 0.0, 120.0, 90.0),
        ):
            field.set_value(value)

        controller.annotations.apply_box_fields()
        controller.inference.resegment_selected()

        self.assertEqual(errors, [])
        self.assertEqual(annotation.box_xyxy, (0.0, 0.0, 100.0, 80.0))
        self.assertFalse(annotation.segmentation_valid)
        self.assertEqual(len(tasks.segmentation_calls), 1)
        _image_index, annotation_id, settings = tasks.segmentation_calls[0]
        self.assertEqual(annotation_id, annotation.id)
        self.assertEqual(settings["box_xyxy"], annotation.box_xyxy)
        self.assertFalse(window.annotation.x1_edit.isEnabled())

        result = BoxSegmentation(
            image_path=Path(image.image_path),
            box_xyxy=annotation.box_xyxy,
            polygon_xyn=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            confidence=0.82,
            reused_predictor=False,
        )
        tasks.segmentation_ready.emit(image.image_index, annotation.id, result)
        tasks.finish()

        self.assertTrue(annotation.segmentation_valid)
        self.assertEqual(annotation.source, AnnotationSource.SAM3_REFINED)
        self.assertEqual(self.model_text(window.annotation, annotation.id, 2), "valid")
        self.assertEqual(window.annotation.segmentation_label.text(), "Segmentation: valid")
        self.assertEqual(controller.mode, UiMode.READY)

    def test_invalid_box_fields_do_not_mutate_or_start_resegmentation(self):
        window, controller, tasks, errors, _infos, _confirmations = (
            self.make_controller()
        )
        image = self.open_image_project(window, controller)
        annotation = Annotation(0, "car", (10, 20, 30, 40), id="ann-1")
        image.replace_sam3_drafts([annotation])
        controller.presentation.load_current_image()
        controller.annotations.select_annotation(annotation.id)
        before = annotation.to_dict()
        for field, value in zip(
            (
                window.annotation.x1_edit,
                window.annotation.y1_edit,
                window.annotation.x2_edit,
                window.annotation.y2_edit,
            ),
            (35.0, 20.0, 25.0, 40.0),
        ):
            field.set_value(value)

        controller.annotations.apply_box_fields()

        self.assertEqual(annotation.to_dict(), before)
        self.assertEqual(tasks.segmentation_calls, [])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0][0], "Invalid Box Coordinates")
        self.assertIn("x1 < x2", errors[0][2]["next_action"])

    def test_annotation_rect_resize_clips_to_image_bounds(self):
        annotation = Annotation(0, "object", (10, 10, 50, 50))
        item = AnnotationRectItem(annotation, QRectF(0, 0, 100, 100))
        item._active_handle = "top_left"
        item._press_scene_rect = QRectF(QPointF(10, 10), QPointF(50, 50))

        item._resize_from_handle(QPointF(-20, -20))
        rect = item._scene_rect()

        self.assertEqual(
            (rect.left(), rect.top(), rect.right(), rect.bottom()),
            (0.0, 0.0, 50.0, 50.0),
        )

    def test_annotation_rect_resize_enforces_minimum_size(self):
        annotation = Annotation(0, "object", (10, 10, 50, 50))
        item = AnnotationRectItem(annotation, QRectF(0, 0, 100, 100))
        item._active_handle = "right"
        item._press_scene_rect = QRectF(QPointF(10, 10), QPointF(50, 50))

        item._resize_from_handle(QPointF(10.5, 30))

        self.assertEqual(item._scene_rect().width(), 2.0)

    def test_annotation_resize_handles_keep_device_size_when_canvas_scales(self):
        annotation = Annotation(0, "object", (10, 10, 50, 50))
        item = AnnotationRectItem(annotation, QRectF(0, 0, 100, 100))

        item.apply_style(True)

        self.assertEqual(len(item._handle_items), 8)
        for handle in item._handle_items.values():
            self.assertTrue(handle.isVisible())
            self.assertTrue(
                handle.flags() & QGraphicsItem.ItemIgnoresTransformations
            )


if __name__ == "__main__":
    unittest.main()
