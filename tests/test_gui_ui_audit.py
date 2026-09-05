import os
import tempfile
import unittest
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from PySide6.QtCore import QCoreApplication, QEvent, QObject, Signal
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from sam3_auto_annotator.core import (
    Annotation,
    AnnotationSource,
    ImageStatus,
    ProjectState,
)
from sam3_auto_annotator.gui.controller import AppController, UiMode
from sam3_auto_annotator.gui.main_window import MainWindow
from sam3_auto_annotator.gui.tasks.inference_task_manager import TaskKind


class MemorySettings:
    def last_directory(self):
        return ""

    def set_last_directory(self, _path):
        pass

    def save_window(self, _window, _splitter):
        pass


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
        self.batch_calls = []

    def _start(self, kind):
        if self.is_running:
            raise RuntimeError("A fake task is already running.")
        self.is_running = True
        self.kind = kind
        self.task_started.emit(kind.value)

    def start_prediction(self, image_index, **settings):
        self._start(TaskKind.PREDICTION)
        self.prediction_calls.append((image_index, dict(settings)))

    def start_segmentation(self, *_args, **_settings):
        self._start(TaskKind.SEGMENTATION)

    def start_batch(self, items, **settings):
        self._start(TaskKind.BATCH)
        self.batch_calls.append((list(items), dict(settings)))

    def request_cancel(self):
        return self.is_running

    def finish(self):
        kind = self.kind.value if self.kind is not None else "unknown"
        self.is_running = False
        self.kind = None
        self.task_finished.emit(kind)


class GuiUiAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.windows = []

    def tearDown(self):
        for window in self.windows:
            window.controller = None
            window.close()
            window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QCoreApplication.processEvents()
        self.temp_dir.cleanup()

    def make_image(self, name, *, size=(100, 80), color="#2563eb"):
        path = self.root / name
        image = QImage(size[0], size[1], QImage.Format_RGB32)
        image.fill(QColor(color))
        self.assertTrue(image.save(str(path)))
        return path

    def make_window(self, *, size=(960, 620)):
        window = MainWindow()
        window.resize(*size)
        window.show_error = lambda *_args, **_kwargs: None
        window.show_info = lambda *_args, **_kwargs: None
        window.confirm = lambda *_args, **_kwargs: True
        window.show()
        self.windows.append(window)
        QCoreApplication.processEvents()
        return window

    def make_controller(self, *, size=(960, 620)):
        window = self.make_window(size=size)
        tasks = FakeTaskManager()
        controller = AppController(window, MemorySettings(), task_manager=tasks)
        return window, controller, tasks

    def make_project(
        self,
        name,
        image_paths,
        *,
        prompts=("car",),
        model_path=None,
    ):
        model_path = model_path or self.root / "model.pt"
        Path(model_path).touch(exist_ok=True)
        project = ProjectState.from_image_paths(
            input_path=image_paths[0] if len(image_paths) == 1 else self.root,
            image_paths=image_paths,
            prompts=prompts,
            model_path=model_path,
            project_name=name,
            half=False,
        )
        for image in project.images:
            image.width = 100
            image.height = 80
        return project

    def assert_footer_widget_visible(self, panel, widget):
        self.assertTrue(widget.isVisibleTo(panel))
        top_left = widget.mapTo(panel, widget.rect().topLeft())
        bottom_right = widget.mapTo(panel, widget.rect().bottomRight())
        self.assertGreaterEqual(top_left.y(), 0)
        self.assertLess(bottom_right.y(), panel.height())

    def test_minimum_window_setup_and_results_keep_footers_visible_without_horizontal_overflow(self):
        window = self.make_window()

        cases = (
            (window.setup, (window.setup.run_button, window.setup.run_all_button)),
            (
                window.results,
                (window.results.export_button, window.results.open_output_button),
            ),
        )
        for panel, footer_widgets in cases:
            with self.subTest(panel=panel.objectName()):
                window.inspector.setCurrentWidget(panel)
                QCoreApplication.processEvents()
                self.assertEqual(panel.scroll_area.horizontalScrollBar().maximum(), 0)
                for widget in footer_widgets:
                    self.assert_footer_widget_visible(panel, widget)

        window.inspector.setCurrentWidget(window.annotation)
        QCoreApplication.processEvents()
        self.assertEqual(
            window.annotation.editor_scroll.horizontalScrollBar().maximum(),
            0,
        )
        self.assertGreaterEqual(
            window.annotation.annotation_table.viewport().height(),
            window.annotation.annotation_table.verticalHeader().defaultSectionSize() * 2,
        )

    def test_long_progress_text_does_not_resize_horizontal_workspace_panels(self):
        window = self.make_window()
        window.workspace.setSizes([190, 490, 270])
        QCoreApplication.processEvents()
        before = window.workspace.sizes()

        window.task_progress.show_running(
            "Processing " + "extremely-long-image-name-" * 80 + ".png",
            maximum=17,
            cancellable=True,
        )
        QCoreApplication.processEvents()

        self.assertEqual(window.workspace.sizes(), before)
        self.assertEqual(window.task_progress.status_label.toolTip(), window.task_progress.status_label.text())

    def test_unchanged_box_and_class_are_noops_and_keep_segmentation_valid(self):
        image_path = self.make_image("annotated.png")
        window, controller, _tasks = self.make_controller()
        project = self.make_project(
            "no-op-edits",
            [image_path],
            prompts=("car", "truck"),
        )
        annotation = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            id="ann-valid",
            source=AnnotationSource.SAM3,
            confidence=0.91,
            polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
        )
        project.images[0].replace_sam3_drafts([annotation])
        controller._load_project(project)
        controller.select_annotation(annotation.id)
        controller.dirty = False
        controller._update_actions()
        before = annotation.to_dict()

        self.assertFalse(window.actions.apply_box.isEnabled())
        self.assertFalse(window.actions.apply_class.isEnabled())
        controller.apply_box_fields()
        controller.apply_selected_class()

        self.assertEqual(annotation.to_dict(), before)
        self.assertTrue(annotation.segmentation_valid)
        self.assertFalse(controller.dirty)
        self.assertFalse(window.actions.apply_box.isEnabled())
        self.assertFalse(window.actions.apply_class.isEnabled())

    def test_controller_canvas_and_table_selection_clear_together(self):
        image_path = self.make_image("selection.png")
        window, controller, _tasks = self.make_controller()
        project = self.make_project("selection-sync", [image_path])
        annotation = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            id="selection-ann",
        )
        project.images[0].replace_sam3_drafts([annotation])
        controller._load_project(project)
        controller.select_annotation(annotation.id)

        window.canvas.select_annotation(None)
        QCoreApplication.processEvents()

        self.assertIsNone(controller.selected_annotation_id)
        self.assertIsNone(window.canvas.selected_annotation_id())
        self.assertIsNone(window.annotation.selected_annotation_id())
        self.assertEqual(
            window.annotation.selection_label.text(),
            "No annotation selected",
        )

    def test_cancel_is_hidden_for_current_prediction_and_visible_for_batch(self):
        image_path = self.make_image("task.png")
        window, controller, tasks = self.make_controller()
        controller._load_project(self.make_project("task-modes", [image_path]))

        controller.run_current()
        QCoreApplication.processEvents()
        self.assertEqual(controller.mode, UiMode.PREDICTING)
        self.assertTrue(window.task_progress.cancel_button.isHidden())

        tasks.finish()
        controller.run_remaining()
        QCoreApplication.processEvents()
        self.assertEqual(controller.mode, UiMode.BATCH)
        self.assertFalse(window.task_progress.cancel_button.isHidden())
        self.assertTrue(window.task_progress.cancel_button.isEnabled())

        tasks.finish()

    def test_corrupt_image_replaces_stale_canvas_with_error_and_disables_image_actions(self):
        valid_path = self.make_image("valid.png")
        corrupt_path = self.root / "corrupt.png"
        corrupt_path.write_bytes(b"this is not an image")
        window, controller, _tasks = self.make_controller()
        errors = []
        window.show_error = lambda *args, **kwargs: errors.append((args, kwargs))
        project = self.make_project("mixed-images", [valid_path, corrupt_path])
        controller._load_project(project)
        QCoreApplication.processEvents()

        self.assertIsNotNone(window.canvas._pixmap_item)
        self.assertIs(window.canvas_area.workspace_stack.currentWidget(), window.canvas)

        controller.select_image(1)
        QCoreApplication.processEvents()

        self.assertIsNone(window.canvas._pixmap_item)
        self.assertTrue(window.canvas.sceneRect().isNull())
        self.assertIs(
            window.canvas_area.workspace_stack.currentWidget(),
            window.canvas_area.image_load_error,
        )
        self.assertIn("corrupt.png", window.canvas_area.image_load_error.detail_label.text())
        self.assertEqual(len(errors), 1)
        for action in (
            window.actions.run_current,
            window.actions.draw_box,
            window.actions.fit,
            window.actions.mark_reviewed,
            window.actions.save_preview,
        ):
            with self.subTest(action=action.text()):
                self.assertFalse(action.isEnabled())

    def test_loading_another_project_resets_project_specific_ui_state(self):
        first_path = self.make_image("first.png", color="#dc2626")
        second_path = self.make_image("second.png", color="#16a34a")
        window, controller, _tasks = self.make_controller()
        first = self.make_project("first-project", [first_path])
        second = self.make_project("second-project", [second_path], prompts=("person",))
        controller._load_project(first)

        window.dataset.search_edit.setText("no-match")
        window.dataset.status_filter.setCurrentIndex(3)
        window.inspector.setCurrentWidget(window.results)
        window.actions.draw_box.setChecked(True)
        window.task_progress.show_result("Stale task result")
        self.assertTrue(window.results.set_preview(first_path))
        controller.last_preview_path = first_path
        QCoreApplication.processEvents()

        controller._load_project(second)
        QCoreApplication.processEvents()

        self.assertIs(controller.project, second)
        self.assertEqual(controller.current_image_index, 0)
        self.assertEqual(window.dataset.search_edit.text(), "")
        self.assertEqual(window.dataset.status_filter.currentIndex(), 0)
        self.assertEqual(window.dataset.image_model.rowCount(), 1)
        self.assertEqual(window.dataset.image_model.images[0].image_name, "second.png")
        self.assertIs(window.inspector.currentWidget(), window.setup)
        self.assertFalse(window.actions.draw_box.isChecked())
        self.assertFalse(window.task_progress.isVisible())
        self.assertIsNone(controller.selected_annotation_id)
        self.assertEqual(window.annotation.annotation_model.rowCount(), 0)
        self.assertEqual(window.annotation.selection_label.text(), "No annotation selected")
        self.assertTrue(window.results.preview_thumb._source.isNull())
        self.assertIsNone(controller.last_preview_path)
        self.assertIn("Project ready", window.results.result_status_label.text())

    def test_save_keeps_existing_export_result_paths_visible(self):
        image_path = self.make_image("save-paths.png")
        window, controller, _tasks = self.make_controller()
        controller._load_project(self.make_project("save-paths", [image_path]))
        window.results.set_output_paths(
            output_dir=self.root,
            box_csv=self.root / "existing.csv",
            detection_dir=self.root / "detection",
            segmentation_dir=self.root / "segmentation",
            skipped_report=self.root / "skipped.json",
        )

        controller.save_project()

        self.assertEqual(
            window.results.result_csv_label.text(),
            str(self.root / "existing.csv"),
        )
        self.assertEqual(
            window.results.result_detection_label.text(),
            str(self.root / "detection"),
        )
        self.assertEqual(
            window.results.result_segmentation_label.text(),
            str(self.root / "segmentation"),
        )

    def test_long_status_context_does_not_expand_window_minimum_width(self):
        window = self.make_window()
        before = window.minimumSizeHint().width()
        message = "image-" + "very-long-name-" * 80 + ".png | 42 annotations | unsaved"

        window.set_status_context(message)
        QCoreApplication.processEvents()

        self.assertLessEqual(window.minimumSizeHint().width(), before)
        self.assertEqual(window.status_context.toolTip(), message)

    def test_empty_project_clears_previous_image_and_annotation_content(self):
        image_path = self.make_image("populated.png")
        window, controller, _tasks = self.make_controller()
        populated = self.make_project("populated", [image_path])
        populated.images[0].replace_sam3_drafts(
            [Annotation(0, "car", (10, 10, 40, 40), id="old-ann")]
        )
        controller._load_project(populated)
        empty = ProjectState(
            input_path=self.root,
            images=[],
            prompts=["car"],
            model_path=self.root / "model.pt",
            project_name="empty",
            half=False,
        )

        controller._load_project(empty)

        self.assertIsNone(controller.current_image)
        self.assertIsNone(window.canvas._pixmap_item)
        self.assertEqual(window.annotation.annotation_model.rowCount(), 0)
        self.assertIn("does not contain any images", window.canvas_area.canvas_hint.text())

    def test_clearing_no_match_filter_restores_selection_without_reloading(self):
        first = self.make_image("first-filter.png")
        second = self.make_image("second-filter.png")
        window, controller, _tasks = self.make_controller()
        controller._load_project(self.make_project("filters", [first, second]))
        original_item = window.canvas._pixmap_item

        window.dataset.search_edit.setText("no-match")
        QCoreApplication.processEvents()
        self.assertEqual(window.dataset.filter_model.rowCount(), 0)
        self.assertEqual(controller.current_image_index, 0)

        window.dataset.search_edit.clear()
        QCoreApplication.processEvents()

        self.assertEqual(window.dataset.selected_image_index(), 0)
        self.assertIs(window.canvas._pixmap_item, original_item)
        self.assertFalse(window.actions.previous_image.isEnabled())
        self.assertTrue(window.actions.next_image.isEnabled())
        self.assertNotIn("hidden by", window.canvas_area.canvas_hint.text())
        self.assertIn("first-filter.png", window.canvas_area.canvas_hint.text())

    def test_navigation_actions_match_first_middle_and_last_image(self):
        paths = [
            self.make_image("first-nav.png"),
            self.make_image("middle-nav.png"),
            self.make_image("last-nav.png"),
        ]
        window, controller, _tasks = self.make_controller()
        controller._load_project(self.make_project("navigation", paths))

        self.assertEqual(controller.current_image_index, 0)
        self.assertFalse(window.actions.previous_image.isEnabled())
        self.assertTrue(window.actions.next_image.isEnabled())

        self.assertTrue(window.dataset.select_image(1))
        QCoreApplication.processEvents()
        self.assertEqual(controller.current_image_index, 1)
        self.assertTrue(window.actions.previous_image.isEnabled())
        self.assertTrue(window.actions.next_image.isEnabled())

        self.assertTrue(window.dataset.select_image(2))
        QCoreApplication.processEvents()
        self.assertEqual(controller.current_image_index, 2)
        self.assertTrue(window.actions.previous_image.isEnabled())
        self.assertFalse(window.actions.next_image.isEnabled())

    def test_run_pending_is_disabled_when_no_prediction_targets_remain(self):
        image_path = self.make_image("already-predicted.png")
        window, controller, _tasks = self.make_controller()
        project = self.make_project("complete-batch", [image_path])
        project.images[0].replace_sam3_drafts([])

        controller._load_project(project)

        self.assertEqual(window.actions.run_remaining.text(), "Run Pending (0)")
        self.assertFalse(window.actions.run_remaining.isEnabled())

    def test_mark_reviewed_disables_itself_after_review(self):
        image_path = self.make_image("review.png")
        window, controller, _tasks = self.make_controller()
        controller._load_project(self.make_project("review-state", [image_path]))

        self.assertTrue(window.actions.mark_reviewed.isEnabled())
        controller.mark_current_reviewed()

        self.assertEqual(controller.current_image.status, ImageStatus.REVIEWED)
        self.assertFalse(window.actions.mark_reviewed.isEnabled())

    def test_reset_is_disabled_for_original_sam3_and_enabled_after_applied_edit(self):
        image_path = self.make_image("reset-state.png")
        window, controller, _tasks = self.make_controller()
        project = self.make_project("reset-action", [image_path])
        annotation = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            id="sam3-original",
            source=AnnotationSource.SAM3,
            confidence=0.88,
            polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
        )
        project.images[0].replace_sam3_drafts([annotation])
        controller._load_project(project)
        controller.select_annotation(annotation.id)

        self.assertFalse(annotation.is_modified_from_sam3)
        self.assertFalse(window.actions.reset_sam3.isEnabled())

        window.annotation.x1_edit.set_value(12)
        self.assertTrue(window.actions.apply_box.isEnabled())
        controller.apply_box_fields()

        self.assertTrue(annotation.is_modified_from_sam3)
        self.assertTrue(window.actions.reset_sam3.isEnabled())


if __name__ == "__main__":
    unittest.main()
