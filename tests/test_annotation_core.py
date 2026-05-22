import tempfile
import unittest
from pathlib import Path

from sam3_auto_annotator.annotation.converters import (
    build_box_rows,
    build_detection_export,
)
from sam3_auto_annotator.annotation.export import export_corrected_detection
from sam3_auto_annotator.annotation.geometry import (
    clip_xyxy,
    validate_xyxy,
    xyxy_to_xywh,
    xyxy_to_yolo_xywhn,
)
from sam3_auto_annotator.annotation.models import (
    Annotation,
    AnnotationSource,
    ImageRecord,
    ImageStatus,
    ProjectState,
)
from sam3_auto_annotator.annotation.sam3 import annotations_from_sam3_result, result_image_size
from sam3_auto_annotator.annotation.store import load_project_state, save_project_state
from sam3_auto_annotator.exporters.yolo_exporter import write_yolo_labels


class FakeBoxes:
    def __init__(self):
        self.xyxy = [(10, 20, 50, 80), (60, 10, 90, 40)]
        self.cls = [0, 1]
        self.conf = [0.9, 0.75]


class FakeMasks:
    def __init__(self):
        self.xyn = [[(0.1, 0.2), (0.5, 0.8)], [(0.6, 0.1), (0.9, 0.4)]]


class FakeResult:
    orig_shape = (100, 120)
    boxes = FakeBoxes()
    masks = FakeMasks()


class GeometryTests(unittest.TestCase):
    def test_xyxy_to_xywh_and_yolo_normalized(self):
        box = (10, 20, 30, 60)

        self.assertEqual(xyxy_to_xywh(box), (20.0, 40.0, 20.0, 40.0))
        self.assertEqual(xyxy_to_yolo_xywhn(box, 100, 200), (0.2, 0.2, 0.2, 0.2))

    def test_clip_xyxy_limits_coordinates_to_image_bounds(self):
        self.assertEqual(clip_xyxy((-10, 5, 120, 90), 100, 80), (0.0, 5.0, 100.0, 80.0))

    def test_validate_xyxy_rejects_empty_or_inverted_boxes(self):
        with self.assertRaises(ValueError):
            validate_xyxy((10, 10, 10, 20))
        with self.assertRaises(ValueError):
            validate_xyxy((30, 20, 10, 40))


class AnnotationModelTests(unittest.TestCase):
    def test_active_annotations_exclude_deleted_items(self):
        image = ImageRecord(
            image_path="images/a.jpg",
            image_index=0,
            annotations=[
                Annotation(0, "car", (0, 0, 10, 10)),
                Annotation(1, "person", (20, 20, 30, 40), deleted=True),
            ],
        )

        self.assertEqual(len(image.annotations), 2)
        self.assertEqual(len(image.active_annotations), 1)
        self.assertEqual(image.active_annotations[0].class_name, "car")

    def test_editing_sam3_annotation_changes_source_and_keeps_id(self):
        annotation = Annotation(
            0,
            "car",
            (5, 5, 20, 20),
            id="draft-1",
            source=AnnotationSource.SAM3,
            confidence=0.9,
        )

        annotation.edit_box((10, 10, 50, 60))

        self.assertEqual(annotation.id, "draft-1")
        self.assertEqual(annotation.source, AnnotationSource.EDITED)
        self.assertEqual(annotation.box_xyxy, (10.0, 10.0, 50.0, 60.0))
        self.assertFalse(annotation.deleted)

    def test_manual_annotation_stays_manual_after_edit(self):
        image = ImageRecord("images/a.jpg", image_index=0)
        annotation = image.add_manual_annotation(1, "person", (1, 2, 11, 22))

        annotation.edit_box((2, 3, 12, 23))

        self.assertEqual(annotation.source, AnnotationSource.MANUAL)
        self.assertEqual(image.status, ImageStatus.EDITED)

    def test_project_state_supports_single_image_and_folder_like_inputs(self):
        single = ProjectState.from_image_paths(
            input_path="images/a.jpg",
            image_paths=[Path("images/a.jpg")],
            prompts=["car"],
        )
        folder = ProjectState.from_image_paths(
            input_path="images",
            image_paths=[Path("images/a.jpg"), Path("images/b.jpg")],
            prompts=["car", "person"],
        )

        self.assertTrue(single.is_single_image)
        self.assertFalse(folder.is_single_image)
        self.assertEqual(folder.class_map, {"car": 0, "person": 1})
        self.assertEqual([image.image_index for image in folder.images], [0, 1])
        self.assertEqual(len(folder.unpredicted_images), 2)

    def test_json_save_load_round_trip(self):
        project = ProjectState.from_image_paths(
            input_path="images",
            image_paths=[Path("images/a.jpg")],
            prompts=["car"],
            model_path="models/sam3.pt",
            project_name="demo",
        )
        image = project.get_image(0)
        image.width = 100
        image.height = 80
        image.replace_sam3_drafts(
            [Annotation(0, "car", (10, 10, 40, 50), id="ann-1", confidence=0.7)]
        )
        image.annotations[0].mark_deleted()
        image.add_manual_annotation(0, "car", (20, 20, 60, 70))

        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "annotation_state.json"
            save_project_state(project, state_path)
            loaded = load_project_state(state_path)

        loaded_image = loaded.get_image(0)
        self.assertEqual(loaded.project_name, "demo")
        self.assertEqual(loaded.model_path, "models/sam3.pt")
        self.assertEqual(loaded_image.status, ImageStatus.EDITED)
        self.assertEqual(len(loaded_image.annotations), 2)
        self.assertEqual(len(loaded_image.active_annotations), 1)
        self.assertEqual(loaded_image.active_annotations[0].source, AnnotationSource.MANUAL)


class ExportPreparationTests(unittest.TestCase):
    def test_build_box_rows_uses_active_corrected_annotations_only(self):
        project = ProjectState.from_image_paths(
            input_path="images",
            image_paths=[Path("images/a.jpg")],
            prompts=["car"],
        )
        image = project.get_image(0)
        image.width = 100
        image.height = 100
        draft = Annotation(0, "car", (10, 10, 40, 40), id="draft")
        image.replace_sam3_drafts([draft])
        draft.edit_box((20, 20, 60, 80))
        image.add_manual_annotation(0, "car", (0, 0, 10, 10))

        rows = build_box_rows(project)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["object_index"], 0)
        self.assertEqual(rows[0]["x1"], "20.000000")
        self.assertEqual(rows[0]["y1"], "20.000000")
        self.assertEqual(rows[0]["x_center_norm"], "0.400000")
        self.assertEqual(rows[0]["height_norm"], "0.600000")
        self.assertEqual(rows[0]["class_count_in_image"], 2)
        self.assertEqual(rows[1]["x1"], "0.000000")

    def test_deleted_draft_is_not_exported_when_manual_replacement_exists(self):
        project = ProjectState.from_image_paths(
            input_path="images",
            image_paths=[Path("images/a.jpg")],
            prompts=["car"],
        )
        image = project.get_image(0)
        image.width = 100
        image.height = 100
        draft = Annotation(0, "car", (10, 10, 40, 40), id="draft")
        image.replace_sam3_drafts([draft])
        draft.mark_deleted()
        image.add_manual_annotation(0, "car", (50, 50, 90, 90))

        rows = build_box_rows(project)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["x1"], "50.000000")
        self.assertEqual(rows[0]["width_norm"], "0.400000")

    def test_empty_annotations_still_return_image_paths_for_empty_yolo_files(self):
        project = ProjectState.from_image_paths(
            input_path="images",
            image_paths=[Path("images/a.jpg"), Path("images/b.jpg")],
            prompts=["car"],
        )
        project.get_image(0).status = ImageStatus.NO_DETECTION

        image_paths, rows = build_detection_export(project)

        self.assertEqual(image_paths, [Path("images/a.jpg"), Path("images/b.jpg")])
        self.assertEqual(rows, [])
        self.assertEqual(len(project.unpredicted_images), 1)

    def test_empty_export_inputs_create_empty_yolo_detection_files(self):
        project = ProjectState.from_image_paths(
            input_path="images",
            image_paths=[Path("images/a.jpg")],
            prompts=["car"],
        )
        image_paths, rows = build_detection_export(project)

        with tempfile.TemporaryDirectory() as temp_dir:
            _, detection_dir = write_yolo_labels(Path(temp_dir), image_paths, [], rows)
            detection_path = detection_dir / "a.txt"

            self.assertTrue(detection_path.exists())
            self.assertEqual(detection_path.read_text(encoding="utf-8"), "")

    def test_export_corrected_detection_writes_csv_and_yolo_from_state(self):
        project = ProjectState.from_image_paths(
            input_path="images",
            image_paths=[Path("images/a.jpg"), Path("images/b.jpg")],
            prompts=["car"],
        )
        image = project.get_image(0)
        image.width = 100
        image.height = 100
        image.add_manual_annotation(0, "car", (10, 10, 50, 50))

        with tempfile.TemporaryDirectory() as temp_dir:
            result = export_corrected_detection(project, Path(temp_dir))

            self.assertTrue(result["box_csv"].exists())
            self.assertTrue((result["yolo_detection_dir"] / "a.txt").exists())
            self.assertTrue((result["yolo_detection_dir"] / "b.txt").exists())
            self.assertEqual((result["yolo_detection_dir"] / "b.txt").read_text(encoding="utf-8"), "")
            self.assertIn("0 0.300000 0.300000 0.400000 0.400000", (result["yolo_detection_dir"] / "a.txt").read_text(encoding="utf-8"))

    def test_build_box_rows_clips_coordinates_before_export(self):
        project = ProjectState.from_image_paths(
            input_path="images",
            image_paths=[Path("images/a.jpg")],
            prompts=["car"],
        )
        image = project.get_image(0)
        image.width = 100
        image.height = 80
        image.add_manual_annotation(0, "car", (-5, 10, 120, 90))

        rows = build_box_rows(project)

        self.assertEqual(rows[0]["x1"], "0.000000")
        self.assertEqual(rows[0]["x2"], "100.000000")
        self.assertEqual(rows[0]["height_norm"], "0.875000")


class Sam3ConversionTests(unittest.TestCase):
    def test_sam3_result_converts_to_editable_annotations(self):
        annotations = annotations_from_sam3_result(FakeResult(), ["car", "person"])

        self.assertEqual(len(annotations), 2)
        self.assertEqual(annotations[0].class_name, "car")
        self.assertEqual(annotations[0].source, AnnotationSource.SAM3)
        self.assertEqual(annotations[0].confidence, 0.9)
        self.assertEqual(annotations[1].polygon_xyn, [[0.6, 0.1], [0.9, 0.4]])

    def test_result_image_size_uses_original_shape_width_height_order(self):
        self.assertEqual(result_image_size(FakeResult()), (120, 100))


if __name__ == "__main__":
    unittest.main()
