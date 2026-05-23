import tempfile
import unittest
from pathlib import Path

from sam3_auto_annotator.annotation.export import export_corrected_detection
from sam3_auto_annotator.annotation.models import (
    Annotation,
    AnnotationSource,
    ImageRecord,
    ProjectState,
)
from sam3_auto_annotator.annotation.sam3 import annotations_from_sam3_result
from sam3_auto_annotator.annotation.segmentation import (
    build_segmentation_rows,
    has_valid_segmentation,
    polygon_xyn_to_pixels,
)


class FakeBoxes:
    xyxy = [(10, 20, 50, 80)]
    cls = [0]
    conf = [0.9]


class FakeMasks:
    xyn = [[(0.1, 0.2), (0.5, 0.2), (0.5, 0.8), (0.1, 0.8)]]


class FakeResult:
    orig_shape = (100, 120)
    boxes = FakeBoxes()
    masks = FakeMasks()


class SegmentationPreviewTests(unittest.TestCase):
    def test_sam3_polygon_data_is_stored_on_annotation(self):
        annotations = annotations_from_sam3_result(FakeResult(), ["car"])

        self.assertEqual(len(annotations), 1)
        self.assertEqual(
            annotations[0].polygon_xyn,
            [[0.1, 0.2], [0.5, 0.2], [0.5, 0.8], [0.1, 0.8]],
        )

    def test_normalized_polygon_converts_to_pixel_points(self):
        points = polygon_xyn_to_pixels([[-0.5, 0.25], [0.5, 1.5]], 200, 100)

        self.assertEqual(points, [(0.0, 25.0), (100.0, 100.0)])

    def test_only_untouched_sam3_annotations_have_valid_segmentation(self):
        sam3 = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            source=AnnotationSource.SAM3,
            polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
        )
        manual = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            source=AnnotationSource.MANUAL,
            polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
        )
        imported = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            source=AnnotationSource.IMPORTED,
            polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
        )

        self.assertTrue(has_valid_segmentation(sam3))
        sam3.edit_box((12, 12, 42, 42), 100, 100)
        self.assertEqual(sam3.source, AnnotationSource.EDITED)
        self.assertFalse(has_valid_segmentation(sam3))
        self.assertFalse(has_valid_segmentation(manual))
        self.assertFalse(has_valid_segmentation(imported))

    def test_edited_sam3_annotation_can_reset_to_original_segmentation(self):
        annotation = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            source=AnnotationSource.SAM3,
            polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
            original_box_xyxy=(10, 10, 40, 40),
            original_class_id=0,
            original_class_name="car",
        )
        project = ProjectState(
            input_path="images",
            prompts=["car", "truck"],
            images=[ImageRecord("images/a.jpg", 0, width=100, height=100, annotations=[annotation])],
        )

        annotation.edit_box((20, 20, 60, 60), 100, 100)
        annotation.change_class(1, "truck")

        self.assertEqual(annotation.source, AnnotationSource.EDITED)
        self.assertFalse(has_valid_segmentation(annotation))
        self.assertEqual(build_segmentation_rows(project), [])

        annotation.reset_to_sam3()

        self.assertEqual(annotation.source, AnnotationSource.SAM3)
        self.assertEqual(annotation.box_xyxy, (10.0, 10.0, 40.0, 40.0))
        self.assertEqual(annotation.class_id, 0)
        self.assertEqual(annotation.class_name, "car")
        self.assertTrue(has_valid_segmentation(annotation))
        self.assertEqual(len(build_segmentation_rows(project)), 1)

    def test_sam3_converter_preserves_original_box_for_reset(self):
        annotation = annotations_from_sam3_result(FakeResult(), ["car"])[0]

        self.assertEqual(annotation.original_box_xyxy, (10.0, 20.0, 50.0, 80.0))
        self.assertEqual(annotation.original_class_id, 0)
        self.assertEqual(annotation.original_class_name, "car")
        self.assertTrue(annotation.can_reset_to_sam3)

    def test_segmentation_rows_include_only_valid_sam3_polygons(self):
        project = ProjectState(
            input_path="images",
            prompts=["car"],
            images=[ImageRecord("images/a.jpg", 0, width=100, height=100)],
        )
        image = project.get_image(0)
        image.replace_sam3_drafts(
            [
                Annotation(
                    0,
                    "car",
                    (10, 10, 40, 40),
                    source=AnnotationSource.SAM3,
                    polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
                ),
                Annotation(0, "car", (50, 50, 80, 80), source=AnnotationSource.MANUAL),
            ]
        )

        rows = build_segmentation_rows(project)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["polygon_point_count"], 3)
        self.assertEqual(rows[0]["yolo_segmentation_line"], "0 0.100000 0.100000 0.400000 0.100000 0.400000 0.400000")

    def test_export_writes_segmentation_only_for_untouched_sam3_polygons(self):
        project = ProjectState(
            input_path="images",
            prompts=["car"],
            images=[ImageRecord("images/a.jpg", 0, width=100, height=100)],
        )
        image = project.get_image(0)
        image.replace_sam3_drafts(
            [
                Annotation(
                    0,
                    "car",
                    (10, 10, 40, 40),
                    source=AnnotationSource.SAM3,
                    polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
                )
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = export_corrected_detection(project, Path(temp_dir))
            segmentation_path = result["yolo_segmentation_dir"] / "a.txt"
            detection_path = result["yolo_detection_dir"] / "a.txt"

            self.assertIn("0 0.100000 0.100000 0.400000", segmentation_path.read_text(encoding="utf-8"))
            self.assertIn("0 0.250000 0.250000 0.300000 0.300000", detection_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
