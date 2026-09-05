import json
import tempfile
import unittest
from pathlib import Path

from services.export_service import export_corrected_detection
from domain import (
    Annotation,
    AnnotationSource,
    ImageRecord,
    ProjectState,
)
from sam3.result_mapper import (
    annotations_from_sam3_result,
    best_box_prompt_segmentation,
)
from domain.segmentation import (
    build_segmentation_rows,
    build_skipped_segmentation_rows,
    has_valid_segmentation,
    polygon_xyn_to_pixels,
    segmentation_status,
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
        self.assertTrue(annotations[0].segmentation_valid)
        self.assertEqual(annotations[0].segmentation_source, "sam3_original")
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
        self.assertFalse(annotation.segmentation_valid)
        self.assertEqual(build_segmentation_rows(project), [])

        annotation.reset_to_sam3()

        self.assertEqual(annotation.source, AnnotationSource.SAM3)
        self.assertEqual(annotation.box_xyxy, (10.0, 10.0, 40.0, 40.0))
        self.assertEqual(annotation.class_id, 0)
        self.assertEqual(annotation.class_name, "car")
        self.assertEqual(annotation.segmentation_source, "sam3_original")
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

    def test_resegment_update_preserves_bbox_class_and_original_metadata(self):
        annotation = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            source=AnnotationSource.SAM3,
            polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
            original_box_xyxy=(10, 10, 40, 40),
            original_polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
            original_class_id=0,
            original_class_name="car",
        )
        annotation.edit_box((20, 20, 60, 60), 100, 100)
        annotation.apply_sam3_box_prompt_segmentation(
            [[0.2, 0.2], [0.6, 0.2], [0.6, 0.6]],
            confidence=0.8,
        )

        self.assertEqual(annotation.source, AnnotationSource.SAM3_REFINED)
        self.assertEqual(annotation.box_xyxy, (20.0, 20.0, 60.0, 60.0))
        self.assertEqual(annotation.class_id, 0)
        self.assertEqual(annotation.class_name, "car")
        self.assertEqual(annotation.original_box_xyxy, (10.0, 10.0, 40.0, 40.0))
        self.assertEqual(
            annotation.original_polygon_xyn,
            [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
        )
        self.assertTrue(annotation.segmentation_valid)
        self.assertEqual(annotation.segmentation_source, "sam3_box_prompt")
        self.assertTrue(has_valid_segmentation(annotation))

    def test_reset_after_resegment_restores_original_polygon(self):
        original_polygon = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]]
        annotation = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            source=AnnotationSource.SAM3,
            polygon_xyn=original_polygon,
            original_box_xyxy=(10, 10, 40, 40),
            original_polygon_xyn=original_polygon,
            original_class_id=0,
            original_class_name="car",
        )
        annotation.edit_box((20, 20, 60, 60), 100, 100)
        annotation.apply_sam3_box_prompt_segmentation([[0.2, 0.2], [0.6, 0.2], [0.6, 0.6]])

        annotation.reset_to_sam3()

        self.assertEqual(annotation.source, AnnotationSource.SAM3)
        self.assertEqual(annotation.box_xyxy, (10.0, 10.0, 40.0, 40.0))
        self.assertEqual(annotation.polygon_xyn, original_polygon)
        self.assertTrue(annotation.segmentation_valid)
        self.assertEqual(annotation.segmentation_source, "sam3_original")

    def test_refined_annotation_exports_segmentation(self):
        project = ProjectState(
            input_path="images",
            prompts=["car"],
            images=[ImageRecord("images/a.jpg", 0, width=100, height=100)],
        )
        project.get_image(0).annotations.append(
            Annotation(
                0,
                "car",
                (20, 20, 60, 60),
                source=AnnotationSource.SAM3_REFINED,
                polygon_xyn=[[0.2, 0.2], [0.6, 0.2], [0.6, 0.6]],
                segmentation_valid=True,
                segmentation_source="sam3_box_prompt",
            )
        )

        rows = build_segmentation_rows(project)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["yolo_segmentation_line"], "0 0.200000 0.200000 0.600000 0.200000 0.600000 0.600000")

    def test_segmentation_status_distinguishes_valid_stale_none_and_invalid(self):
        valid = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            source=AnnotationSource.SAM3,
            polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
        )
        stale = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            source=AnnotationSource.SAM3,
            polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
        )
        stale.edit_box((12, 12, 42, 42), 100, 100)
        none = Annotation(0, "car", (10, 10, 40, 40), source=AnnotationSource.MANUAL)
        invalid = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            source=AnnotationSource.SAM3,
            polygon_xyn=[[0.1, 0.1], [0.4, 0.1]],
        )

        self.assertEqual(segmentation_status(valid), "valid")
        self.assertEqual(segmentation_status(stale), "stale")
        self.assertEqual(segmentation_status(none), "none")
        self.assertEqual(segmentation_status(invalid), "invalid")

    def test_manual_and_imported_annotations_can_become_segmentation_valid_after_resegment(self):
        manual = Annotation(0, "car", (10, 10, 40, 40), source=AnnotationSource.MANUAL)
        imported = Annotation(0, "car", (50, 50, 80, 80), source=AnnotationSource.IMPORTED)

        manual.apply_sam3_box_prompt_segmentation([[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]])
        imported.apply_sam3_box_prompt_segmentation([[0.5, 0.5], [0.8, 0.5], [0.8, 0.8]])

        self.assertEqual(manual.source, AnnotationSource.SAM3_REFINED)
        self.assertEqual(imported.source, AnnotationSource.SAM3_REFINED)
        self.assertTrue(has_valid_segmentation(manual))
        self.assertTrue(has_valid_segmentation(imported))

    def test_invalid_resegment_polygon_does_not_mutate_annotation(self):
        annotation = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            source=AnnotationSource.EDITED,
            polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
            segmentation_valid=False,
        )
        before = annotation.to_dict()

        with self.assertRaises(ValueError):
            annotation.apply_sam3_box_prompt_segmentation([[0.1, 0.1], [0.4, 0.1]])

        self.assertEqual(annotation.to_dict(), before)

    def test_box_prompt_helper_chooses_highest_confidence_valid_polygon(self):
        class MultiBoxes:
            conf = [0.2, 0.9]

        class MultiMasks:
            xyn = [
                [(0.1, 0.1), (0.2, 0.1)],
                [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7)],
            ]

        class MultiResult:
            boxes = MultiBoxes()
            masks = MultiMasks()

        polygon, confidence = best_box_prompt_segmentation([MultiResult()])

        self.assertEqual(polygon, [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7]])
        self.assertEqual(confidence, 0.9)

    def test_box_prompt_helper_rejects_no_valid_polygon(self):
        class EmptyMasks:
            xyn = [[(0.1, 0.1), (0.2, 0.1)]]

        class EmptyResult:
            boxes = None
            masks = EmptyMasks()

        with self.assertRaises(ValueError):
            best_box_prompt_segmentation([EmptyResult()])

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

    def test_export_reports_skipped_segmentation_reasons(self):
        project = ProjectState(
            input_path="images",
            prompts=["car"],
            images=[ImageRecord("images/a.jpg", 0, width=100, height=100)],
        )
        valid = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            source=AnnotationSource.SAM3,
            polygon_xyn=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
        )
        stale = Annotation(
            0,
            "car",
            (20, 20, 60, 60),
            source=AnnotationSource.SAM3,
            polygon_xyn=[[0.2, 0.2], [0.6, 0.2], [0.6, 0.6]],
        )
        stale.edit_box((22, 22, 62, 62), 100, 100)
        none = Annotation(0, "car", (50, 50, 80, 80), source=AnnotationSource.MANUAL)
        invalid = Annotation(
            0,
            "car",
            (5, 5, 20, 20),
            source=AnnotationSource.SAM3,
            polygon_xyn=[[0.05, 0.05], [0.2, 0.2]],
        )
        project.get_image(0).annotations.extend([valid, stale, none, invalid])

        skipped_rows = build_skipped_segmentation_rows(project)

        self.assertEqual([row["reason"] for row in skipped_rows], [
            "segmentation stale after bbox/class edit",
            "no polygon",
            "polygon has too few points",
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            result = export_corrected_detection(project, Path(temp_dir))
            report_path = result["segmentation_skipped_report"]

            self.assertEqual(len(result["segmentation_rows"]), 1)
            self.assertEqual(len(result["skipped_segmentation_rows"]), 3)
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["total_skipped_segmentations"], 3)
            self.assertEqual(report["skipped_segmentations"][0]["annotation_id"], stale.id)


if __name__ == "__main__":
    unittest.main()
