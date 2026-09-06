import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from domain import Annotation, AnnotationSource, ImageRecord, ImageStatus, ProjectState
from domain.geometry import validate_xyxy
from sam3.result_mapper import best_box_prompt_segmentation
from services.export_service import export_corrected_detection
from services.project_service import save_state_to_output, verify_source_image_sizes
from storage.image_catalog import sanitize_name
from storage.yolo_importer import import_yolo_detection_labels, parse_yolo_detection_line


class AuditRegressionTests(unittest.TestCase):
    def test_non_finite_geometry_and_yolo_values_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_xyxy((0, 0, math.nan, 10))
        with self.assertRaises(ValueError):
            parse_yolo_detection_line("0 nan 0.5 0.2 0.2", 100, 100, ["car"])
        with self.assertRaises(ValueError):
            parse_yolo_detection_line("0 1.2 0.5 0.2 0.2", 100, 100, ["car"])
        with self.assertRaises(ValueError):
            parse_yolo_detection_line("10001 0.5 0.5 0.2 0.2", 100, 100, ["car"])

    def test_missing_yolo_label_leaves_existing_image_state_untouched(self):
        annotation = Annotation(0, "car", (10, 10, 20, 20), id="existing")
        project = ProjectState(
            input_path="images",
            prompts=["car"],
            images=[
                ImageRecord(
                    "images/a.jpg",
                    0,
                    width=100,
                    height=100,
                    status=ImageStatus.PREDICTED,
                    annotations=[annotation],
                )
            ],
        )
        with tempfile.TemporaryDirectory() as temp:
            summary = import_yolo_detection_labels(project, temp)
        self.assertEqual(summary.missing_label_files, 1)
        self.assertEqual(project.images[0].status, ImageStatus.PREDICTED)
        self.assertEqual(project.images[0].active_annotations[0].id, "existing")

    def test_malformed_only_yolo_file_does_not_mean_no_detection(self):
        annotation = Annotation(0, "car", (10, 10, 20, 20), id="existing")
        project = ProjectState(
            input_path="images",
            prompts=["car"],
            images=[
                ImageRecord(
                    "images/a.jpg",
                    0,
                    width=100,
                    height=100,
                    status=ImageStatus.PREDICTED,
                    annotations=[annotation],
                )
            ],
        )
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "a.txt").write_text("bad row\n", encoding="utf-8")
            summary = import_yolo_detection_labels(project, temp)
        self.assertEqual(summary.invalid_lines, 1)
        self.assertEqual(project.images[0].status, ImageStatus.PREDICTED)
        self.assertEqual(project.images[0].active_annotations[0].id, "existing")

    def test_yolo_import_rolls_back_all_images_when_later_file_fails(self):
        project = ProjectState(
            input_path="images",
            prompts=["car"],
            images=[
                ImageRecord("images/a.jpg", 0, width=100, height=100),
                ImageRecord("images/b.jpg", 1, width=100, height=100),
            ],
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (root / "b.txt").write_bytes(b"\xff")
            with self.assertRaises(UnicodeDecodeError):
                import_yolo_detection_labels(project, root)
        self.assertEqual(project.images[0].status, ImageStatus.NOT_PREDICTED)
        self.assertEqual(project.images[0].annotations, [])
        self.assertEqual(project.images[1].status, ImageStatus.NOT_PREDICTED)

    def test_reset_without_original_polygon_discards_refined_polygon(self):
        annotation = Annotation(
            0,
            "car",
            (10, 10, 40, 40),
            source=AnnotationSource.SAM3,
            polygon_xyn=None,
        )
        annotation.edit_box((20, 20, 50, 50), 100, 100)
        annotation.apply_sam3_box_prompt_segmentation(
            [[0.2, 0.2], [0.5, 0.2], [0.5, 0.5]]
        )
        annotation.reset_to_sam3()
        self.assertEqual(annotation.box_xyxy, (10.0, 10.0, 40.0, 40.0))
        self.assertIsNone(annotation.polygon_xyn)
        self.assertFalse(annotation.segmentation_valid)
        self.assertIsNone(annotation.segmentation_source)
        self.assertEqual(annotation.source, AnnotationSource.SAM3)

    def test_error_does_not_demote_existing_review_state(self):
        image = ImageRecord("a.jpg", 0, status=ImageStatus.REVIEWED)
        image.mark_error("display failed")
        self.assertEqual(image.status, ImageStatus.REVIEWED)
        self.assertEqual(image.error_message, "display failed")
        pending = ImageRecord("b.jpg", 1)
        pending.mark_error("inference failed")
        self.assertEqual(pending.status, ImageStatus.ERROR)

    def test_export_replaces_managed_tree_and_removes_stale_report(self):
        project = ProjectState(
            input_path="images",
            prompts=["car"],
            images=[ImageRecord("images/a.jpg", 0, width=100, height=100)],
        )
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            stale_detection = output / "yolo_labels" / "detection" / "old.txt"
            stale_segmentation = output / "yolo_labels" / "segmentation" / "old.txt"
            stale_detection.parent.mkdir(parents=True)
            stale_segmentation.parent.mkdir(parents=True)
            stale_detection.write_text("stale", encoding="utf-8")
            stale_segmentation.write_text("stale", encoding="utf-8")
            (output / "segmentation_skipped_report.json").write_text("{}", encoding="utf-8")

            export_corrected_detection(project, output)

            self.assertFalse(stale_detection.exists())
            self.assertFalse(stale_segmentation.exists())
            self.assertFalse((output / "segmentation_skipped_report.json").exists())
            self.assertTrue((output / "yolo_labels" / "detection" / "a.txt").is_file())

    def test_failed_save_restores_project_name(self):
        project = ProjectState(input_path="images", prompts=[], images=[], project_name="before")
        with tempfile.TemporaryDirectory() as temp:
            with patch(
                "services.project_service.save_project_state",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaises(OSError):
                    save_state_to_output(project, Path(temp) / "after")
        self.assertEqual(project.project_name, "before")

    def test_changed_source_dimensions_block_export_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "a.png"
            Image.new("RGB", (20, 10)).save(image_path)
            project = ProjectState(
                input_path=str(image_path),
                prompts=[],
                images=[ImageRecord(str(image_path), 0, width=10, height=10)],
            )
            with self.assertRaisesRegex(ValueError, "dimensions changed"):
                verify_source_image_sizes(project)

    def test_unicode_project_names_remain_distinct(self):
        first = sanitize_name("ชิ้นงานกล้องหน้า_annotations")
        second = sanitize_name("ชิ้นงานกล้องหลัง_annotations")
        self.assertNotEqual(first, second)
        self.assertIn("ชิ้นงาน", first)

    def test_box_prompt_mapper_prefers_spatial_match_over_remote_confidence(self):
        class Boxes:
            conf = [0.99, 0.60]

        class Masks:
            xyn = [
                [(0.7, 0.7), (0.9, 0.7), (0.9, 0.9)],
                [(0.1, 0.1), (0.25, 0.1), (0.25, 0.25)],
            ]
            data = None

        class Result:
            orig_shape = (100, 100)
            boxes = Boxes()
            masks = Masks()

        polygon, confidence = best_box_prompt_segmentation(
            [Result()], requested_box=(0, 0, 30, 30)
        )
        self.assertEqual(polygon, [[0.1, 0.1], [0.25, 0.1], [0.25, 0.25]])
        self.assertEqual(confidence, 0.60)

    def test_duplicate_annotation_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Annotation ids"):
            ProjectState(
                input_path="images",
                prompts=["car"],
                images=[
                    ImageRecord(
                        "images/a.jpg",
                        0,
                        annotations=[Annotation(0, "car", (1, 1, 10, 10), id="same")],
                    ),
                    ImageRecord(
                        "images/b.jpg",
                        1,
                        annotations=[Annotation(0, "car", (2, 2, 12, 12), id="same")],
                    ),
                ],
            )


if __name__ == "__main__":
    unittest.main()
