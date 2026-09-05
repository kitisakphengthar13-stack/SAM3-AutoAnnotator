import tempfile
import unittest
from pathlib import Path

from services.export_rows import build_box_rows
from domain import Annotation, AnnotationSource, ImageRecord, ImageStatus, ProjectState
from storage.yolo_importer import (
    annotations_from_yolo_file,
    class_name_for_id,
    import_yolo_detection_labels,
    parse_yolo_detection_line,
    yolo_xywhn_to_xyxy,
)


class YoloImporterTests(unittest.TestCase):
    def test_yolo_xywhn_to_xyxy_conversion(self):
        self.assertEqual(
            yolo_xywhn_to_xyxy(0.5, 0.5, 0.25, 0.5, 200, 100),
            (75.0, 25.0, 125.0, 75.0),
        )

    def test_valid_and_invalid_yolo_lines(self):
        annotation = parse_yolo_detection_line("1 0.5 0.5 0.2 0.4", 100, 200, ["car", "person"])

        self.assertEqual(annotation.class_id, 1)
        self.assertEqual(annotation.class_name, "person")
        self.assertEqual(annotation.source, AnnotationSource.IMPORTED)
        self.assertEqual(annotation.box_xyxy, (40.0, 60.0, 60.0, 140.0))

        with self.assertRaises(ValueError):
            parse_yolo_detection_line("1 0.5 0.5 0.2", 100, 200, ["car"])

    def test_unknown_class_id_maps_to_class_name(self):
        self.assertEqual(class_name_for_id(4, ["car"]), "class_4")

    def test_import_extends_project_prompts_and_preserves_yolo_class_ids(self):
        project = ProjectState(
            input_path="images",
            prompts=["car", "class_2"],
            images=[ImageRecord("images/labeled.jpg", 0, width=100, height=100)],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            label_dir = Path(temp_dir)
            (label_dir / "labeled.txt").write_text(
                "4 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
            summary = import_yolo_detection_labels(project, label_dir)

        annotation = project.images[0].active_annotations[0]
        self.assertEqual(annotation.class_id, 4)
        self.assertEqual(annotation.class_name, project.prompts[4])
        self.assertEqual(
            project.prompts,
            ["car", "class_2", "class_2_2", "class_3", "class_4"],
        )
        self.assertEqual(summary.added_classes, 3)

    def test_file_parser_skips_invalid_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            label_path = Path(temp_dir) / "image.txt"
            label_path.write_text("0 0.5 0.5 0.2 0.2\nbad line\n", encoding="utf-8")

            annotations, invalid_lines = annotations_from_yolo_file(label_path, 100, 100, ["object"])

        self.assertEqual(len(annotations), 1)
        self.assertEqual(invalid_lines, 1)

    def test_missing_empty_imported_and_skipped_images(self):
        project = ProjectState(
            input_path="images",
            prompts=["car"],
            images=[
                ImageRecord("images/missing.jpg", 0, width=100, height=100),
                ImageRecord("images/empty.jpg", 1, width=100, height=100),
                ImageRecord("images/labeled.jpg", 2, width=100, height=100),
                ImageRecord("images/edited.jpg", 3, width=100, height=100, status=ImageStatus.EDITED),
                ImageRecord("images/reviewed.jpg", 4, width=100, height=100, status=ImageStatus.REVIEWED),
            ],
        )
        project.get_image(3).annotations = [Annotation(0, "car", (1, 1, 10, 10))]
        project.get_image(4).annotations = [Annotation(0, "car", (2, 2, 20, 20))]

        with tempfile.TemporaryDirectory() as temp_dir:
            label_dir = Path(temp_dir)
            (label_dir / "empty.txt").write_text("", encoding="utf-8")
            (label_dir / "labeled.txt").write_text("0 0.5 0.5 0.2 0.2\nbad\n", encoding="utf-8")

            summary = import_yolo_detection_labels(project, label_dir)

        self.assertEqual(project.get_image(0).status, ImageStatus.NOT_PREDICTED)
        self.assertEqual(project.get_image(1).status, ImageStatus.NO_DETECTION)
        self.assertEqual(project.get_image(2).status, ImageStatus.PREDICTED)
        self.assertEqual(project.get_image(3).status, ImageStatus.EDITED)
        self.assertEqual(project.get_image(4).status, ImageStatus.REVIEWED)
        self.assertEqual(len(project.get_image(2).active_annotations), 1)
        self.assertEqual(summary.missing_label_files, 1)
        self.assertEqual(summary.no_detection_images, 1)
        self.assertEqual(summary.skipped_images, 2)
        self.assertEqual(summary.imported_boxes, 1)
        self.assertEqual(summary.invalid_lines, 1)

    def test_imported_annotation_becomes_edited_when_changed_and_exports(self):
        project = ProjectState(
            input_path="images",
            prompts=["car"],
            images=[ImageRecord("images/labeled.jpg", 0, width=100, height=100)],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            label_dir = Path(temp_dir)
            (label_dir / "labeled.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            import_yolo_detection_labels(project, label_dir)

        annotation = project.get_image(0).active_annotations[0]
        self.assertEqual(annotation.source, AnnotationSource.IMPORTED)
        annotation.edit_box((10, 10, 30, 30), 100, 100)
        self.assertEqual(annotation.source, AnnotationSource.EDITED)

        rows = build_box_rows(project)
        self.assertEqual(rows[0]["x1"], "10.000000")
        self.assertEqual(rows[0]["width_norm"], "0.200000")


if __name__ == "__main__":
    unittest.main()
