import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from sam3_auto_annotator.core import AnnotationSource, ImageStatus
from sam3_auto_annotator.services.annotation_service import (
    add_manual_annotation,
    edit_annotation_box,
    mark_image_reviewed,
)
from sam3_auto_annotator.services.project_service import (
    create_project,
    ensure_image_sizes,
    export_project,
    load_state,
    parse_prompts,
    save_state_to_output,
)


class ProjectIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = Path(__file__).resolve().parents[1] / "images_test"
        cls.fixture_paths = sorted(cls.fixture_dir.glob("car_*.jpg"))
        if len(cls.fixture_paths) < 3:
            raise unittest.SkipTest(
                "The backend integration test needs images_test/car_1.jpg through car_3.jpg."
            )

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_dir = self.root / "images"
        self.image_dir.mkdir()
        self.image_paths = []
        for source in self.fixture_paths[:3]:
            destination = self.image_dir / source.name
            shutil.copy2(source, destination)
            self.image_paths.append(destination)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _new_project(self):
        project = create_project(
            input_path=self.image_dir,
            prompts=parse_prompts(" car\ncar "),
            model_path=self.root / "sam3-model.pt",
            project_name="cars",
            confidence=0.61,
            half=False,
        )
        return ensure_image_sizes(project)

    def test_project_lifecycle_exports_csv_yolo_and_empty_labels(self):
        project = self._new_project()

        self.assertEqual(project.prompts, ["car"])
        self.assertEqual([Path(item.image_path) for item in project.images], self.image_paths)
        for record, image_path in zip(project.images, self.image_paths, strict=True):
            with Image.open(image_path) as image:
                self.assertEqual((record.width, record.height), image.size)

        annotated = project.images[0]
        width, height = annotated.width, annotated.height
        annotation = add_manual_annotation(
            annotated,
            class_id=0,
            class_name="car",
            box_xyxy=(width * 0.10, height * 0.15, width * 0.85, height * 0.90),
        )
        edited_box = (width * 0.15, height * 0.20, width * 0.75, height * 0.80)
        edit_annotation_box(annotated, annotation.id, edited_box)
        mark_image_reviewed(annotated)

        intentionally_empty = project.images[1]
        mark_image_reviewed(intentionally_empty)
        project.images[2].replace_sam3_drafts([])

        self.assertEqual(annotation.source, AnnotationSource.EDITED)
        self.assertEqual(annotation.box_xyxy, tuple(float(value) for value in edited_box))
        self.assertEqual(annotated.status, ImageStatus.REVIEWED)
        self.assertEqual(intentionally_empty.status, ImageStatus.REVIEWED)
        self.assertEqual(project.images[2].status, ImageStatus.NO_DETECTION)

        project_dir = self.root / "saved-project"
        state_path = save_state_to_output(project, project_dir)
        self.assertEqual(state_path, project_dir / "annotation_state.json")
        self.assertTrue(state_path.is_file())
        self.assertEqual(list(project_dir.glob(".annotation_state.json.*.tmp")), [])

        restored = load_state(state_path)
        self.assertEqual(restored.project_name, "saved-project")
        self.assertEqual(restored.prompts, ["car"])
        self.assertEqual(restored.confidence, 0.61)
        self.assertFalse(restored.half)
        self.assertEqual(
            [record.status for record in restored.images],
            [ImageStatus.REVIEWED, ImageStatus.REVIEWED, ImageStatus.NO_DETECTION],
        )
        restored_annotation = restored.images[0].active_annotations[0]
        self.assertEqual(restored_annotation.id, annotation.id)
        self.assertEqual(restored_annotation.box_xyxy, annotation.box_xyxy)

        export_dir = self.root / "export"
        exported = export_project(restored, export_dir)

        with exported["box_csv"].open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["image_name"], self.image_paths[0].name)
        self.assertEqual(rows[0]["class_id"], "0")
        self.assertEqual(rows[0]["class_name"], "car")

        detection_files = {
            path.name: path.read_text(encoding="utf-8")
            for path in exported["yolo_detection_dir"].glob("*.txt")
        }
        segmentation_files = {
            path.name: path.read_text(encoding="utf-8")
            for path in exported["yolo_segmentation_dir"].glob("*.txt")
        }
        expected_names = {f"{path.stem}.txt" for path in self.image_paths}
        self.assertEqual(set(detection_files), expected_names)
        self.assertEqual(set(segmentation_files), expected_names)
        self.assertTrue(detection_files[f"{self.image_paths[0].stem}.txt"].startswith("0 "))
        self.assertEqual(detection_files[f"{self.image_paths[1].stem}.txt"], "")
        self.assertEqual(detection_files[f"{self.image_paths[2].stem}.txt"], "")
        self.assertTrue(all(text == "" for text in segmentation_files.values()))

        self.assertEqual(len(exported["rows"]), 1)
        self.assertEqual(exported["segmentation_rows"], [])
        self.assertEqual(len(exported["skipped_segmentation_rows"]), 1)
        self.assertTrue(exported["segmentation_skipped_report"].is_file())
        with exported["run_summary"].open(encoding="utf-8") as summary_file:
            summary = json.load(summary_file)
        self.assertEqual(summary["images_processed"], 3)
        self.assertEqual(summary["images_not_predicted"], 0)
        self.assertEqual(summary["total_detections"], 1)

    def test_failed_atomic_replace_preserves_previous_state_and_removes_temp_file(self):
        project = self._new_project()
        project_dir = self.root / "saved-project"
        state_path = save_state_to_output(project, project_dir)
        previous_bytes = state_path.read_bytes()

        project.confidence = 0.25
        with patch(
            "sam3_auto_annotator.storage.project_store.os.replace",
            side_effect=OSError("simulated replace failure"),
        ):
            with self.assertRaisesRegex(OSError, "simulated replace failure"):
                save_state_to_output(project, project_dir)

        self.assertEqual(state_path.read_bytes(), previous_bytes)
        self.assertEqual(list(project_dir.glob(".annotation_state.json.*.tmp")), [])
        self.assertEqual(load_state(state_path).confidence, 0.61)


if __name__ == "__main__":
    unittest.main()
