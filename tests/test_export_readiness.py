import unittest
from types import SimpleNamespace

from services.export_service import evaluate_export_readiness


class ExportReadinessTests(unittest.TestCase):
    def test_readiness_counts_review_incomplete_and_stale_segmentation(self):
        project = SimpleNamespace(
            images=[
                SimpleNamespace(
                    status="reviewed",
                    active_annotations=[SimpleNamespace(segmentation_valid=True)],
                ),
                SimpleNamespace(
                    status="not_predicted",
                    active_annotations=[SimpleNamespace(segmentation_valid=False)],
                ),
                SimpleNamespace(
                    status="error",
                    active_annotations=[],
                ),
            ]
        )

        readiness = evaluate_export_readiness(project)

        self.assertEqual(readiness.total_images, 3)
        self.assertEqual(readiness.reviewed_images, 1)
        self.assertEqual(readiness.needs_review, 2)
        self.assertEqual(readiness.incomplete_images, 2)
        self.assertEqual(readiness.stale_segmentations, 1)
        self.assertTrue(readiness.has_warnings)

    def test_fully_reviewed_project_without_stale_segmentation_is_ready(self):
        project = SimpleNamespace(
            images=[
                SimpleNamespace(
                    status="reviewed",
                    active_annotations=[SimpleNamespace(segmentation_valid=True)],
                )
            ]
        )

        readiness = evaluate_export_readiness(project)

        self.assertFalse(readiness.has_warnings)
        self.assertEqual(readiness.needs_review, 0)
        self.assertEqual(readiness.incomplete_images, 0)
        self.assertEqual(readiness.stale_segmentations, 0)


if __name__ == "__main__":
    unittest.main()
