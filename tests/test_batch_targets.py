import unittest

from sam3_auto_annotator.core import ImageRecord, ImageStatus, ProjectState
from sam3_auto_annotator.services.project_service import remaining_prediction_targets


class BatchTargetSelectionTests(unittest.TestCase):
    def test_remaining_targets_include_not_predicted_and_error_only(self):
        project = ProjectState(
            input_path="images",
            prompts=["object"],
            images=[
                ImageRecord("images/not_predicted.jpg", 0, status=ImageStatus.NOT_PREDICTED),
                ImageRecord("images/error.jpg", 1, status=ImageStatus.ERROR),
                ImageRecord("images/predicted.jpg", 2, status=ImageStatus.PREDICTED),
                ImageRecord("images/edited.jpg", 3, status=ImageStatus.EDITED),
                ImageRecord("images/reviewed.jpg", 4, status=ImageStatus.REVIEWED),
                ImageRecord("images/no_detection.jpg", 5, status=ImageStatus.NO_DETECTION),
            ],
        )

        targets = remaining_prediction_targets(project)

        self.assertEqual(
            [target.image_name for target in targets],
            ["not_predicted.jpg", "error.jpg"],
        )


if __name__ == "__main__":
    unittest.main()
