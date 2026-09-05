import sys
import types
import unittest
from unittest.mock import patch

from sam3.predictor import create_predictor


class PredictorPrecisionTests(unittest.TestCase):
    def _create_with_fake_ultralytics(self, half):
        captured = []

        class FakeSemanticPredictor:
            def __init__(self, *, overrides):
                self.overrides = overrides
                captured.append(overrides)

        ultralytics = types.ModuleType("ultralytics")
        ultralytics.__path__ = []
        models = types.ModuleType("ultralytics.models")
        models.__path__ = []
        sam = types.ModuleType("ultralytics.models.sam")
        sam.SAM3SemanticPredictor = FakeSemanticPredictor
        ultralytics.models = models
        models.sam = sam

        with patch.dict(
            sys.modules,
            {
                "ultralytics": ultralytics,
                "ultralytics.models": models,
                "ultralytics.models.sam": sam,
            },
        ):
            predictor = create_predictor("models/sam3.pt", conf=0.45, half=half)
        return predictor, captured[0]

    def test_half_precision_uses_current_quantize_16_setting(self):
        predictor, overrides = self._create_with_fake_ultralytics(True)

        self.assertIs(predictor.overrides, overrides)
        self.assertEqual(overrides["quantize"], 16)
        self.assertNotIn("half", overrides)

    def test_full_precision_uses_explicit_quantize_32_setting(self):
        _, overrides = self._create_with_fake_ultralytics(False)

        self.assertEqual(overrides["quantize"], 32)
        self.assertEqual(overrides["task"], "segment")
        self.assertEqual(overrides["mode"], "predict")
        self.assertFalse(overrides["save"])


if __name__ == "__main__":
    unittest.main()
