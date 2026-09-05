import tempfile
import unittest
from pathlib import Path

from domain import AnnotationSource
from sam3.predictor_cache import PredictorCache
from services.prediction_service import PredictionService


class FakeBoxes:
    def __init__(self):
        self.xyxy = [(10, 20, 50, 80)]
        self.cls = [0]
        self.conf = [0.85]


class FakeMasks:
    def __init__(self):
        self.xyn = [[(0.1, 0.2), (0.5, 0.2), (0.5, 0.8)]]
        self.data = None


class FakeResult:
    orig_shape = (100, 120)

    def __init__(self):
        self.boxes = FakeBoxes()
        self.masks = FakeMasks()


class FakePredictor:
    def __init__(self, result):
        self.result = result
        self.image_paths = []
        self.calls = []

    def set_image(self, image_path):
        self.image_paths.append(image_path)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


class PredictionServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.image_path = root / "image.jpg"
        self.model_path = root / "sam3.pt"
        self.image_path.write_bytes(b"fake image")
        self.model_path.write_bytes(b"fake weights")
        self.predictor = FakePredictor(FakeResult())
        self.factory_calls = []

        def factory(model_path, conf, half):
            self.factory_calls.append((model_path, conf, half))
            return self.predictor

        self.service = PredictionService(PredictorCache(factory=factory))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_whole_image_prediction_maps_result_and_reuses_predictor(self):
        first = self.service.predict_image(
            image_path=self.image_path,
            model_path=self.model_path,
            prompts=["car"],
            confidence=0.5,
            half=True,
        )
        second = self.service.predict_image(
            image_path=self.image_path,
            model_path=self.model_path,
            prompts=["car"],
            confidence=0.5,
            half=True,
        )

        self.assertFalse(first.reused_predictor)
        self.assertTrue(second.reused_predictor)
        self.assertEqual((first.width, first.height), (120, 100))
        self.assertEqual(len(first.annotations), 1)
        self.assertEqual(first.annotations[0].source, AnnotationSource.SAM3)
        self.assertEqual(first.annotations[0].class_name, "car")
        self.assertEqual(len(self.factory_calls), 1)
        self.assertEqual(self.predictor.calls[0], {"text": ["car"]})

    def test_box_prompt_uses_xyxy_and_selects_valid_polygon(self):
        result = self.service.segment_box(
            image_path=self.image_path,
            model_path=self.model_path,
            box_xyxy=(5, 6, 70, 80),
            class_name="car",
            confidence=0.5,
            half=False,
        )

        self.assertEqual(result.box_xyxy, (5.0, 6.0, 70.0, 80.0))
        self.assertEqual(
            result.polygon_xyn,
            [[0.1, 0.2], [0.5, 0.2], [0.5, 0.8]],
        )
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(
            self.predictor.calls[-1],
            {"bboxes": [(5.0, 6.0, 70.0, 80.0)], "text": ["car"]},
        )

    def test_input_validation_happens_before_predictor_creation(self):
        with self.assertRaises(TypeError):
            self.service.predict_image(
                image_path=self.image_path,
                model_path=self.model_path,
                prompts="car",
                confidence=0.5,
            )
        with self.assertRaises(ValueError):
            self.service.predict_image(
                image_path=self.image_path,
                model_path=self.model_path,
                prompts=["car"],
                confidence=0,
            )
        with self.assertRaises(TypeError):
            self.service.predict_image(
                image_path=self.image_path,
                model_path=self.model_path,
                prompts=[None],
                confidence=0.5,
            )
        with self.assertRaises(FileNotFoundError):
            self.service.predict_image(
                image_path=self.image_path.with_name("missing.jpg"),
                model_path=self.model_path,
                prompts=["car"],
                confidence=0.5,
            )

        self.assertEqual(self.factory_calls, [])

    def test_empty_predictor_result_has_a_clear_error(self):
        class EmptyPredictor(FakePredictor):
            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return []

        service = PredictionService(
            PredictorCache(factory=lambda **_: EmptyPredictor(None))
        )

        with self.assertRaisesRegex(ValueError, "no prediction results"):
            service.predict_image(
                image_path=self.image_path,
                model_path=self.model_path,
                prompts=["car"],
                confidence=0.5,
            )


if __name__ == "__main__":
    unittest.main()
