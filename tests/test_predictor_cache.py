import tempfile
import unittest
from pathlib import Path

from sam3.predictor_cache import PredictorCache, PredictorCacheKey


class PredictorCacheTests(unittest.TestCase):
    def test_cache_key_normalizes_model_path_and_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pt"
            model_path.write_text("fake", encoding="utf-8")

            key = PredictorCacheKey.from_settings(model_path, 0.5000001, True)

            self.assertEqual(key.model_path, str(model_path.resolve()))
            self.assertEqual(key.conf, 0.5)
            self.assertTrue(key.half)

    def test_reuses_predictor_for_same_settings(self):
        calls = []

        def factory(model_path, conf, half):
            predictor = object()
            calls.append((model_path, conf, half, predictor))
            return predictor

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pt"
            model_path.write_text("fake", encoding="utf-8")
            cache = PredictorCache(factory=factory)

            first, first_reused = cache.get_predictor(model_path, 0.5, True)
            second, second_reused = cache.get_predictor(model_path, 0.5, True)

        self.assertIs(first, second)
        self.assertFalse(first_reused)
        self.assertTrue(second_reused)
        self.assertEqual(len(calls), 1)

    def test_recreates_predictor_when_relevant_settings_change(self):
        calls = []

        def factory(model_path, conf, half):
            predictor = object()
            calls.append((model_path, conf, half, predictor))
            return predictor

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pt"
            model_path.write_text("fake", encoding="utf-8")
            cache = PredictorCache(factory=factory)

            first, _ = cache.get_predictor(model_path, 0.5, True)
            second, reused = cache.get_predictor(model_path, 0.6, True)

        self.assertIsNot(first, second)
        self.assertFalse(reused)
        self.assertEqual(len(calls), 2)

    def test_failed_creation_does_not_replace_existing_predictor(self):
        good_predictor = object()
        should_fail = False

        def factory(model_path, conf, half):
            if should_fail:
                raise RuntimeError("load failed")
            return good_predictor

        with tempfile.TemporaryDirectory() as temp_dir:
            first_model = Path(temp_dir) / "first.pt"
            second_model = Path(temp_dir) / "second.pt"
            first_model.write_text("fake", encoding="utf-8")
            second_model.write_text("fake", encoding="utf-8")
            cache = PredictorCache(factory=factory)

            first, _ = cache.get_predictor(first_model, 0.5, True)
            should_fail = True
            with self.assertRaises(RuntimeError):
                cache.get_predictor(second_model, 0.5, True)
            cached, reused = cache.get_predictor(first_model, 0.5, True)

        self.assertIs(first, good_predictor)
        self.assertIs(cached, good_predictor)
        self.assertTrue(reused)


if __name__ == "__main__":
    unittest.main()
