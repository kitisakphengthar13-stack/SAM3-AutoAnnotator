import unittest

from domain.annotation import Annotation, AnnotationSource


class AnnotationStateValidationTests(unittest.TestCase):
    def base_state(self):
        return {
            "id": "ann-1",
            "class_id": 0,
            "class_name": "car",
            "box_xyxy": [10, 20, 50, 60],
            "source": "manual",
        }

    def test_string_booleans_are_rejected_in_loaded_state(self):
        for field in ("deleted", "segmentation_valid"):
            state = self.base_state()
            state[field] = "false"
            with self.subTest(field=field):
                with self.assertRaises(TypeError):
                    Annotation.from_dict(state)

    def test_malformed_loaded_polygons_are_rejected(self):
        for polygon in (
            [[0.1, 0.1], [0.2, 0.2]],
            [[0.1, 0.1], [1.2, 0.2], [0.2, 0.8]],
            [[0.1, 0.1], [float("nan"), 0.2], [0.2, 0.8]],
        ):
            state = self.base_state()
            state["polygon_xyn"] = polygon
            with self.subTest(polygon=polygon):
                with self.assertRaises(ValueError):
                    Annotation.from_dict(state)

    def test_empty_runtime_polygon_remains_box_only_annotation(self):
        annotation = Annotation(
            class_id=0,
            class_name="car",
            box_xyxy=(10, 20, 50, 60),
            source=AnnotationSource.SAM3,
            polygon_xyn=[],
        )
        self.assertEqual(annotation.polygon_xyn, [])
        self.assertEqual(annotation.original_polygon_xyn, [])
        self.assertFalse(annotation.segmentation_valid)

    def test_valid_segmentation_requires_polygon(self):
        state = self.base_state()
        state["segmentation_valid"] = True
        with self.assertRaises(ValueError):
            Annotation.from_dict(state)

    def test_invalid_class_metadata_is_rejected(self):
        for class_id, class_name in ((-1, "car"), (0, "  ")):
            with self.subTest(class_id=class_id, class_name=class_name):
                with self.assertRaises(ValueError):
                    Annotation(
                        class_id=class_id,
                        class_name=class_name,
                        box_xyxy=(10, 20, 50, 60),
                        source=AnnotationSource.MANUAL,
                    )


if __name__ == "__main__":
    unittest.main()
