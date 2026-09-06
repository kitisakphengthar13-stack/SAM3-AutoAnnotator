import unittest

from domain.annotation import Annotation, AnnotationSource
from domain.segmentation import build_yolo_segmentation_line, has_valid_segmentation, validate_polygon_xyn


class PolygonGeometryIntegrityTests(unittest.TestCase):
    def test_collinear_polygon_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero area"):
            validate_polygon_xyn([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])

    def test_self_intersecting_polygon_is_rejected(self):
        bow_tie = [[0.1, 0.1], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]]
        with self.assertRaisesRegex(ValueError, "self-intersects"):
            validate_polygon_xyn(bow_tie)
        with self.assertRaisesRegex(ValueError, "self-intersects"):
            build_yolo_segmentation_line(0, bow_tie)

    def test_duplicate_closing_point_is_normalized(self):
        points = [[0.1, 0.1], [0.8, 0.1], [0.8, 0.8], [0.1, 0.1]]
        self.assertEqual(len(validate_polygon_xyn(points)), 3)

    def test_runtime_annotation_reports_degenerate_polygon_invalid(self):
        annotation = Annotation(
            class_id=0,
            class_name="car",
            box_xyxy=(10, 10, 50, 50),
            source=AnnotationSource.SAM3,
            polygon_xyn=[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
            segmentation_valid=True,
        )
        self.assertFalse(has_valid_segmentation(annotation))


if __name__ == "__main__":
    unittest.main()
