import unittest

from sam3_auto_annotator.core import (
    Annotation,
    AnnotationSource,
    ImageRecord,
    ImageStatus,
)
from sam3_auto_annotator.services.annotation_service import (
    add_manual_annotation,
    apply_box_segmentation,
    change_annotation_class,
    delete_annotation,
    edit_annotation_box,
    mark_image_reviewed,
    reset_annotation_to_sam3,
)


class AnnotationServiceTests(unittest.TestCase):
    def setUp(self):
        self.original = Annotation(
            class_id=0,
            class_name="car",
            box_xyxy=(10, 10, 60, 70),
            id="sam3-1",
            source=AnnotationSource.SAM3,
            confidence=0.9,
            polygon_xyn=[[0.1, 0.1], [0.6, 0.1], [0.6, 0.7]],
        )
        self.image = ImageRecord(
            image_path="images/example.jpg",
            image_index=0,
            width=100,
            height=80,
            status=ImageStatus.PREDICTED,
            annotations=[self.original],
        )

    def test_add_manual_annotation_clips_before_mutating(self):
        annotation = add_manual_annotation(
            self.image,
            class_id=1,
            class_name=" person ",
            box_xyxy=(-5, 5, 120, 90),
        )

        self.assertEqual(annotation.box_xyxy, (0.0, 5.0, 100.0, 80.0))
        self.assertEqual(annotation.class_name, "person")
        self.assertEqual(annotation.source, AnnotationSource.MANUAL)
        self.assertEqual(self.image.status, ImageStatus.EDITED)

    def test_invalid_manual_box_does_not_append_partial_annotation(self):
        before = list(self.image.annotations)

        with self.assertRaises(ValueError):
            add_manual_annotation(self.image, 1, "person", (10, 10, 10, 20))

        self.assertEqual(self.image.annotations, before)
        self.assertEqual(self.image.status, ImageStatus.PREDICTED)

    def test_edit_box_clips_and_invalidates_segmentation(self):
        annotation = edit_annotation_box(
            self.image,
            "sam3-1",
            (-10, 4, 120, 75),
        )

        self.assertEqual(annotation.box_xyxy, (0.0, 4.0, 100.0, 75.0))
        self.assertEqual(annotation.source, AnnotationSource.EDITED)
        self.assertFalse(annotation.segmentation_valid)
        self.assertEqual(self.image.status, ImageStatus.EDITED)

    def test_change_class_validates_input_and_marks_image_edited(self):
        annotation = change_annotation_class(self.image, "sam3-1", 2, " bicycle ")

        self.assertEqual((annotation.class_id, annotation.class_name), (2, "bicycle"))
        self.assertFalse(annotation.segmentation_valid)
        self.assertEqual(self.image.status, ImageStatus.EDITED)

        with self.assertRaises(ValueError):
            change_annotation_class(self.image, "sam3-1", -1, "invalid")
        with self.assertRaises(TypeError):
            change_annotation_class(self.image, "sam3-1", 1.0, "invalid")

    def test_unchanged_box_and_class_are_noops(self):
        before = self.original.to_dict()

        edit_annotation_box(self.image, "sam3-1", self.original.box_xyxy)
        change_annotation_class(
            self.image,
            "sam3-1",
            self.original.class_id,
            self.original.class_name,
        )

        self.assertEqual(self.original.to_dict(), before)
        self.assertEqual(self.image.status, ImageStatus.PREDICTED)
        self.assertTrue(self.original.segmentation_valid)

    def test_delete_is_soft_delete_and_rejects_a_second_delete(self):
        annotation = delete_annotation(self.image, "sam3-1")

        self.assertTrue(annotation.deleted)
        self.assertEqual(self.image.active_annotations, [])
        self.assertEqual(len(self.image.annotations), 1)
        with self.assertRaises(ValueError):
            delete_annotation(self.image, "sam3-1")

    def test_reset_restores_original_sam3_geometry_and_class(self):
        edit_annotation_box(self.image, "sam3-1", (20, 20, 50, 50))
        change_annotation_class(self.image, "sam3-1", 3, "truck")

        annotation = reset_annotation_to_sam3(self.image, "sam3-1")

        self.assertEqual(annotation.box_xyxy, (10.0, 10.0, 60.0, 70.0))
        self.assertEqual((annotation.class_id, annotation.class_name), (0, "car"))
        self.assertEqual(annotation.source, AnnotationSource.SAM3)
        self.assertTrue(annotation.segmentation_valid)
        self.assertEqual(self.image.status, ImageStatus.EDITED)

    def test_reset_and_review_are_noops_when_already_in_target_state(self):
        before = self.original.to_dict()
        reset_annotation_to_sam3(self.image, "sam3-1")
        self.assertEqual(self.original.to_dict(), before)
        self.assertEqual(self.image.status, ImageStatus.PREDICTED)

        self.image.mark_reviewed()
        mark_image_reviewed(self.image)
        self.assertEqual(self.image.status, ImageStatus.REVIEWED)

    def test_apply_box_segmentation_validates_normalized_polygon(self):
        edit_annotation_box(self.image, "sam3-1", (20, 20, 50, 50))

        annotation = apply_box_segmentation(
            self.image,
            "sam3-1",
            [[0.2, 0.2], [0.5, 0.2], [0.5, 0.5]],
            confidence=0.75,
        )

        self.assertEqual(annotation.source, AnnotationSource.SAM3_REFINED)
        self.assertTrue(annotation.segmentation_valid)
        self.assertEqual(annotation.confidence, 0.75)
        with self.assertRaises(ValueError):
            apply_box_segmentation(
                self.image,
                "sam3-1",
                [[-0.1, 0.2], [0.5, 0.2], [0.5, 0.5]],
            )

    def test_unknown_annotation_id_and_partial_image_size_are_rejected(self):
        with self.assertRaises(KeyError):
            edit_annotation_box(self.image, "missing", (1, 1, 2, 2))

        partial = ImageRecord("images/partial.jpg", 1, width=100)
        with self.assertRaises(ValueError):
            add_manual_annotation(partial, 0, "car", (1, 1, 2, 2))

    def test_review_allows_an_intentionally_empty_image(self):
        empty = ImageRecord("images/empty.jpg", 1)

        returned = mark_image_reviewed(empty)

        self.assertIs(returned, empty)
        self.assertEqual(empty.status, ImageStatus.REVIEWED)


if __name__ == "__main__":
    unittest.main()
