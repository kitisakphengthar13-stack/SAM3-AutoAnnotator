"""Validated annotation editing operations used by the GUI controller.

Qt widgets should only collect input and present the result.  This module owns
the mutations that turn those inputs into changes to an :class:`ImageRecord`.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isclose

from domain import Annotation, ImageRecord, ImageStatus
from domain.geometry import clip_xyxy, validate_xyxy


def _require_image(image: ImageRecord) -> ImageRecord:
    if not isinstance(image, ImageRecord):
        raise TypeError("image must be an ImageRecord.")
    return image


def _require_annotation(
    image: ImageRecord,
    annotation_id: str,
    *,
    active: bool = True,
) -> Annotation:
    _require_image(image)
    if not isinstance(annotation_id, str) or not annotation_id.strip():
        raise ValueError("annotation_id must be a non-empty string.")

    annotation = image.annotation_by_id(annotation_id)
    if annotation is None:
        raise KeyError(f"No annotation with id {annotation_id!r} exists on this image.")
    if active and not annotation.is_active:
        raise ValueError(f"Annotation {annotation_id!r} has already been deleted.")
    return annotation


def _validated_class(class_id: int, class_name: str) -> tuple[int, str]:
    if not isinstance(class_id, int) or isinstance(class_id, bool):
        raise TypeError("class_id must be a non-negative integer.")
    normalized_id = int(class_id)
    if normalized_id < 0:
        raise ValueError("class_id must be a non-negative integer.")

    if class_name is None:
        raise ValueError("class_name must not be empty.")
    normalized_name = str(class_name).strip()
    if not normalized_name:
        raise ValueError("class_name must not be empty.")
    return normalized_id, normalized_name


def _validated_box(image: ImageRecord, box_xyxy) -> tuple[float, float, float, float]:
    if (image.width is None) != (image.height is None):
        raise ValueError("Image width and height must either both be known or both be unset.")
    # Validate the unmodified shape first so inverted input is never made to
    # look valid by clipping.
    box = validate_xyxy(box_xyxy)
    if image.width is None:
        return box
    return clip_xyxy(box, image.width, image.height)


def add_manual_annotation(
    image: ImageRecord,
    class_id: int,
    class_name: str,
    box_xyxy,
) -> Annotation:
    """Add one manually drawn box and mark the image as edited."""

    image = _require_image(image)
    normalized_id, normalized_name = _validated_class(class_id, class_name)
    box = _validated_box(image, box_xyxy)
    return image.add_manual_annotation(normalized_id, normalized_name, box)


def edit_annotation_box(
    image: ImageRecord,
    annotation_id: str,
    box_xyxy,
) -> Annotation:
    """Replace an active annotation's box, clipping to known image bounds."""

    annotation = _require_annotation(image, annotation_id)
    box = _validated_box(image, box_xyxy)
    if all(
        isclose(old, new, rel_tol=0.0, abs_tol=0.005)
        for old, new in zip(annotation.box_xyxy, box)
    ):
        return annotation
    if image.width is None:
        annotation.edit_box(box)
    else:
        annotation.edit_box(box, image.width, image.height)
    image.mark_edited()
    return annotation


def change_annotation_class(
    image: ImageRecord,
    annotation_id: str,
    class_id: int,
    class_name: str,
) -> Annotation:
    """Apply a validated class to an active annotation."""

    annotation = _require_annotation(image, annotation_id)
    normalized_id, normalized_name = _validated_class(class_id, class_name)
    if (
        annotation.class_id == normalized_id
        and annotation.class_name == normalized_name
    ):
        return annotation
    annotation.change_class(normalized_id, normalized_name)
    image.mark_edited()
    return annotation


def delete_annotation(image: ImageRecord, annotation_id: str) -> Annotation:
    """Soft-delete an active annotation so project history stays recoverable."""

    annotation = _require_annotation(image, annotation_id)
    annotation.mark_deleted()
    image.mark_edited()
    return annotation


def reset_annotation_to_sam3(image: ImageRecord, annotation_id: str) -> Annotation:
    """Restore the geometry and class captured from the original SAM3 result."""

    annotation = _require_annotation(image, annotation_id)
    if not annotation.is_modified_from_sam3:
        return annotation
    annotation.reset_to_sam3()
    image.mark_edited()
    return annotation


def apply_box_segmentation(
    image: ImageRecord,
    annotation_id: str,
    polygon_xyn: Sequence[Sequence[float]],
    confidence: float | None = None,
) -> Annotation:
    """Attach a normalized polygon returned by SAM3 box-prompt inference."""

    annotation = _require_annotation(image, annotation_id)
    points: list[list[float]] = []
    for point in polygon_xyn or ():
        if len(point) != 2:
            raise ValueError("Each polygon point must contain exactly x and y.")
        x, y = (float(value) for value in point)
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError("Normalized polygon coordinates must be between 0 and 1.")
        points.append([x, y])
    if len(points) < 3:
        raise ValueError("Re-segmentation requires a polygon with at least three points.")

    normalized_confidence = None if confidence is None else float(confidence)
    if normalized_confidence is not None and not 0.0 <= normalized_confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1.")

    annotation.apply_sam3_box_prompt_segmentation(points, normalized_confidence)
    image.mark_edited()
    return annotation


def mark_image_reviewed(image: ImageRecord) -> ImageRecord:
    """Mark an image as reviewed, including intentional empty images."""

    image = _require_image(image)
    if image.status == ImageStatus.REVIEWED:
        return image
    image.mark_reviewed()
    return image


__all__ = [
    "add_manual_annotation",
    "apply_box_segmentation",
    "change_annotation_class",
    "delete_annotation",
    "edit_annotation_box",
    "mark_image_reviewed",
    "reset_annotation_to_sam3",
]
