from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Optional
from uuid import uuid4

from domain.geometry import clip_xyxy, validate_xyxy
from domain.segmentation import validate_polygon_xyn


class AnnotationSource(str, Enum):
    SAM3 = "sam3"
    SAM3_REFINED = "sam3_refined"
    EDITED = "edited"
    MANUAL = "manual"
    IMPORTED = "imported"


def _strict_bool(value, field_name, *, allow_none=False):
    if value is None and allow_none:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean.")
    return value


def _optional_polygon(polygon_xyn):
    if polygon_xyn is None:
        return None
    try:
        if len(polygon_xyn) == 0:
            return None
    except TypeError:
        pass
    return validate_polygon_xyn(polygon_xyn)


def _class_metadata(class_id, class_name, *, prefix="Annotation"):
    normalized_id = int(class_id)
    normalized_name = str(class_name).strip()
    if normalized_id < 0:
        raise ValueError(f"{prefix} class id must be non-negative.")
    if not normalized_name:
        raise ValueError(f"{prefix} class name must not be empty.")
    return normalized_id, normalized_name


@dataclass
class Annotation:
    class_id: int
    class_name: str
    box_xyxy: tuple[float, float, float, float]
    id: str = field(default_factory=lambda: str(uuid4()))
    source: AnnotationSource = AnnotationSource.SAM3
    confidence: Optional[float] = None
    polygon_xyn: Optional[list[list[float]]] = None
    segmentation_valid: Optional[bool] = None
    segmentation_source: Optional[str] = None
    original_box_xyxy: Optional[tuple[float, float, float, float]] = None
    original_polygon_xyn: Optional[list[list[float]]] = None
    original_class_id: Optional[int] = None
    original_class_name: Optional[str] = None
    deleted: bool = False

    def __post_init__(self):
        self.id = str(self.id).strip()
        if not self.id:
            raise ValueError("Annotation id must not be empty.")
        self.source = (
            self.source
            if isinstance(self.source, AnnotationSource)
            else AnnotationSource(self.source)
        )
        self.box_xyxy = validate_xyxy(self.box_xyxy)
        if self.original_box_xyxy is not None:
            self.original_box_xyxy = validate_xyxy(self.original_box_xyxy)

        self.class_id, self.class_name = _class_metadata(
            self.class_id, self.class_name
        )
        if self.original_class_id is not None:
            original_name = (
                self.class_name
                if self.original_class_name is None
                else self.original_class_name
            )
            self.original_class_id, normalized_original_name = _class_metadata(
                self.original_class_id,
                original_name,
                prefix="Original annotation",
            )
            if self.original_class_name is not None:
                self.original_class_name = normalized_original_name
        elif self.original_class_name is not None:
            normalized_name = str(self.original_class_name).strip()
            if not normalized_name:
                raise ValueError("Original annotation class name must not be empty.")
            self.original_class_name = normalized_name

        if self.confidence is not None:
            self.confidence = float(self.confidence)
            if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
                raise ValueError("Annotation confidence must be finite and between 0 and 1.")

        self.polygon_xyn = _optional_polygon(self.polygon_xyn)
        self.original_polygon_xyn = _optional_polygon(self.original_polygon_xyn)
        self.deleted = _strict_bool(self.deleted, "deleted")

        if self.source == AnnotationSource.SAM3:
            self.original_box_xyxy = self.original_box_xyxy or self.box_xyxy
            if self.original_polygon_xyn is None:
                self.original_polygon_xyn = self.polygon_xyn
            if self.original_class_id is None:
                self.original_class_id = self.class_id
            if self.original_class_name is None:
                self.original_class_name = self.class_name

        if self.segmentation_valid is None:
            self.segmentation_valid = (
                self.source == AnnotationSource.SAM3 and bool(self.polygon_xyn)
            )
        else:
            self.segmentation_valid = _strict_bool(
                self.segmentation_valid,
                "segmentation_valid",
                allow_none=True,
            )
        if self.segmentation_valid and self.polygon_xyn is None:
            raise ValueError("A valid segmentation requires a polygon.")
        if self.segmentation_source is None and self.segmentation_valid:
            self.segmentation_source = (
                "sam3_box_prompt"
                if self.source == AnnotationSource.SAM3_REFINED
                else "sam3_original"
            )

    @property
    def is_active(self):
        return not self.deleted

    @property
    def can_reset_to_sam3(self):
        return self.original_box_xyxy is not None

    @property
    def is_modified_from_sam3(self):
        if not self.can_reset_to_sam3:
            return False
        return any(
            (
                self.source != AnnotationSource.SAM3,
                self.box_xyxy != self.original_box_xyxy,
                self.class_id != self.original_class_id,
                self.class_name != self.original_class_name,
                self.polygon_xyn != self.original_polygon_xyn,
                self.deleted,
            )
        )

    def edit_box(self, box_xyxy, image_width=None, image_height=None):
        self.box_xyxy = (
            clip_xyxy(box_xyxy, image_width, image_height)
            if image_width is not None and image_height is not None
            else validate_xyxy(box_xyxy)
        )
        self.source = AnnotationSource.EDITED
        self.segmentation_valid = False
        self.deleted = False

    def change_class(self, class_id, class_name):
        self.class_id, self.class_name = _class_metadata(class_id, class_name)
        self.source = AnnotationSource.EDITED
        self.segmentation_valid = False
        self.deleted = False

    def reset_to_sam3(self):
        if not self.can_reset_to_sam3:
            raise ValueError("This annotation has no original SAM3 geometry to restore.")
        self.box_xyxy = validate_xyxy(self.original_box_xyxy)
        self.polygon_xyn = self.original_polygon_xyn
        if self.original_class_id is not None:
            self.class_id = int(self.original_class_id)
        if self.original_class_name is not None:
            self.class_name = self.original_class_name
        self.source = AnnotationSource.SAM3
        self.segmentation_valid = bool(self.original_polygon_xyn)
        self.segmentation_source = "sam3_original" if self.segmentation_valid else None
        self.deleted = False

    def apply_sam3_box_prompt_segmentation(self, polygon_xyn, confidence=None):
        polygon = validate_polygon_xyn(polygon_xyn)
        self.polygon_xyn = polygon
        self.source = AnnotationSource.SAM3_REFINED
        self.segmentation_valid = True
        self.segmentation_source = "sam3_box_prompt"
        if confidence is not None:
            confidence = float(confidence)
            if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("Segmentation confidence must be finite and between 0 and 1.")
            self.confidence = confidence
        self.deleted = False

    def mark_deleted(self):
        self.deleted = True

    def to_dict(self):
        return {
            "id": self.id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "box_xyxy": list(self.box_xyxy),
            "source": self.source.value,
            "confidence": self.confidence,
            "polygon_xyn": self.polygon_xyn,
            "segmentation_valid": self.segmentation_valid,
            "segmentation_source": self.segmentation_source,
            "original_box_xyxy": (
                None if self.original_box_xyxy is None else list(self.original_box_xyxy)
            ),
            "original_polygon_xyn": self.original_polygon_xyn,
            "original_class_id": self.original_class_id,
            "original_class_name": self.original_class_name,
            "deleted": self.deleted,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            class_id=data["class_id"],
            class_name=data["class_name"],
            box_xyxy=tuple(data["box_xyxy"]),
            source=data.get("source", AnnotationSource.SAM3),
            confidence=data.get("confidence"),
            polygon_xyn=data.get("polygon_xyn"),
            segmentation_valid=data.get("segmentation_valid"),
            segmentation_source=data.get("segmentation_source"),
            original_box_xyxy=(
                None
                if data.get("original_box_xyxy") is None
                else tuple(data["original_box_xyxy"])
            ),
            original_polygon_xyn=data.get("original_polygon_xyn"),
            original_class_id=data.get("original_class_id"),
            original_class_name=data.get("original_class_name"),
            deleted=data.get("deleted", False),
        )
