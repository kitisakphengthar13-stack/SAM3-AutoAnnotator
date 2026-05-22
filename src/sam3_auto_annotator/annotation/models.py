from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from sam3_auto_annotator.annotation.geometry import clip_xyxy, validate_xyxy


STATE_VERSION = 1


class AnnotationSource(str, Enum):
    SAM3 = "sam3"
    EDITED = "edited"
    MANUAL = "manual"
    IMPORTED = "imported"


class ImageStatus(str, Enum):
    NOT_PREDICTED = "not_predicted"
    PREDICTED = "predicted"
    EDITED = "edited"
    REVIEWED = "reviewed"
    NO_DETECTION = "no_detection"
    ERROR = "error"


def _enum_value(value):
    return value.value if isinstance(value, Enum) else value


def _coerce_source(value):
    return value if isinstance(value, AnnotationSource) else AnnotationSource(value)


def _coerce_status(value):
    return value if isinstance(value, ImageStatus) else ImageStatus(value)


@dataclass
class Annotation:
    class_id: int
    class_name: str
    box_xyxy: Tuple[float, float, float, float]
    id: str = field(default_factory=lambda: str(uuid4()))
    source: AnnotationSource = AnnotationSource.SAM3
    confidence: Optional[float] = None
    polygon_xyn: Optional[List[List[float]]] = None
    deleted: bool = False

    def __post_init__(self):
        self.source = _coerce_source(self.source)
        self.box_xyxy = validate_xyxy(self.box_xyxy)
        self.class_id = int(self.class_id)
        self.confidence = None if self.confidence is None else float(self.confidence)

    @property
    def is_active(self):
        return not self.deleted

    def edit_box(self, box_xyxy, image_width=None, image_height=None):
        if image_width is not None and image_height is not None:
            self.box_xyxy = clip_xyxy(box_xyxy, image_width, image_height)
        else:
            self.box_xyxy = validate_xyxy(box_xyxy)
        if self.source in {AnnotationSource.SAM3, AnnotationSource.IMPORTED}:
            self.source = AnnotationSource.EDITED
        self.deleted = False

    def change_class(self, class_id, class_name):
        self.class_id = int(class_id)
        self.class_name = str(class_name)
        if self.source in {AnnotationSource.SAM3, AnnotationSource.IMPORTED}:
            self.source = AnnotationSource.EDITED

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
            deleted=bool(data.get("deleted", False)),
        )


@dataclass
class ImageRecord:
    image_path: str
    image_index: int
    image_name: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    status: ImageStatus = ImageStatus.NOT_PREDICTED
    annotations: List[Annotation] = field(default_factory=list)
    error_message: Optional[str] = None

    def __post_init__(self):
        self.status = _coerce_status(self.status)
        self.image_index = int(self.image_index)
        if self.image_name is None:
            self.image_name = Path(self.image_path).name
        if self.width is not None:
            self.width = int(self.width)
        if self.height is not None:
            self.height = int(self.height)
        self.annotations = [
            annotation if isinstance(annotation, Annotation) else Annotation.from_dict(annotation)
            for annotation in self.annotations
        ]

    @property
    def active_annotations(self):
        return [annotation for annotation in self.annotations if annotation.is_active]

    @property
    def has_active_annotations(self):
        return bool(self.active_annotations)

    @property
    def is_predicted(self):
        return self.status != ImageStatus.NOT_PREDICTED

    def replace_sam3_drafts(self, annotations: Sequence[Annotation]):
        self.annotations = list(annotations)
        self.status = ImageStatus.PREDICTED if self.annotations else ImageStatus.NO_DETECTION
        self.error_message = None

    def add_manual_annotation(self, class_id, class_name, box_xyxy):
        annotation = Annotation(
            class_id=class_id,
            class_name=class_name,
            box_xyxy=box_xyxy,
            source=AnnotationSource.MANUAL,
        )
        self.annotations.append(annotation)
        self.status = ImageStatus.EDITED
        return annotation

    def mark_reviewed(self):
        self.status = ImageStatus.REVIEWED

    def mark_error(self, message):
        self.status = ImageStatus.ERROR
        self.error_message = str(message)

    def to_dict(self):
        return {
            "image_path": self.image_path,
            "image_name": self.image_name,
            "image_index": self.image_index,
            "width": self.width,
            "height": self.height,
            "status": self.status.value,
            "annotations": [annotation.to_dict() for annotation in self.annotations],
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            image_path=data["image_path"],
            image_name=data.get("image_name"),
            image_index=data["image_index"],
            width=data.get("width"),
            height=data.get("height"),
            status=data.get("status", ImageStatus.NOT_PREDICTED),
            annotations=data.get("annotations", []),
            error_message=data.get("error_message"),
        )


@dataclass
class ProjectState:
    input_path: str
    prompts: List[str]
    images: List[ImageRecord]
    model_path: Optional[str] = None
    project_name: Optional[str] = None
    version: int = STATE_VERSION

    def __post_init__(self):
        self.prompts = list(self.prompts)
        self.images = [
            image if isinstance(image, ImageRecord) else ImageRecord.from_dict(image)
            for image in self.images
        ]

    @property
    def class_map(self) -> Dict[str, int]:
        return {class_name: class_id for class_id, class_name in enumerate(self.prompts)}

    @property
    def is_single_image(self):
        return len(self.images) == 1

    @property
    def unpredicted_images(self):
        return [image for image in self.images if image.status == ImageStatus.NOT_PREDICTED]

    @classmethod
    def from_image_paths(cls, input_path, image_paths, prompts, model_path=None, project_name=None):
        images = [
            ImageRecord(image_path=str(image_path), image_index=index)
            for index, image_path in enumerate(image_paths)
        ]
        return cls(
            input_path=str(input_path),
            model_path=None if model_path is None else str(model_path),
            prompts=list(prompts),
            project_name=project_name,
            images=images,
        )

    def get_image(self, image_index):
        for image in self.images:
            if image.image_index == image_index:
                return image
        raise KeyError(f"No image with index {image_index}.")

    def active_annotations(self):
        annotations = []
        for image in self.images:
            annotations.extend(image.active_annotations)
        return annotations

    def to_dict(self):
        return {
            "version": self.version,
            "input_path": self.input_path,
            "model_path": self.model_path,
            "project_name": self.project_name,
            "prompts": self.prompts,
            "class_map": self.class_map,
            "images": [image.to_dict() for image in self.images],
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            version=data.get("version", STATE_VERSION),
            input_path=data["input_path"],
            model_path=data.get("model_path"),
            project_name=data.get("project_name"),
            prompts=data["prompts"],
            images=data.get("images", []),
        )
