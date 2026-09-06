from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

from domain.annotation import Annotation, AnnotationSource


STATE_VERSION = 2
SUPPORTED_STATE_VERSIONS = frozenset({1, STATE_VERSION})


class ImageStatus(str, Enum):
    NOT_PREDICTED = "not_predicted"
    PREDICTED = "predicted"
    EDITED = "edited"
    REVIEWED = "reviewed"
    NO_DETECTION = "no_detection"
    ERROR = "error"


@dataclass
class ImageRecord:
    image_path: str
    image_index: int
    image_name: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    status: ImageStatus = ImageStatus.NOT_PREDICTED
    annotations: list[Annotation] = field(default_factory=list)
    error_message: Optional[str] = None
    source_size_bytes: Optional[int] = None
    source_mtime_ns: Optional[int] = None
    source_sha256: Optional[str] = None

    def __post_init__(self):
        self.status = (
            self.status if isinstance(self.status, ImageStatus) else ImageStatus(self.status)
        )
        self.image_path = str(self.image_path)
        self.image_index = int(self.image_index)
        if self.image_name is None:
            self.image_name = Path(self.image_path).name
        if self.width is not None:
            self.width = int(self.width)
            if self.width <= 0:
                raise ValueError("Image width must be positive when known.")
        if self.height is not None:
            self.height = int(self.height)
            if self.height <= 0:
                raise ValueError("Image height must be positive when known.")
        if self.source_size_bytes is not None:
            self.source_size_bytes = int(self.source_size_bytes)
            if self.source_size_bytes < 0:
                raise ValueError("Source file size cannot be negative.")
        if self.source_mtime_ns is not None:
            self.source_mtime_ns = int(self.source_mtime_ns)
            if self.source_mtime_ns < 0:
                raise ValueError("Source modification time cannot be negative.")
        if self.source_sha256 is not None:
            digest = str(self.source_sha256).strip().lower()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("Source SHA-256 must contain 64 hexadecimal characters.")
            self.source_sha256 = digest
        self.annotations = [
            item if isinstance(item, Annotation) else Annotation.from_dict(item)
            for item in self.annotations
        ]

    @property
    def active_annotations(self):
        return [item for item in self.annotations if item.is_active]

    @property
    def has_active_annotations(self):
        return bool(self.active_annotations)

    @property
    def is_predicted(self):
        return self.status != ImageStatus.NOT_PREDICTED

    def annotation_by_id(self, annotation_id):
        return next((item for item in self.annotations if item.id == annotation_id), None)

    def replace_sam3_drafts(self, annotations: Sequence[Annotation]):
        self.annotations = list(annotations)
        self.status = (
            ImageStatus.PREDICTED if self.annotations else ImageStatus.NO_DETECTION
        )
        self.error_message = None

    def add_manual_annotation(self, class_id, class_name, box_xyxy):
        annotation = Annotation(
            class_id=class_id,
            class_name=class_name,
            box_xyxy=box_xyxy,
            source=AnnotationSource.MANUAL,
        )
        self.annotations.append(annotation)
        self.mark_edited()
        return annotation

    def mark_reviewed(self):
        self.status = ImageStatus.REVIEWED
        self.error_message = None

    def mark_edited(self):
        self.status = ImageStatus.EDITED
        self.error_message = None

    def mark_error(self, message):
        if self.status in {ImageStatus.NOT_PREDICTED, ImageStatus.ERROR}:
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
            "annotations": [item.to_dict() for item in self.annotations],
            "error_message": self.error_message,
            "source_size_bytes": self.source_size_bytes,
            "source_mtime_ns": self.source_mtime_ns,
            "source_sha256": self.source_sha256,
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
            source_size_bytes=data.get("source_size_bytes"),
            source_mtime_ns=data.get("source_mtime_ns"),
            source_sha256=data.get("source_sha256"),
        )


@dataclass
class ProjectState:
    input_path: str
    prompts: list[str]
    images: list[ImageRecord]
    model_path: Optional[str] = None
    project_name: Optional[str] = None
    confidence: float = 0.5
    half: bool = True
    version: int = STATE_VERSION

    def __post_init__(self):
        self.version = STATE_VERSION
        self.input_path = str(self.input_path)
        self.prompts = [str(item).strip() for item in self.prompts if str(item).strip()]
        if len(self.prompts) != len(set(self.prompts)):
            raise ValueError("Class prompts must be unique.")
        self.confidence = float(self.confidence)
        if not 0.01 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.01 and 1.0.")
        if not isinstance(self.half, bool):
            raise TypeError("half must be a boolean.")
        self.images = [
            item if isinstance(item, ImageRecord) else ImageRecord.from_dict(item)
            for item in self.images
        ]
        indexes = [item.image_index for item in self.images]
        if len(indexes) != len(set(indexes)):
            raise ValueError("Image indexes must be unique within a project.")
        paths = [str(Path(item.image_path).resolve()).casefold() for item in self.images]
        if len(paths) != len(set(paths)):
            raise ValueError("Image paths must be unique within a project.")

        annotation_ids = [
            annotation.id
            for image in self.images
            for annotation in image.annotations
        ]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("Annotation ids must be unique within a project.")
        for image in self.images:
            for annotation in image.active_annotations:
                if not 0 <= annotation.class_id < len(self.prompts):
                    raise ValueError(
                        f"Annotation {annotation.id} references class id "
                        f"{annotation.class_id}, but the project has {len(self.prompts)} classes."
                    )
                if self.prompts[annotation.class_id] != annotation.class_name:
                    raise ValueError(
                        f"Annotation {annotation.id} class metadata is inconsistent with prompts."
                    )

    @property
    def class_map(self):
        return {name: index for index, name in enumerate(self.prompts)}

    @property
    def is_single_image(self):
        return len(self.images) == 1

    @property
    def unpredicted_images(self):
        return [item for item in self.images if item.status == ImageStatus.NOT_PREDICTED]

    @classmethod
    def from_image_paths(
        cls,
        input_path,
        image_paths,
        prompts,
        model_path=None,
        project_name=None,
        confidence=0.5,
        half=True,
    ):
        return cls(
            input_path=str(input_path),
            model_path=None if model_path is None else str(model_path),
            prompts=list(prompts),
            project_name=project_name,
            confidence=confidence,
            half=half,
            images=[
                ImageRecord(image_path=str(path), image_index=index)
                for index, path in enumerate(image_paths)
            ],
        )

    def get_image(self, image_index):
        image = next((item for item in self.images if item.image_index == image_index), None)
        if image is None:
            raise KeyError(f"No image with index {image_index}.")
        return image

    def active_annotations(self):
        return [item for image in self.images for item in image.active_annotations]

    def to_dict(self):
        return {
            "version": STATE_VERSION,
            "input_path": self.input_path,
            "model_path": self.model_path,
            "project_name": self.project_name,
            "prompts": self.prompts,
            "confidence": self.confidence,
            "half": self.half,
            "class_map": self.class_map,
            "images": [item.to_dict() for item in self.images],
        }

    @classmethod
    def from_dict(cls, data):
        source_version = int(data.get("version", 1))
        if source_version not in SUPPORTED_STATE_VERSIONS:
            raise ValueError(
                f"Project state version {source_version} is not supported. "
                f"Supported versions: {sorted(SUPPORTED_STATE_VERSIONS)}."
            )
        return cls(
            input_path=data["input_path"],
            model_path=data.get("model_path"),
            project_name=data.get("project_name"),
            prompts=data["prompts"],
            confidence=data.get("confidence", 0.5),
            half=data.get("half", True),
            images=data.get("images", []),
        )
