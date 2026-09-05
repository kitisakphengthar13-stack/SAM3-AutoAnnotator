"""Application workflows used by the desktop interface."""

from services.annotation_service import (
    add_manual_annotation,
    apply_box_segmentation,
    change_annotation_class,
    delete_annotation,
    edit_annotation_box,
    mark_image_reviewed,
    reset_annotation_to_sam3,
)
from services.prediction_service import (
    BoxSegmentation,
    ImagePrediction,
    PredictionService,
)

__all__ = [
    "BoxSegmentation",
    "ImagePrediction",
    "PredictionService",
    "add_manual_annotation",
    "apply_box_segmentation",
    "change_annotation_class",
    "delete_annotation",
    "edit_annotation_box",
    "mark_image_reviewed",
    "reset_annotation_to_sam3",
]
