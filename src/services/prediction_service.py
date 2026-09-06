"""Synchronous SAM3 inference workflows, independent of Qt threading."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from domain import Annotation
from domain.geometry import validate_xyxy
from sam3.predictor import create_box_segmenter
from sam3.predictor_cache import PredictorCache
from sam3.result_mapper import (
    annotations_from_sam3_result,
    best_box_prompt_segmentation,
    result_image_size,
)


@dataclass(frozen=True)
class ImagePrediction:
    image_path: Path
    annotations: list[Annotation]
    width: int | None
    height: int | None
    reused_predictor: bool


@dataclass(frozen=True)
class BoxSegmentation:
    image_path: Path
    box_xyxy: tuple[float, float, float, float]
    polygon_xyn: list[list[float]]
    confidence: float | None
    reused_predictor: bool


def _existing_file(path, field_name: str) -> Path:
    if path is None or not str(path).strip():
        raise ValueError(f"{field_name} must not be empty.")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{field_name} does not exist or is not a file: {resolved}")
    return resolved


def _validated_confidence(confidence: float) -> float:
    try:
        value = float(confidence)
    except (TypeError, ValueError) as exc:
        raise TypeError("confidence must be a number between 0.01 and 1.0.") from exc
    if not 0.01 <= value <= 1.0:
        raise ValueError("confidence must be between 0.01 and 1.0.")
    return value


def _validated_prompts(prompts: Iterable[str]) -> list[str]:
    if isinstance(prompts, (str, bytes)) or prompts is None:
        raise TypeError("prompts must be an iterable of class names, not a string.")
    source = list(prompts)
    if any(not isinstance(prompt, str) for prompt in source):
        raise TypeError("Every class prompt must be a string.")
    normalized = [prompt.strip() for prompt in source]
    if not normalized or any(not prompt for prompt in normalized):
        raise ValueError("Enter at least one non-empty class prompt.")
    if len(normalized) != len(set(normalized)):
        raise ValueError("Class prompts must be unique.")
    return normalized


def _validated_half(half: bool) -> bool:
    if not isinstance(half, bool):
        raise TypeError("half must be a boolean.")
    return half


def _first_result(results):
    if results is None:
        raise ValueError("SAM3 returned no prediction results.")
    try:
        return results[0]
    except (IndexError, TypeError):
        try:
            return next(iter(results))
        except (StopIteration, TypeError) as exc:
            raise ValueError("SAM3 returned no prediction results.") from exc


class PredictionService:
    """Run SAM3 requests serially while keeping only one model mode cached."""

    def __init__(
        self,
        predictor_cache: PredictorCache | None = None,
        box_segmenter_factory=create_box_segmenter,
    ):
        self.predictor_cache = predictor_cache or PredictorCache()
        self._box_segmenter_factory = box_segmenter_factory
        self._box_segmenter = None
        self._box_segmenter_model_path = None
        self._inference_lock = Lock()

    def _drop_box_segmenter(self):
        self._box_segmenter = None
        self._box_segmenter_model_path = None

    def _get_box_segmenter(self, model_file):
        key = str(Path(model_file).resolve())
        reused = self._box_segmenter is not None and self._box_segmenter_model_path == key
        if not reused:
            self._box_segmenter = self._box_segmenter_factory(key)
            self._box_segmenter_model_path = key
        return self._box_segmenter, reused

    def predict_image(
        self,
        *,
        image_path,
        model_path,
        prompts: Iterable[str],
        confidence: float,
        half: bool = True,
    ) -> ImagePrediction:
        image_file = _existing_file(image_path, "image_path")
        model_file = _existing_file(model_path, "model_path")
        normalized_prompts = _validated_prompts(prompts)
        normalized_confidence = _validated_confidence(confidence)
        use_half = _validated_half(half)

        with self._inference_lock:
            # Avoid retaining both the semantic predictor and visual SAM model in
            # GPU memory. Switching modes intentionally drops the other cache.
            self._drop_box_segmenter()
            predictor, reused = self.predictor_cache.get_predictor(
                model_path=model_file,
                conf=normalized_confidence,
                half=use_half,
            )
            predictor.set_image(str(image_file))
            result = _first_result(predictor(text=normalized_prompts))

        annotations = annotations_from_sam3_result(result, normalized_prompts)
        width, height = result_image_size(result)
        return ImagePrediction(
            image_path=image_file,
            annotations=annotations,
            width=width,
            height=height,
            reused_predictor=reused,
        )

    def segment_box(
        self,
        *,
        image_path,
        model_path,
        box_xyxy,
        class_name: str,
        confidence: float,
        half: bool = True,
    ) -> BoxSegmentation:
        image_file = _existing_file(image_path, "image_path")
        model_file = _existing_file(model_path, "model_path")
        box = validate_xyxy(box_xyxy)
        if not isinstance(class_name, str):
            raise TypeError("class_name must be a string.")
        if not class_name.strip():
            raise ValueError("class_name must not be empty.")
        _validated_confidence(confidence)
        _validated_half(half)

        # Re-segmentation is a positional single-object task. Ultralytics documents
        # SAM3SemanticPredictor bboxes as concept exemplars that can return similar
        # objects elsewhere in the image, so use the base SAM visual-prompt API here.
        with self._inference_lock:
            self.predictor_cache.clear()
            segmenter, reused = self._get_box_segmenter(model_file)
            results = segmenter.predict(
                source=str(image_file),
                bboxes=list(box),
                verbose=False,
            )
            polygon_xyn, result_confidence = best_box_prompt_segmentation(
                results,
                requested_box=box,
            )

        return BoxSegmentation(
            image_path=image_file,
            box_xyxy=box,
            polygon_xyn=polygon_xyn,
            confidence=(
                None if result_confidence is None else float(result_confidence)
            ),
            reused_predictor=reused,
        )


__all__ = ["BoxSegmentation", "ImagePrediction", "PredictionService"]
