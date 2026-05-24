from PySide6.QtCore import QThread, Signal

from sam3_auto_annotator.annotation.sam3 import (
    annotations_from_sam3_result,
    best_box_prompt_segmentation,
    result_image_size,
)
from sam3_auto_annotator.gui.predictor_cache import PredictorCache


class PredictionWorker(QThread):
    finished_prediction = Signal(int, list, object, object)
    failed = Signal(int, str)
    status = Signal(str)

    def __init__(
        self,
        image_index,
        image_path,
        model_path,
        prompts,
        conf,
        half=True,
        predictor_cache=None,
        parent=None,
    ):
        super().__init__(parent)
        self.image_index = image_index
        self.image_path = image_path
        self.model_path = model_path
        self.prompts = list(prompts)
        self.conf = float(conf)
        self.half = bool(half)
        self.predictor_cache = predictor_cache or PredictorCache()

    def run(self):
        try:
            if self.predictor_cache.has_predictor(self.model_path, self.conf, self.half):
                self.status.emit("Reusing loaded SAM3 model...")
            else:
                self.status.emit("Loading SAM3 model...")
            predictor, reused = self.predictor_cache.get_predictor(
                model_path=self.model_path,
                conf=self.conf,
                half=self.half,
            )
            if reused:
                self.status.emit("Running SAM3 with cached model...")
            else:
                self.status.emit("Running SAM3...")
            predictor.set_image(str(self.image_path))
            results = predictor(text=self.prompts)
            result = results[0]
            annotations = annotations_from_sam3_result(result, self.prompts)
            width, height = result_image_size(result)
            self.finished_prediction.emit(self.image_index, annotations, width, height)
        except Exception as exc:
            self.failed.emit(self.image_index, str(exc))


class BoxPromptSegmentationWorker(QThread):
    finished_segmentation = Signal(int, str, list, object)
    failed = Signal(int, str, str)
    status = Signal(str)

    def __init__(
        self,
        image_index,
        annotation_id,
        image_path,
        model_path,
        box_xyxy,
        class_name,
        conf,
        half=True,
        predictor_cache=None,
        parent=None,
    ):
        super().__init__(parent)
        self.image_index = image_index
        self.annotation_id = annotation_id
        self.image_path = image_path
        self.model_path = model_path
        self.box_xyxy = tuple(float(value) for value in box_xyxy)
        self.class_name = str(class_name)
        self.conf = float(conf)
        self.half = bool(half)
        self.predictor_cache = predictor_cache or PredictorCache()

    def run(self):
        try:
            if self.predictor_cache.has_predictor(self.model_path, self.conf, self.half):
                self.status.emit("Reusing loaded SAM3 model...")
            else:
                self.status.emit("Loading SAM3 model...")
            predictor, reused = self.predictor_cache.get_predictor(
                model_path=self.model_path,
                conf=self.conf,
                half=self.half,
            )
            self.status.emit(
                "Running SAM3 box-prompt segmentation with cached model..."
                if reused
                else "Running SAM3 box-prompt segmentation..."
            )
            predictor.set_image(str(self.image_path))
            results = predictor(bboxes=[self.box_xyxy], text=[self.class_name])
            polygon_xyn, confidence = best_box_prompt_segmentation(results)
            self.finished_segmentation.emit(
                self.image_index,
                self.annotation_id,
                polygon_xyn,
                confidence,
            )
        except Exception as exc:
            self.failed.emit(self.image_index, self.annotation_id, str(exc))


class BatchPredictionWorker(QThread):
    progress = Signal(int, int, str)
    image_finished = Signal(int, list, object, object)
    image_failed = Signal(int, str)
    status = Signal(str)
    failed = Signal(str)
    completed = Signal(dict)
    cancelled = Signal(dict)

    def __init__(
        self,
        image_records,
        model_path,
        prompts,
        conf,
        half=True,
        predictor_cache=None,
        parent=None,
    ):
        super().__init__(parent)
        self.image_records = list(image_records)
        self.model_path = model_path
        self.prompts = list(prompts)
        self.conf = float(conf)
        self.half = bool(half)
        self.predictor_cache = predictor_cache or PredictorCache()
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def run(self):
        total = len(self.image_records)
        summary = {
            "processed": 0,
            "predicted": 0,
            "no_detection": 0,
            "errors": 0,
            "total": total,
        }
        try:
            if self.predictor_cache.has_predictor(self.model_path, self.conf, self.half):
                self.status.emit("Reusing loaded SAM3 model for batch...")
            else:
                self.status.emit("Loading SAM3 model for batch...")
            predictor, reused = self.predictor_cache.get_predictor(
                model_path=self.model_path,
                conf=self.conf,
                half=self.half,
            )
            self.status.emit(
                "Running SAM3 batch with cached model..." if reused else "Running SAM3 batch..."
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        for batch_index, image in enumerate(self.image_records, start=1):
            if self._cancel_requested:
                self.cancelled.emit(summary)
                return

            self.progress.emit(batch_index, total, image.image_path)
            try:
                predictor.set_image(str(image.image_path))
                results = predictor(text=self.prompts)
                result = results[0]
                annotations = annotations_from_sam3_result(result, self.prompts)
                width, height = result_image_size(result)
                summary["processed"] += 1
                if annotations:
                    summary["predicted"] += 1
                else:
                    summary["no_detection"] += 1
                self.image_finished.emit(image.image_index, annotations, width, height)
            except Exception as exc:
                summary["processed"] += 1
                summary["errors"] += 1
                self.image_failed.emit(image.image_index, str(exc))

        self.completed.emit(summary)
