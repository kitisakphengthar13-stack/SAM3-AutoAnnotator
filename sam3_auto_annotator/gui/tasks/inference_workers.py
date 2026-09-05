from __future__ import annotations

from dataclasses import dataclass
import logging
from threading import Event

from PySide6.QtCore import QObject, QThread, Signal, Slot


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchItem:
    image_index: int
    image_path: str


class PredictionWorker(QObject):
    result = Signal(int, object)
    failed = Signal(int, str)
    status = Signal(str)
    finished = Signal()

    def __init__(self, service, image_index, settings, parent=None):
        super().__init__(parent)
        self.service = service
        self.image_index = int(image_index)
        self.settings = dict(settings)

    @Slot()
    def run(self):
        try:
            self.status.emit("Running SAM3 on the selected image…")
            prediction = self.service.predict_image(**self.settings)
            self.result.emit(self.image_index, prediction)
        except Exception as exc:
            logger.exception("SAM3 prediction failed for image index %s", self.image_index)
            self.failed.emit(self.image_index, str(exc))
        finally:
            self.finished.emit()


class BoxSegmentationWorker(QObject):
    result = Signal(int, str, object)
    failed = Signal(int, str, str)
    status = Signal(str)
    finished = Signal()

    def __init__(self, service, image_index, annotation_id, settings, parent=None):
        super().__init__(parent)
        self.service = service
        self.image_index = int(image_index)
        self.annotation_id = str(annotation_id)
        self.settings = dict(settings)

    @Slot()
    def run(self):
        try:
            self.status.emit("Re-segmenting the selected box with SAM3…")
            segmentation = self.service.segment_box(**self.settings)
            self.result.emit(self.image_index, self.annotation_id, segmentation)
        except Exception as exc:
            logger.exception(
                "SAM3 box segmentation failed for image index %s, annotation %s",
                self.image_index,
                self.annotation_id,
            )
            self.failed.emit(self.image_index, self.annotation_id, str(exc))
        finally:
            self.finished.emit()


class BatchPredictionWorker(QObject):
    progress = Signal(int, int, str)
    image_result = Signal(int, object)
    image_failed = Signal(int, str)
    status = Signal(str)
    completed = Signal(dict)
    cancelled = Signal(dict)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, service, items, common_settings, parent=None):
        super().__init__(parent)
        self.service = service
        self.items = list(items)
        self.common_settings = dict(common_settings)
        self._cancel_requested = Event()

    def request_cancel(self):
        self._cancel_requested.set()

    def _cancelled(self):
        return (
            self._cancel_requested.is_set()
            or QThread.currentThread().isInterruptionRequested()
        )

    @Slot()
    def run(self):
        summary = {
            "processed": 0,
            "predicted": 0,
            "no_detection": 0,
            "errors": 0,
            "total": len(self.items),
        }
        try:
            self.status.emit("Running SAM3 on remaining images…")
            for position, item in enumerate(self.items, start=1):
                if self._cancelled():
                    self.cancelled.emit(summary)
                    return
                self.progress.emit(position, len(self.items), item.image_path)
                try:
                    prediction = self.service.predict_image(
                        image_path=item.image_path,
                        **self.common_settings,
                    )
                    summary["processed"] += 1
                    key = "predicted" if prediction.annotations else "no_detection"
                    summary[key] += 1
                    self.image_result.emit(item.image_index, prediction)
                except Exception as exc:
                    logger.exception(
                        "SAM3 batch prediction failed for image index %s",
                        item.image_index,
                    )
                    summary["processed"] += 1
                    summary["errors"] += 1
                    self.image_failed.emit(item.image_index, str(exc))
            self.completed.emit(summary)
        except Exception as exc:
            logger.exception("SAM3 batch worker failed")
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()
