from __future__ import annotations

from enum import Enum

from PySide6.QtCore import QObject, QThread, Signal

from sam3_auto_annotator.gui.tasks.inference_workers import (
    BatchItem,
    BatchPredictionWorker,
    BoxSegmentationWorker,
    PredictionWorker,
)


class TaskKind(str, Enum):
    PREDICTION = "prediction"
    BATCH = "batch"
    SEGMENTATION = "segmentation"


class InferenceTaskManager(QObject):
    task_started = Signal(str)
    status = Signal(str)
    progress = Signal(int, int, str)
    prediction_ready = Signal(int, object)
    prediction_failed = Signal(int, str)
    segmentation_ready = Signal(int, str, object)
    segmentation_failed = Signal(int, str, str)
    batch_image_ready = Signal(int, object)
    batch_image_failed = Signal(int, str)
    batch_completed = Signal(dict)
    batch_cancelled = Signal(dict)
    task_failed = Signal(str)
    task_finished = Signal(str)

    def __init__(self, prediction_service, parent=None):
        super().__init__(parent)
        self.prediction_service = prediction_service
        self._thread = None
        self._worker = None
        self._kind = None

    @property
    def is_running(self):
        return self._thread is not None

    @property
    def kind(self):
        return self._kind

    def start_prediction(self, image_index, **settings):
        worker = PredictionWorker(
            self.prediction_service,
            image_index,
            settings,
        )
        worker.result.connect(self.prediction_ready)
        worker.failed.connect(self.prediction_failed)
        self._start(TaskKind.PREDICTION, worker)

    def start_segmentation(self, image_index, annotation_id, **settings):
        worker = BoxSegmentationWorker(
            self.prediction_service,
            image_index,
            annotation_id,
            settings,
        )
        worker.result.connect(self.segmentation_ready)
        worker.failed.connect(self.segmentation_failed)
        self._start(TaskKind.SEGMENTATION, worker)

    def start_batch(self, items, **common_settings):
        worker = BatchPredictionWorker(
            self.prediction_service,
            [BatchItem(item.image_index, item.image_path) for item in items],
            common_settings,
        )
        worker.progress.connect(self.progress)
        worker.image_result.connect(self.batch_image_ready)
        worker.image_failed.connect(self.batch_image_failed)
        worker.completed.connect(self.batch_completed)
        worker.cancelled.connect(self.batch_cancelled)
        worker.failed.connect(self.task_failed)
        self._start(TaskKind.BATCH, worker)

    def _start(self, kind, worker):
        if self.is_running:
            raise RuntimeError("Another SAM3 task is already running.")

        thread = QThread(self)
        thread.setObjectName(f"sam3-{kind.value}-thread")
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.status.connect(self.status)
        worker.finished.connect(thread.quit)
        # Schedule deletion while the worker thread can still process deferred
        # events.  Connecting this to ``thread.finished`` risks posting the
        # deletion only after that event loop has already stopped.
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup)

        self._thread = thread
        self._worker = worker
        self._kind = kind
        self.task_started.emit(kind.value)
        thread.start()

    def request_cancel(self):
        if not self.is_running:
            return False
        self._thread.requestInterruption()
        cancel = getattr(self._worker, "request_cancel", None)
        if cancel is not None:
            cancel()
        return True

    def _cleanup(self):
        kind = self._kind.value if self._kind is not None else "unknown"
        self._thread = None
        self._worker = None
        self._kind = None
        self.task_finished.emit(kind)
