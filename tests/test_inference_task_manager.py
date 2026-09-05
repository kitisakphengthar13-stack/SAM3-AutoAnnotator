import threading
import unittest
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QThread, QTimer
from PySide6.QtTest import QSignalSpy

from sam3_auto_annotator.core import ImageRecord
from sam3_auto_annotator.gui.tasks.inference_task_manager import (
    InferenceTaskManager,
    TaskKind,
)
from sam3_auto_annotator.services.prediction_service import ImagePrediction


def wait_for_signal(spy, timeout=3000):
    """Wait while pumping the Qt event dispatcher used by queued thread signals.

    ``QSignalSpy.wait()`` does not reliably dispatch cross-thread queued signals
    on every supported PySide6/Python combination, so the spy remains the
    assertion recorder while a short nested Qt event loop performs the wait.
    """

    if spy.count():
        return True
    loop = QEventLoop()
    poll_timer = QTimer()
    poll_timer.setInterval(5)
    poll_timer.timeout.connect(lambda: loop.quit() if spy.count() else None)
    timeout_timer = QTimer()
    timeout_timer.setSingleShot(True)
    timeout_timer.timeout.connect(loop.quit)
    poll_timer.start()
    timeout_timer.start(timeout)
    loop.exec()
    poll_timer.stop()
    timeout_timer.stop()
    return bool(spy.count())


class FakePredictionService:
    def __init__(self, *, failure=None, annotations=None):
        self.failure = failure
        self.annotations = [] if annotations is None else list(annotations)
        self.calls = []

    def predict_image(self, **settings):
        self.calls.append(dict(settings))
        if self.failure is not None:
            raise self.failure
        return ImagePrediction(
            image_path=Path(settings["image_path"]),
            annotations=list(self.annotations),
            width=640,
            height=480,
            reused_predictor=False,
        )


class GatedPredictionService(FakePredictionService):
    def __init__(self):
        super().__init__(annotations=[object()])
        self.entered = threading.Event()
        self.release = threading.Event()

    def predict_image(self, **settings):
        self.calls.append(dict(settings))
        self.entered.set()
        if not self.release.wait(timeout=3):
            raise TimeoutError("The cancellation test did not release fake inference.")
        return ImagePrediction(
            image_path=Path(settings["image_path"]),
            annotations=list(self.annotations),
            width=640,
            height=480,
            reused_predictor=False,
        )


class InferenceTaskManagerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self.managers = []
        self.release_events = []

    def tearDown(self):
        for event in self.release_events:
            event.set()
        for manager in self.managers:
            if manager.is_running:
                manager.request_cancel()
                thread = manager._thread
                thread.quit()
                thread.wait(3000)
            QCoreApplication.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QCoreApplication.processEvents()

    def _manager(self, service):
        manager = InferenceTaskManager(service)
        self.managers.append(manager)
        return manager

    def assert_manager_stopped_without_orphan_thread(self, manager):
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QCoreApplication.processEvents()
        self.assertFalse(manager.is_running)
        self.assertIsNone(manager.kind)
        live_children = [
            thread
            for thread in manager.findChildren(QThread)
            if thread.isRunning()
        ]
        self.assertEqual(live_children, [])

    def test_prediction_success_emits_result_and_releases_thread(self):
        service = FakePredictionService(annotations=[object()])
        manager = self._manager(service)
        started = QSignalSpy(manager.task_started)
        ready = QSignalSpy(manager.prediction_ready)
        failed = QSignalSpy(manager.prediction_failed)
        finished = QSignalSpy(manager.task_finished)

        manager.start_prediction(
            7,
            image_path="car_1.jpg",
            model_path="unused.pt",
            prompts=["car"],
            confidence=0.5,
            half=False,
        )

        self.assertTrue(wait_for_signal(finished))
        self.assertEqual(started.at(0), [TaskKind.PREDICTION.value])
        self.assertEqual(failed.count(), 0)
        self.assertEqual(ready.count(), 1)
        image_index, prediction = ready.at(0)
        self.assertEqual(image_index, 7)
        self.assertEqual(prediction.image_path, Path("car_1.jpg"))
        self.assertEqual((prediction.width, prediction.height), (640, 480))
        self.assertEqual(finished.at(0), [TaskKind.PREDICTION.value])
        self.assertEqual(service.calls[0]["prompts"], ["car"])
        self.assert_manager_stopped_without_orphan_thread(manager)

    def test_prediction_failure_emits_context_and_releases_thread(self):
        service = FakePredictionService(
            failure=RuntimeError("synthetic inference failure")
        )
        manager = self._manager(service)
        ready = QSignalSpy(manager.prediction_ready)
        failed = QSignalSpy(manager.prediction_failed)
        finished = QSignalSpy(manager.task_finished)

        manager.start_prediction(
            9,
            image_path="car_2.jpg",
            model_path="unused.pt",
            prompts=["car"],
            confidence=0.5,
            half=False,
        )

        self.assertTrue(wait_for_signal(finished))
        self.assertEqual(ready.count(), 0)
        self.assertEqual(failed.count(), 1)
        self.assertEqual(failed.at(0), [9, "synthetic inference failure"])
        self.assertEqual(finished.at(0), [TaskKind.PREDICTION.value])
        self.assert_manager_stopped_without_orphan_thread(manager)

    def test_batch_cancel_stops_before_next_image_and_releases_thread(self):
        service = GatedPredictionService()
        self.release_events.append(service.release)
        manager = self._manager(service)
        progress = QSignalSpy(manager.progress)
        image_ready = QSignalSpy(manager.batch_image_ready)
        completed = QSignalSpy(manager.batch_completed)
        cancelled = QSignalSpy(manager.batch_cancelled)
        finished = QSignalSpy(manager.task_finished)
        items = [
            ImageRecord(f"car_{index + 1}.jpg", index)
            for index in range(3)
        ]

        manager.start_batch(
            items,
            model_path="unused.pt",
            prompts=["car"],
            confidence=0.5,
            half=False,
        )

        self.assertTrue(wait_for_signal(progress))
        self.assertTrue(service.entered.wait(timeout=1))
        self.assertTrue(manager.request_cancel())
        service.release.set()

        self.assertTrue(wait_for_signal(cancelled))
        self.assertTrue(wait_for_signal(finished))
        self.assertEqual(completed.count(), 0)
        self.assertEqual(image_ready.count(), 1)
        self.assertEqual(len(service.calls), 1)
        summary = cancelled.at(0)[0]
        self.assertEqual(
            summary,
            {
                "processed": 1,
                "predicted": 1,
                "no_detection": 0,
                "errors": 0,
                "total": 3,
            },
        )
        self.assertEqual(finished.at(0), [TaskKind.BATCH.value])
        self.assert_manager_stopped_without_orphan_thread(manager)


if __name__ == "__main__":
    unittest.main()
