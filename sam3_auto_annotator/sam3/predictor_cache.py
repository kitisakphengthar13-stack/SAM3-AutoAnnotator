from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from sam3_auto_annotator.sam3.predictor import create_predictor


@dataclass(frozen=True)
class PredictorCacheKey:
    model_path: str
    conf: float
    half: bool

    @classmethod
    def from_settings(cls, model_path, conf, half):
        return cls(
            model_path=str(Path(model_path).resolve()),
            conf=round(float(conf), 6),
            half=bool(half),
        )


class PredictorCache:
    def __init__(self, factory=create_predictor):
        self._factory = factory
        self._key = None
        self._predictor = None
        self._lock = Lock()

    def has_predictor(self, model_path, conf, half):
        key = PredictorCacheKey.from_settings(model_path, conf, half)
        with self._lock:
            return self._key == key and self._predictor is not None

    def get_predictor(self, model_path, conf, half):
        key = PredictorCacheKey.from_settings(model_path, conf, half)
        with self._lock:
            if self._key == key and self._predictor is not None:
                return self._predictor, True

            predictor = self._factory(
                model_path=key.model_path,
                conf=key.conf,
                half=key.half,
            )
            self._key = key
            self._predictor = predictor
            return predictor, False

    def clear(self):
        with self._lock:
            self._key = None
            self._predictor = None
