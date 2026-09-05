from enum import Enum


class UiMode(str, Enum):
    EMPTY = "empty"
    READY = "ready"
    PREDICTING = "predicting"
    BATCH = "batch"
    RESEGMENTING = "resegmenting"
