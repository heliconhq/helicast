from enum import auto

from strenum import StrEnum

__all__ = [
    "EstimatorMode",
]


class EstimatorMode(StrEnum):
    """Enumeration used to determine the mode of the estimator."""

    FIT = auto()
    TRANSFORM = auto()
    INVERSE_TRANSFORM = auto()
    PREDICT = auto()
