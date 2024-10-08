import numpy as np
import pandas as pd

from helicast.base import (
    StatelessEstimator,
    TransformerMixin,
    dataclass,
)

__all__ = [
    "Clipper",
]


@dataclass
class Clipper(TransformerMixin, StatelessEstimator):
    """Non-invertible transformer that clips the values of a DataFrame to a given range.

    Args:
        lower: Lower bound of the clipping range. If None, no lower bound is applied.
        upper: Upper bound of the clipping range. If None, no upper bound is applied.
    """

    lower: float | int | None = None
    upper: float | int | None = None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return np.clip(X, a_min=self.lower, a_max=self.upper)
