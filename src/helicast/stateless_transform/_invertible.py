import numpy as np
import pandas as pd

from helicast.base import (
    InvertibleTransformerMixin,
    StatelessEstimator,
    dataclass,
)

__all__ = [
    "SquareTransformer",
]


@dataclass
class SquareTransformer(InvertibleTransformerMixin, StatelessEstimator):
    """Invertible stateless transformer that applies the square function to a
    DataFrame."""

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return np.square(X)

    def _inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return np.sqrt(X)
