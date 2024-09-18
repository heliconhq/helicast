from typing import Protocol, runtime_checkable

import pandas as pd
from typing_extensions import Self

__all__ = [
    "TransformerType",
    "InvertibleTransformerType",
    "PredictorType",
]


@runtime_checkable
class TransformerType(Protocol):
    """Protocol for objects that have a ``fit`` and ``transform`` method."""

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None, **kwargs
    ) -> Self: ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...


@runtime_checkable
class InvertibleTransformerType(Protocol):
    """Protocol for objects that have a ``fit``, ``transform`` and ``inverse_transform``
    method."""

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None, **kwargs
    ) -> Self: ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame: ...


@runtime_checkable
class PredictorType(Protocol):
    """Protocol for objects that have a ``fit`` and ``predict`` method."""

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None, **kwargs
    ) -> Self: ...

    def predict(self, X: pd.DataFrame) -> pd.DataFrame: ...
