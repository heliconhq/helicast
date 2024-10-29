from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from typing_extensions import Self

from helicast.base import (
    HelicastBaseEstimator,
    PredictorMixin,
    dataclass,
)


@dataclass
class DummyPredictor(PredictorMixin, HelicastBaseEstimator):
    a: int
    b: str
    c: float | None = None

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None, **kwargs) -> Self:
        self.y_ = deepcopy(y)
        return self

    def _predict(self, X: pd.DataFrame) -> pd.DataFrame | pd.Series | np.ndarray:
        return deepcopy(self.y_)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


@pytest.fixture
def X(df) -> pd.DataFrame:
    return df.copy()


@pytest.fixture
def y(df) -> pd.DataFrame:
    return df.copy()


def test_predictor(X, y):
    # --- Fit DF - DF
    model = DummyPredictor(a=1, b="2")
    model.fit(X, y)
    y_pred = model.predict(X)
    assert isinstance(y_pred, pd.DataFrame)

    # --- Fit DF - Series
    model = DummyPredictor(a=1, b="2")
    model.fit(X, y[y.columns[0]])
    y_pred = model.predict(X)
    assert isinstance(y_pred, pd.DataFrame)
    assert y_pred.columns == [y.columns[0]]
