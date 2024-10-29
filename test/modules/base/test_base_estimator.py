from copy import deepcopy
from io import BytesIO

import pandas as pd
import pytest
from joblib import dump, load
from sklearn.base import clone
from typing_extensions import Self

from helicast.base import (
    HelicastBaseEstimator,
    StatelessEstimator,
    dataclass,
    is_fitted,
)


@dataclass
class DummyEstimator(HelicastBaseEstimator):
    a: int
    b: str
    c: float | None = None

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None, **kwargs) -> Self:
        return self


@dataclass
class DummyStatelessEstimator(StatelessEstimator):
    a: int
    b: str
    c: float | None = None


@pytest.fixture
def estimator() -> HelicastBaseEstimator:
    return DummyEstimator(a=1, b="2")


@pytest.fixture
def stateless_estimator() -> StatelessEstimator:
    return DummyStatelessEstimator(a=1, b="2")


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


@pytest.fixture
def fitted_estimator(df) -> HelicastBaseEstimator:
    estimator = DummyEstimator(a=1, b="2")
    estimator.fit(df)
    return estimator


def test_get_set_params():
    # --- Test get_params() after initialization
    estimator = DummyEstimator(a=1, b="2")
    params = estimator.get_params()
    assert params == {"a": 1, "b": "2", "c": None}

    # --- Test set_params() with validation
    estimator.set_params(c="3")
    assert estimator.get_params() == {"a": 1, "b": "2", "c": 3}


def test_fit(
    estimator: HelicastBaseEstimator,
    stateless_estimator: StatelessEstimator,
    df: pd.DataFrame,
):
    # --- Test fit() with column names stored in feature_names_in_ for STATEFUL
    assert is_fitted(estimator) is False
    estimator.fit(df)
    assert estimator.feature_names_in_ == ["a", "b"]
    assert is_fitted(estimator) is True

    # --- Test fit() with column names stored in feature_names_in_ for STATELESS
    assert is_fitted(stateless_estimator) is True
    stateless_estimator.fit(df)
    assert stateless_estimator.feature_names_in_ == ["a", "b"]
    assert is_fitted(stateless_estimator) is True


def test_pickling(fitted_estimator: HelicastBaseEstimator):
    pickle = BytesIO()
    dump(fitted_estimator, pickle)

    pickle.seek(0)
    new_obj = load(pickle)

    assert id(new_obj) != id(fitted_estimator)
    assert new_obj.__dict__ == fitted_estimator.__dict__


def test_clone(fitted_estimator: HelicastBaseEstimator):
    new_obj = clone(fitted_estimator)

    assert not hasattr(clone, "feature_names_in_")
    assert is_fitted(new_obj) is False

    keys_1 = set(fitted_estimator.__dict__.keys())
    keys_2 = set(new_obj.__dict__.keys())
    for i in keys_1 & keys_2:
        assert getattr(fitted_estimator, i) == getattr(new_obj, i)

    assert id(new_obj) != id(fitted_estimator)


def test_deepcopy(fitted_estimator: HelicastBaseEstimator):
    new_obj = deepcopy(fitted_estimator)

    assert id(new_obj) != id(fitted_estimator)
    assert new_obj.__dict__ == fitted_estimator.__dict__
    assert is_fitted(new_obj) is is_fitted(fitted_estimator)
