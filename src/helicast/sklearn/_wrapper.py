from functools import partial, wraps
from inspect import isclass
from typing import Any, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from typing_extensions import Self

from helicast.base import (
    BaseEstimator,
    InvertibleTransformerMixin,
    PredictorMixin,
    dataclass,
)
from helicast.utils import validate_equal_to_reference

__all__ = [
    "HelicastWrapper",
]


def _has_method(obj, name: str) -> bool:
    """ "Check if a class has a method with the given name."""
    if hasattr(obj, name) and callable(getattr(obj, name, None)):
        return True
    return False


def check_method(method=None, *, name: str):
    """Ensure that the estimator attribute has a method with the given name ``name``."""
    if method is None:
        return partial(check_method, name=name)

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not _has_method(self.estimator, name):
            estimator_name = self.estimator.__class__.__name__
            raise AttributeError(
                f"Estimator {estimator_name} does not have '{name}' method."
            )
        return method(self, *args, **kwargs)

    return wrapper


@dataclass
class HelicastWrapper(BaseEstimator, InvertibleTransformerMixin, PredictorMixin):
    estimator: Any

    def __post_init__(self):
        if hasattr(self.estimator, "set_output"):
            self.estimator.set_output(transform="pandas")

    @check_method(name="fit")
    def _fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, None], **kwargs
    ) -> Self:
        if y is not None:
            y = y.squeeze()
        return self.estimator.fit(X, y)

    @check_method(name="transform")
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.estimator.transform(X)

    @check_method(name="inverse_transform")
    def _inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_inv_tr = self.estimator.inverse_transform(X)

        if isinstance(X_inv_tr, np.ndarray):
            X_inv_tr = pd.DataFrame(X_inv_tr, index=X.index)

        if len(X_inv_tr.columns) != len(self.feature_names_in_):
            raise ValueError(
                f"Columns of X_inv_tr do not match feature_names_in_. Found "
                f"{len(X_inv_tr.columns)} columns, expected {len(self.feature_names_in_)}."
            )
        if set(X_inv_tr.columns) != set(self.feature_names_in_):
            X_inv_tr.columns = self.feature_names_in_
        else:
            columns = validate_equal_to_reference(
                X_inv_tr.columns, self.feature_names_in_
            )
            X_inv_tr = X_inv_tr[columns]
        return X_inv_tr

    @check_method(name="predict")
    def _predict(self, X: pd.DataFrame) -> pd.DataFrame:
        y_pred = self.estimator.predict(X)

        if isinstance(y_pred, np.ndarray):
            # Force 2D array
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            y_pred = pd.DataFrame(y_pred, columns=self.target_names_in_, index=X.index)
        elif isinstance(y_pred, pd.DataFrame):
            pass
        else:
            raise TypeError(
                f"y_pred must be a numpy array or a pd.DataFrame. Found {type(y_pred)}."
            )
        return y_pred

    def get_params(self, deep: bool = True) -> dict:
        return self.estimator.get_params(deep)

    def set_params(self, **params):
        return self.estimator.set_params(**params)

    def __sklearn_clone__(self):
        return HelicastWrapper(estimator=clone(self.estimator))
