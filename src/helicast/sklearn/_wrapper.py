from warnings import warn

import numpy as np
import pandas as pd
from sklearn.base import clone
from typing_extensions import Self

from helicast.base import (
    HelicastBaseEstimator,
    InvertibleTransformerMixin,
    PredictorMixin,
    dataclass,
)
from helicast.typing import PredictorType, TransformerType
from helicast.utils import (
    check_method,
    is_fitable,
    is_invertible_transformer,
    is_predictor,
    is_transformer,
    validate_equal_to_reference,
)

__all__ = [
    "HelicastWrapper",
    "helicast_auto_wrap",
]


def helicast_auto_wrap(
    obj: HelicastBaseEstimator | TransformerType | PredictorType,
) -> HelicastBaseEstimator:
    """Automatically wraps an object in a HelicastWrapper if it is not already a
    HelicastBaseEstimator. It accepts sklearn estimators and HelicastBaseEstimators."""

    if isinstance(obj, HelicastBaseEstimator):
        return obj
    elif isinstance(obj, (TransformerType, PredictorType)):
        return HelicastWrapper(obj)

    raise ValueError(
        f"Object {obj} ({type(obj)=}) should implement the fit/transform or fit/predict methods."
    )


@dataclass
class HelicastWrapper(
    HelicastBaseEstimator, InvertibleTransformerMixin, PredictorMixin
):
    """Wraps a sklearn-like estimator in a HelicastWrapper, such that the estimator
    behaves like a HelicastBaseEstimator (with DataFrame input/output).

    Args:
        estimator: Estimator-like object that should implement the fit/transform or
            fit/predict methods.
    """

    estimator: TransformerType | PredictorType

    def __post_init__(self):
        if isinstance(self.estimator, HelicastWrapper):
            warn(
                "You're wrapping a HelicastWrapper in a HelicastWrapper, this is not "
                "recommended. The estimator of the inner HelicastWrapper will be "
                "exposed to the outer HelicastWrapper, avoiding the double wrapping."
            )
            self.estimator = self.estimator.estimator

        try:
            self.estimator.set_output(transform="pandas")
        except Exception:
            pass

    ##################
    ### PROPERTIES ###
    ##################
    @property
    def _can_fit(self) -> bool:
        return is_fitable(self.estimator)

    @property
    def _can_transform(self) -> bool:
        return is_transformer(self.estimator)

    @property
    def _can_inverse_transform(self) -> bool:
        return is_invertible_transformer(self.estimator)

    @property
    def _can_predict(self) -> bool:
        return is_predictor(self.estimator)

    #####################
    ### SKLEARN LOGIC ###
    #####################
    @check_method(check=is_fitable)
    def _fit(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None, **kwargs
    ) -> Self:
        if y is not None:
            y = y.squeeze()
        return self.estimator.fit(X, y, **kwargs)

    @check_method(check=is_transformer)
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.estimator.transform(X)

    @check_method(check=is_invertible_transformer)
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

    @check_method(check=is_predictor)
    def _predict(self, X: pd.DataFrame) -> pd.DataFrame:
        y_pred = self.estimator.predict(X)

        if not isinstance(y_pred, (np.ndarray, pd.DataFrame)):
            raise TypeError(
                f"y_pred must be a numpy array or a pd.DataFrame. Found {type(y_pred)}."
            )

        if isinstance(y_pred, np.ndarray):
            # Force 2D array
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            y_pred = pd.DataFrame(y_pred, columns=self.target_names_in_, index=X.index)

        return y_pred

    def get_params(self, deep: bool = True) -> dict:
        return self.estimator.get_params(deep)

    def set_params(self, **params):
        return self.estimator.set_params(**params)

    def __sklearn_clone__(self):
        return HelicastWrapper(estimator=clone(self.estimator))

    def __getattr__(self, name):
        try:
            return getattr(self.estimator, name)
        except AttributeError as e:
            raise AttributeError(
                f"Wrapped estimator {self.estimator} does not have attribute {name}."
            ) from e
