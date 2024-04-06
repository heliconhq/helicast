import logging
from typing import List, Literal, Union

import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score
from typing_extensions import Self

from helicast.base import PydanticBaseEstimator, _validate_X, _validate_X_y
from helicast.logging import configure_logging
from helicast.typing import UNSET

configure_logging()
logger = logging.getLogger(__name__)


__all__ = ["PandasTransformedTargetRegressor"]


def _check_columns(columns: List[str], reference_columns: List[str]) -> None:
    """Check that ``columns`` and ``reference_columns`` contains exactly the same
    elements, otherwise, raise a verbose ``RuntimeError``."""
    extra = set(columns) - set(reference_columns)
    missing = set(reference_columns) - set(columns)

    msg = []
    if extra:
        msg.append(f"Columns {extra} were not seen during fit!")
    if missing:
        msg.append(f"Columns {missing} were seen during fit but are missing!")

    if msg:
        msg = [f" -{i}\n" for i in msg]
        raise RuntimeError(f"Invalid columns:\n{msg}")


class PandasTransformedTargetRegressor(PydanticBaseEstimator):
    regressor: BaseEstimator
    transformer: Union[BaseEstimator, None] = None
    no_overlap: bool = True

    feature_names_in_: List[str] = Field(UNSET, init=False, repr=False)
    target_names_in_: List[str] = Field(UNSET, init=False, repr=False)

    def _check_is_fitted(self):
        attrs = ["feature_names_in_", "target_names_in_"]
        if any(getattr(self, i, UNSET) is UNSET for i in attrs):
            raise RuntimeError(
                f"You must first fit the {self.__class__.__name__} object!"
            )

    def _check_X_columns(self, X: pd.DataFrame) -> None:
        _check_columns(X.columns.to_list(), self.feature_names_in_)

    def _check_y_columns(self, y: pd.DataFrame) -> None:
        _check_columns(y.columns.to_list(), self.target_names_in_)

    def _check_X_y_columns(self, X: pd.DataFrame, y: pd.DataFrame):
        self._check_X_columns(X)
        self._check_y_columns(y)

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series, None] = None,
        **fit_params,
    ) -> Self:

        X, y = _validate_X_y(X, y, same_length=True)
        if y is None:
            raise TypeError("y cannot be None!")

        self.feature_names_in_ = X.columns.to_list()
        self.target_names_in_ = y.columns.to_list()

        y_trans = self.transformer.fit_transform(y)
        if isinstance(y_trans, np.ndarray):
            if y_trans.ndim == 1:
                y_trans = np.reshape(y_trans, (-1, 1))
            y_trans = pd.DataFrame(y_trans, columns=y.columns, index=y.index)

        self.regressor.fit(X, y_trans, **fit_params)
        return self

    def predict(self, X: pd.DataFrame, **predict_params) -> pd.DataFrame:
        self._check_is_fitted()
        X = _validate_X(X)
        self._check_X_columns(X)

        # Reorder columns!
        y_hat = self.regressor.predict(X[self.feature_names_in_], **predict_params)
        y_hat = self.transformer.inverse_transform(y_hat)

        if isinstance(y_hat, pd.DataFrame):
            if set(y_hat.columns) != set(self.target_names_in_):
                raise ValueError(f"{y_hat.columns=}, {self.target_names_in_=}.")
        elif isinstance(y_hat, pd.Series):
            y_hat = y_hat.to_frame(
                name=y_hat.name if y_hat.name is not None else self.target_names_in_[0]
            )
        elif isinstance(y_hat, np.ndarray):
            if y_hat.ndim == 1:
                y_hat = np.reshape(y_hat, (-1, 1))
            y_hat = pd.DataFrame(y_hat, columns=self.target_names_in_, index=X.index)
        else:
            raise TypeError(f"Unexpected type {type(y_hat)=}")

        return y_hat

    def score(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series, None] = None,
        scoring: Literal["r2", "mse", "rmse"] = "r2",
    ) -> float:
        X, y = _validate_X_y(X, y)
        if y is None:
            raise TypeError(f"{type(y)=} cannot be None!")
        self._check_X_y_columns(X, y)

        y_hat = self.predict(X)

        match scoring:
            case "r2":
                score = float(r2_score(y, y_hat))
            case "mse":
                score = float(mean_squared_error(y, y_hat))
            case "rmse":
                score = float(mean_squared_error(y, y_hat))
                score = float(np.sqrt(score))
            case _:
                raise ValueError(f"{scoring=}")

        return score
