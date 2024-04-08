import logging
from typing import List, Literal, Union

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score
from typing_extensions import Self

from helicast.base import (
    PydanticBaseEstimator,
    _check_fitted,
    _check_X_columns,
    _validate_X,
    _validate_X_y,
    cast_to_DataFrame,
    ignore_data_conversion_warnings,
    reoder_columns_if_needed,
)
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
    """
    TODO: Document the class
    """

    regressor: BaseEstimator
    transformer: Union[BaseEstimator, None] = None
    no_overlap: bool = True

    _feature_names_in: List[str] = PrivateAttr(UNSET)
    _target_names_in: List[str] = PrivateAttr(UNSET)

    @ignore_data_conversion_warnings
    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series, None] = None,
        **fit_params,
    ) -> Self:

        X, y = _validate_X_y(X, y, same_length=True)
        if y is None:
            raise TypeError("y cannot be None!")

        self._feature_names_in = X.columns.to_list()
        self._target_names_in = y.columns.to_list()

        if self.transformer is None:
            y_trans = y.copy()
        else:
            y_trans = self.transformer.fit_transform(y)

        y_trans = cast_to_DataFrame(y_trans, columns=y.columns, index=y.index)
        self.regressor.fit(X, y_trans, **fit_params)
        return self

    @ignore_data_conversion_warnings
    def predict(self, X: pd.DataFrame, **predict_params) -> pd.DataFrame:
        _check_fitted(self, ["_feature_names_in", "_target_names_in"])
        X = _validate_X(X)
        _check_X_columns(self, X, self._feature_names_in)

        # Predict
        X = reoder_columns_if_needed(X, columns=self._feature_names_in)
        y_hat = self.regressor.predict(X, **predict_params)
        y_hat = cast_to_DataFrame(y_hat, columns=self._target_names_in, index=X.index)

        # Rescale into the original units
        if self.transformer is not None:
            y_hat = self.transformer.inverse_transform(y_hat)
            y_hat = cast_to_DataFrame(
                y_hat, columns=self._target_names_in, index=X.index
            )

        return y_hat

    @ignore_data_conversion_warnings
    def score(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series, None] = None,
        scoring: Literal["r2", "mse", "rmse"] = "r2",
    ) -> float:
        X, y = _validate_X_y(X, y, same_length=True, allow_y_None=False)

        X = reoder_columns_if_needed(X, self._feature_names_in)
        y = reoder_columns_if_needed(y, self._target_names_in)

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
