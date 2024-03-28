from logging import getLogger
from typing import ClassVar, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import PrivateAttr
from sklearn.base import BaseEstimator

from helicast.logging import configure_logging
from helicast.transform._base import ColumnType, DataFrameTransformer
from helicast.typing import UNSET

configure_logging()
logger = getLogger(__name__)

__all__ = ["MaxAbsScaler", "StandardScaler", "QuantileTransformer"]


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler


class MinMaxScaler(_MinMaxScaler):
    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, None] = None):
        X, y = DataFrameTransformer._validate_X_y(self, X, y)
        return super().fit(X, y)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = super().transform(X)
        X_new = pd.DataFrame(X, columns=X.columns, index=X.index)
        return X_new


class SklearnTransformer(DataFrameTransformer):
    _transformer_type: ClassVar[type]

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> None:
        self._transformer = self._transformer_type(**self.get_params())
        if y is None:
            self._transformer.fit(X, None)
        else:
            self._transformer.fit(X, y)

    def _transform(
        self, X: pd.DataFrame, y: pd.DataFrame | None = None
    ) -> pd.DataFrame | np.ndarray:

        return self._transformer.transform(X)

    def _inverse_transform(
        self, X: pd.DataFrame, y: pd.DataFrame | None = None
    ) -> pd.DataFrame | np.ndarray:

        return self._transformer.inverse_transform(X)


class MaxAbsScaler(SklearnTransformer):
    from sklearn.preprocessing import MaxAbsScaler as _MaxAbsScaler

    _considered_types: ClassVar[ColumnType] = ColumnType.NUMBER
    _transformer_type: ClassVar[type] = PrivateAttr(_MaxAbsScaler)


class StandardScaler(SklearnTransformer):
    from sklearn.preprocessing import StandardScaler as _StandardScaler

    with_mean: bool = True
    with_std: bool = True

    _considered_types: ClassVar[ColumnType] = ColumnType.NUMBER
    _transformer_type: ClassVar[type] = PrivateAttr(_StandardScaler)


class QuantileTransformer(SklearnTransformer):
    from sklearn.preprocessing import QuantileTransformer as _QuantileTransformer

    n_quantiles: int = 1000
    output_distribution: Literal["uniform", "normal"] = "uniform"
    subsample: int = 10000
    random_state: Optional[int] = None

    _considered_types: ClassVar[ColumnType] = ColumnType.NUMBER
    _transformer_type: ClassVar[type] = PrivateAttr(_QuantileTransformer)
