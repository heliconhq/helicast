import logging
from copy import deepcopy
from typing import Annotated, List, Sequence, Union

import pandas as pd
from pydantic import BeforeValidator, Field, PositiveInt, TypeAdapter, ValidationError

from helicast.base import BaseEstimator, StatelessTransformerMixin, dataclass
from helicast.column_filters._base import AllSelector, ColumnFilter
from helicast.logging import configure_logging
from helicast.utils import link_docs_to_class

configure_logging()
logger = logging.getLogger(__name__)

__all__ = [
    "LaggedColumnsAdder",
    "add_lagged_columns",
    "FutureColumnsAdder",
    "add_future_columns",
]


def _validate_positive_shifts(
    v: Union[PositiveInt, List[PositiveInt]],
) -> List[PositiveInt]:
    """Use in a validator to validate shifts.

    * If ``v`` is an integer, it checks that it
    is a strictly positive number and then returns ``list(range(1, v+1))``, e.g.,
    ``v=2`` will return ``[1, 2]``.
    * If ``v`` is some sort of iterable, it casts it onto a Python list and each element
    is checked to be strictly positive, e.g., ``v=[1, 3, 5]`` will work while
    ``v=[0, 1, 3, 5]`` won't.

    """

    value = deepcopy(v)

    error = None
    try:
        value = TypeAdapter(List[int]).validate_python(value)
        try:
            value = TypeAdapter(List[PositiveInt]).validate_python(value)
            if len(value) == 0:
                raise ValueError(f"{v} gives a list of length 0, should be at least 1!")
            return value
        except ValidationError as e:
            error = e
    except ValidationError:
        pass
    if error:
        raise ValueError(error)

    value = TypeAdapter(PositiveInt).validate_python(value)
    value = list(range(1, value + 1))
    return value


@dataclass
class LaggedColumnsAdder(StatelessTransformerMixin, BaseEstimator):
    """_summary_

    Args:
        lags: _description_
        filter: _description_
        dropna_boundary: _description_
    """

    lags: Annotated[
        Union[PositiveInt, Sequence[PositiveInt]],
        BeforeValidator(_validate_positive_shifts),
    ]

    filter: ColumnFilter = AllSelector()

    dropna_boundary: bool = Field(default=True)

    def _transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame | None] = None, **kwargs
    ) -> None:
        X_new = X.copy()
        selected_columns = self.filter(X)
        for column in selected_columns:
            for i in self.lags:
                X_new[f"{column}_lagged_{i}"] = X_new[column].shift(i)

        if self.dropna_boundary:
            X_new = X_new.iloc[max(self.lags) :]

        return X_new


@link_docs_to_class(cls=LaggedColumnsAdder)
def add_lagged_columns(
    X: pd.DataFrame,
    lags: Union[PositiveInt, Sequence[PositiveInt]],
    filter: ColumnFilter = AllSelector(),
    dropna_boundary: bool = True,
):
    tr = LaggedColumnsAdder(lags=lags, filter=filter, dropna_boundary=dropna_boundary)
    return tr.fit_transform(X, None)


@dataclass
class FutureColumnsAdder(StatelessTransformerMixin, BaseEstimator):
    """_summary_

    Args:
        lags: _description_
        filter: _description_
        dropna_boundary: _description_
    """

    futures: Annotated[
        Union[PositiveInt, Sequence[PositiveInt]],
        BeforeValidator(_validate_positive_shifts),
    ]

    filter: ColumnFilter = AllSelector()

    dropna_boundary: bool = Field(default=True)

    def _transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame | None] = None, **kwargs
    ) -> None:
        X_new = X.copy()
        selected_columns = self.filter(X)
        for column in selected_columns:
            for i in self.futures:
                X_new[f"{column}_future_{i}"] = X_new[column].shift(-i)

        if self.dropna_boundary:
            X_new = X_new.iloc[: -max(self.futures)]

        return X_new


@link_docs_to_class(cls=FutureColumnsAdder)
def add_future_columns(
    X: pd.DataFrame,
    futures: Union[PositiveInt, Sequence[PositiveInt]],
    filter: ColumnFilter = AllSelector(),
    dropna_boundary: bool = True,
):
    tr = FutureColumnsAdder(
        futures=futures, filter=filter, dropna_boundary=dropna_boundary
    )
    return tr.fit_transform(X, None)
