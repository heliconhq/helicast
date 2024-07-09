import logging
from typing import Annotated, Iterable, List, Sequence, Union

import pandas as pd
from pydantic import BeforeValidator, Field, PositiveInt, TypeAdapter

from helicast.base import HelicastBaseEstimator, StatelessTransformerMixin, dataclass
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


def _validate_positive_range(
    value: Union[PositiveInt, List[PositiveInt]],
) -> List[PositiveInt]:
    """Validate positive ranges. Examples:

    * ``value = 2`` or ``value = "2"`` will return ``[1, 2]``;
    * ``value = (1, 3, 5)`` will return ``[1, 3, 5]``;
    """
    if isinstance(value, Iterable) and not isinstance(value, str):
        value = list(value)
    else:
        value = TypeAdapter(PositiveInt).validate_python(value)
        value = list(range(1, value + 1))

    return TypeAdapter(List[PositiveInt]).validate_python(value)


@dataclass
class LaggedColumnsAdder(StatelessTransformerMixin, HelicastBaseEstimator):
    """Add lagged columns to a DataFrame, that is, columns with values from previous
    rows.

    Args:
        shifts: Lag values to use. If a single value is provided, the shifts will be
            generated from 1 to the provided value. If a list of values is provided,
            the shifts will be the provided values. Shifts must be positive integers.
        dropna_boundary: If True, drop the rows with NaN values at the beginning of the
            DataFrame. Defaults to True.
    """

    shifts: Annotated[
        Union[PositiveInt, Sequence[PositiveInt]],
        BeforeValidator(_validate_positive_range),
    ]

    dropna_boundary: bool = Field(default=True)

    def _transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame | None] = None, **kwargs
    ) -> None:
        new_columns = {}
        for c in X.columns:
            for i in self.shifts:
                new_columns[f"{c}_lagged_{i}"] = X[c].shift(i)

        X_new = pd.concat([X, pd.DataFrame(new_columns)], axis=1)

        if self.dropna_boundary:
            X_new = X_new.iloc[max(self.shifts) :]

        return X_new


@link_docs_to_class(cls=LaggedColumnsAdder)
def add_lagged_columns(
    X: pd.DataFrame,
    shifts: Union[PositiveInt, Sequence[PositiveInt]],
    dropna_boundary: bool = True,
):
    tr = LaggedColumnsAdder(shifts=shifts, dropna_boundary=dropna_boundary)
    return tr.fit_transform(X, None)


@dataclass
class FutureColumnsAdder(StatelessTransformerMixin, HelicastBaseEstimator):
    """Add future columns to a DataFrame.

    Args:
        shifts: Future values to use. If a single value is provided, the shifts will
            be generated from 1 to the provided value. If a list of values is provided,
            the shifts will be the provided values. Shifts must be positive integers.
        dropna_boundary: If True, drop the rows with NaN values at the end of the
            DataFrame. Defaults to True.
    """

    shifts: Annotated[
        Union[PositiveInt, Sequence[PositiveInt]],
        BeforeValidator(_validate_positive_range),
    ]

    dropna_boundary: bool = Field(default=True)

    def _transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame | None] = None, **kwargs
    ) -> None:
        new_columns = {}
        for c in X.columns:
            for i in self.shifts:
                new_columns[f"{c}_future_{i}"] = X[c].shift(-i)

        X_new = pd.concat([X, pd.DataFrame(new_columns)], axis=1)

        if self.dropna_boundary:
            X_new = X_new.iloc[: -max(self.shifts)]

        return X_new


@link_docs_to_class(cls=FutureColumnsAdder)
def add_future_columns(
    X: pd.DataFrame,
    shifts: Union[PositiveInt, Sequence[PositiveInt]],
    dropna_boundary: bool = True,
):
    tr = FutureColumnsAdder(shifts=shifts, dropna_boundary=dropna_boundary)
    return tr.fit_transform(X, None)
