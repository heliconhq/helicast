import re
from abc import ABC, abstractmethod
from logging import getLogger
from typing import List, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator, validate_call
from sklearn.base import BaseEstimator
from typing_extensions import Self

from helicast.base import PydanticBaseEstimator
from helicast.logging import configure_logging
from helicast.utils import get_init_args

configure_logging()
logger = getLogger(__name__)

__all__ = [
    "AllSelector",
]


@validate_call(config={"strict": True})
def _sort_columns(selected_columns: List[str], all_columns: List[str]) -> List[str]:
    """For a list of strings ``selected_columns``, reorder them so that they respect
    the ordering of ``all_columns``. Note that ``selected_columns`` MUST be a subset
    of ``selected_columns`` (otherwise, ``ValueError`` is raised)

    Returns:
        The reordered ``selected_columns``.
    """
    extra = set(selected_columns) - set(all_columns)
    if extra:
        raise ValueError(f"Found columns {extra} which are not in `all_columns`!")
    return [i for i in all_columns if i in set(selected_columns)]


class ColumnFilter(PydanticBaseEstimator, ABC):
    """Abstract base class for DataFrame columns filtering. Child classes need to
    implement the ``_select_columns`` method which takes a pd.DataFrame in and outputs
    a list of columns which are to be selected.
    """

    @abstractmethod
    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        """Abstract method that implements the logic for selecting the columns of a
        pd.DataFrame. Must be implemented in each child class. Don't worry about
        reordering the list, it is done in the __call__ method!

        Args:
            X: Input pd.DataFrame.

        Returns:
            List of selected columns.
        """
        pass

    @validate_call(
        config={
            "arbitrary_types_allowed": True,
            "validate_return": True,
            "strict": True,
        }
    )
    def __call__(self, X: pd.DataFrame) -> List[str]:
        """Returns a list of column names matching the specific condition. The returned
        list is ordered such that it follows ``X.columns`` ordering.

        Args:
            X: Input pd.DataFrame.

        Returns:
            A list of column names which match with the condition.
        """
        columns = self._select_columns(X)
        columns = _sort_columns(columns, X.columns.to_list())

        # Check the dropped columns just for logging
        dropped_columns = list(set(X.columns) - set(columns))
        dropped_columns = _sort_columns(dropped_columns, X.columns.to_list())
        logger.info(f"Selecting {columns}, dropped {dropped_columns}")
        return columns

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> Self:
        """Does nothing, just there for compatibility purposes."""
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs) -> pd.DataFrame:
        """Takes a pd.DataFrame in and outputs another pd.DataFrame whose columns
        have been selected according to the object rule.

        Args:
            X: Input pd.DataFrame
            y: Ignored (defaults to None)

        Raises:
            TypeError: If ``X`` is not a DataFrame.

        Returns:
            A pd.DataFrame whose columns have been filtered.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X should be a pd.DataFrame. Found {type(X)=}.")
        return X[self(X)].copy()

    def fit_transform(self, X: pd.DataFrame, y=None, **kwargs) -> pd.DataFrame:
        """Equivalent to ``self.transform`` as ``self.fit`` does nothing.

        Args:
            X: Input pd.DataFrame
            y: Ignored (defaults to None)

        Returns:
            A pd.DataFrame whose columns have been filtered.
        """
        return self.transform(X, y, **kwargs)

    def __or__(self, __value) -> Self:
        """Implements the basic ``|`` operator (bitwise or)."""
        return ColumnFilterOr(left=self, right=__value)

    def __and__(self, __value) -> Self:
        """Implements the basic ``&`` operator (bitwise and)."""
        return ColumnFilterAnd(left=self, right=__value)

    def __invert__(self) -> Self:
        """Implements the basic ``~`` operator (bitwise not)."""
        return ColumnFilterNot(filter=self)


class ColumnFilterOr(ColumnFilter):
    """Holds the implementation for the bitwise **or** operator ``|`` between two
    ColumnFilter objects. The operator is equivalent to a set union.

    Args:
        left: Left-handside ColumnFilter
        right: Right-handside ColumnFilter
    """

    left: ColumnFilter
    right: ColumnFilter

    class Config:
        arbitrary_types_allowed = True
        strict = True

    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        left_columns = self.left(X)
        right_columns = self.right(X)
        return list(set(left_columns).union(set(right_columns)))

    def __invert__(self) -> ColumnFilter:
        """De Morgan's Laws"""
        return ColumnFilterAnd(left=~self.left, right=~self.right)


class ColumnFilterAnd(ColumnFilter):
    """Holds the implementation for the bitwise **and** operator ``&`` between two
    ColumnFilter objects. The operator is equivalent to a set intersection.

    Args:
        left: Left-handside ColumnFilter
        right: Right-handside ColumnFilter
    """

    left: ColumnFilter
    right: ColumnFilter

    class Config:
        arbitrary_types_allowed = True
        strict = True

    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        left_columns = self.left(X)
        right_columns = self.right(X)
        return list(set(left_columns).intersection(set(right_columns)))

    def __invert__(self) -> ColumnFilter:
        """De Morgan's Laws"""
        return ColumnFilterOr(left=~self.left, right=~self.right)


class ColumnFilterNot(ColumnFilter):
    """Holds the implementation for the binary **not** operator ``~`` for a
    ColumnFilter object. The operator is equivalent to a set different between the set
    of all columns and the set of selected columns by the ColumnFilter object
    ``self.filter``.

    Args:
        filter: ColumnFilter to negate.
    """

    filter: ColumnFilter

    class Config:
        arbitrary_types_allowed = True
        strict = True

    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        columns = self.filter(X)
        return list(set(X.columns) - set(columns))

    def __invert__(self) -> ColumnFilter:
        """'not not' cancels out"""
        return self.filter


class AllSelector(ColumnFilter):
    """Dummy column filter that selects all the columns (does not modify the DataFrame)."""

    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        return X.columns.to_list()

    def __call__(self, X: pd.DataFrame) -> List[str]:
        return X.columns.to_list()
