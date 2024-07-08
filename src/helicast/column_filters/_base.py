from abc import abstractmethod
from logging import getLogger
from typing import List

import pandas as pd
from pydantic import validate_call
from typing_extensions import Self

from helicast.base import BaseEstimator, StatelessTransformerMixin, dataclass
from helicast.logging import configure_logging
from helicast.utils import validate_subset_of_reference

configure_logging()
logger = getLogger(__name__)

__all__ = [
    "AllSelector",
]


@dataclass
class ColumnFilter(StatelessTransformerMixin, BaseEstimator):
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

    @validate_call(config={"arbitrary_types_allowed": True})
    def select_columns(self, X: pd.DataFrame) -> List:
        """Select the columns of a pd.DataFrame according to the object rule and
        return them as a list. The list is ordered such that it follows ``X.columns``
        ordering.

        Args:
            X: Input pd.DataFrame.

        Returns:
            A list of column names which match with the condition.
        """
        columns = self._select_columns(X)
        columns = validate_subset_of_reference(columns, X.columns, reorder=True)
        return columns

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.select_columns(X)]

    def __or__(self, _value) -> Self:
        """Implements the basic ``|`` operator (bitwise or)."""
        return ColumnFilterOr(left=self, right=_value)

    def __and__(self, __value) -> Self:
        """Implements the basic ``&`` operator (bitwise and)."""
        return ColumnFilterAnd(left=self, right=__value)

    def __invert__(self) -> Self:
        """Implements the basic ``~`` operator (bitwise not)."""
        return ColumnFilterNot(filter=self)


@dataclass
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

    def _select_columns(self, X: pd.DataFrame) -> List:
        left_columns = self.left.select_columns(X)
        right_columns = self.right.select_columns(X)
        return list(set(left_columns).union(set(right_columns)))

    def __invert__(self) -> ColumnFilter:
        """De Morgan's Laws"""
        return ColumnFilterAnd(left=~self.left, right=~self.right)


@dataclass
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
        left_columns = self.left.select_columns(X)
        right_columns = self.right.select_columns(X)
        return list(set(left_columns).intersection(set(right_columns)))

    def __invert__(self) -> ColumnFilter:
        """De Morgan's Laws"""
        return ColumnFilterOr(left=~self.left, right=~self.right)


@dataclass
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
        columns = self.filter.select_columns(X)
        return list(set(X.columns) - set(columns))

    def __invert__(self) -> ColumnFilter:
        """'not not' cancels out"""
        return self.filter


@dataclass
class AllSelector(ColumnFilter):
    """Dummy column filter that selects all the columns (does not modify the DataFrame)."""

    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        return X.columns.to_list()
