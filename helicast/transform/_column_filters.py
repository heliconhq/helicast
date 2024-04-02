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
    "RegexSelector",
    "select_columns_by_regex",
    "RegexRemover",
    "remove_columns_by_regex",
    "DTypeSelector",
    "select_columns_by_dtype",
    "DTypeRemover",
    "remove_columns_by_dtype",
    "AllSelector",
    "NameSelector",
    "select_columns_by_names",
    "NameRemover",
    "remove_columns_by_names",
]


@validate_call(config={"strict": True})
def _sort_columns(selected_columns: List[str], all_columns: List[str]) -> List[str]:
    """For a list of strings ``selected_columns``, reorder them so that they respect
    the ordering of ``all_columns``. Note that ``selected_columns`` MUST be a subset
    of ``selected_columns`` (otherwise, a ``ValueError`` is raised)

    Returns:
        The reordered ``selected_columns``.
    """
    extra = set(selected_columns) - set(all_columns)
    if extra:
        raise ValueError(f"Found columns {extra} which are not in `all_columns`!")
    return [i for i in all_columns if i in set(selected_columns)]


class ColumnFilter(PydanticBaseEstimator, ABC):
    """Abstract base class for DataFrame column filtering. Child classes need to
    implement the ``_select_columns`` method which takes a pd.DataFrame in and outputs
    a list of columns which are to be selected.
    """

    @abstractmethod
    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        """Abstract method that implements the logic for selecting the columns of a
        pd.DataFrame. Must be implemented in each child class. Don't worry about
        reordering the list, everything is done in the __call__ method!

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
        list is ordered such that it follows X.columns ordering.

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

    def fit(self, X, y=None, **kwargs) -> Self:
        """Does nothing, just there for compatibility purposes."""
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs) -> pd.DataFrame:
        """Takes a pd.DataFrame in and outputs another pd.DataFrame whose columns
        have been selected according to the object rule.

        Args:
            X: Input pd.DataFrame

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
        return ColumnFilterOr(left=self, right=__value)

    def __and__(self, __value) -> Self:
        return ColumnFilterAnd(left=self, right=__value)

    def __invert__(self) -> Self:
        return ColumnFilterNot(filter=self)


class ColumnFilterOr(ColumnFilter):
    """Holds the implementation for the binary **or** operator ``|`` between two
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
    """Holds the implementation for the binary **and** operator ``&`` between two
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


class RegexBase(ColumnFilter):
    """Base model for regex-based filtering."""

    patterns: Union[str, List[str]]

    @field_validator("patterns")
    @classmethod
    def auto_cast_to_list(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            v = [v]
        return v

    def _get_matched_columns(self, X: pd.DataFrame) -> List[str]:
        matched_columns = set()
        for pattern in self.patterns:
            local_match = [c for c in X.columns if re.search(pattern, c)]
            logger.info(f"For pattern '{pattern}', matched {local_match}.")
            matched_columns = matched_columns.union(set(local_match))

        matched_columns = list(matched_columns)
        logger.info(f"All unique columns matching: {matched_columns}.")
        return matched_columns

    def __or__(self, __value: object) -> ColumnFilter:
        if isinstance(__value, self.__class__):
            patterns = self.patterns + __value.patterns
            patterns = list(np.unique(patterns))
            return self.__class__(patterns=patterns)
        else:
            return ColumnFilter.__or__(self, __value)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, self.__class__):
            return set(self.patterns) == set(__value.patterns)
        return False


class RegexSelector(RegexBase):
    """ColumnFilter implementing a regex selection rule.

    Args:
        patterns: str of list of str that specifies the regex patterns. If a list of str
         is provided, a column will be matched if it matches with at least one of the
         patterns.
    """

    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        return self._get_matched_columns(X)

    def __invert__(self):
        return RegexRemover(patterns=self.patterns)


def select_columns_by_regex(
    df: pd.DataFrame, patterns: Union[str, List[str]]
) -> pd.DataFrame:
    return RegexSelector(patterns=patterns).fit_transform(df)


class RegexRemover(RegexBase):
    """ColumnFilter implementing a regex exclusion rule.

    Args:
        patterns: str of list of str that specifies the regex patterns. If a list of str
         is provided, a column will be matched if it matches with at least one of the
         patterns.
    """

    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        return list(set(X.columns) - set(self._get_matched_columns(X)))

    def __invert__(self):
        return RegexSelector(patterns=self.patterns)


def remove_columns_by_regex(
    df: pd.DataFrame, patterns: Union[str, List[str]]
) -> pd.DataFrame:
    return RegexRemover(patterns=patterns).fit_transform(df)


class DTypeBase(ColumnFilter):
    dtypes: Union[str, List[str]]

    @field_validator("dtypes")
    @classmethod
    def auto_cast_to_list(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            v = [v]
        return v

    def __or__(self, __value: object) -> ColumnFilter:
        if isinstance(__value, self.__class__):
            dtypes = self.dtypes + __value.dtypes
            dtypes = list(np.unique(dtypes))
            return self.__class__(dtypes=dtypes)
        else:
            return ColumnFilter.__or__(self, __value)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, self.__class__):
            return set(self.dtypes) == set(__value.dtypes)
        return False


class DTypeSelector(DTypeBase):
    """ColumnFilter implementing a dtype selection rule.

    Args:
        dtypes: str of list of str that specifies the types, as specified in pandas
         ``select_dtypes`` pd.DataFrame method. Here, ``dtypes`` will be passed as
         the ``include`` argument of ``select_dtypes``.
    """

    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        return X.select_dtypes(include=self.dtypes).columns.to_list()

    def __invert__(self):
        return DTypeRemover(dtypes=self.dtypes)


def select_columns_by_dtype(
    df: pd.DataFrame, dtypes: Union[str, List[str]]
) -> pd.DataFrame:
    return DTypeSelector(dtypes=dtypes).fit_transform(df)


class DTypeRemover(DTypeBase):
    """ColumnFilter implementing a dtype exclusion rule.

    Args:
        dtypes: str of list of str that specifies the types, as specified in pandas
         ``select_dtypes`` pd.DataFrame method. Here, ``dtypes`` will be passed as
         the ``exclude`` argument of ``select_dtypes``.
    """

    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        return X.select_dtypes(exclude=self.dtypes).columns.to_list()

    def __invert__(self):
        return DTypeSelector(dtypes=self.dtype)


def remove_columns_by_dtype(
    df: pd.DataFrame, dtypes: Union[str, List[str]]
) -> pd.DataFrame:
    return DTypeRemover(dtypes=dtypes).fit_transform(df)


class AllSelector(ColumnFilter):
    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        return X.columns.to_list()

    def __call__(self, X: pd.DataFrame) -> List[str]:
        return X.columns.to_list()


class NameBase(ColumnFilter):
    """Base model for name-based filtering."""

    names: Union[str, List[str]]

    @field_validator("names")
    @classmethod
    def auto_cast_to_list(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            v = [v]
        return v

    def __or__(self, __value: object) -> ColumnFilter:
        if isinstance(__value, self.__class__):
            names = self.names + __value.names
            names = list(np.unique(names))
            return self.__class__(names=names)
        else:
            return ColumnFilter.__or__(self, __value)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, self.__class__):
            return set(self.names) == set(__value.names)
        return False


class NameSelector(NameBase):
    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        return list(set(self.names))


def select_columns_by_names(
    df: pd.DataFrame, names: Union[str, List[str]]
) -> pd.DataFrame:
    return NameSelector(names=names).fit_transform(df)


class NameRemover(NameBase):
    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        return list(set(X.columns) - set(self.names))


def remove_columns_by_names(
    df: pd.DataFrame, names: Union[str, List[str]]
) -> pd.DataFrame:
    return NameRemover(names=names).fit_transform(df)
