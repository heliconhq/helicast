from logging import getLogger
from typing import Annotated, List, Union

import numpy as np
import pandas as pd
from pydantic import BeforeValidator

from helicast.base import dataclass
from helicast.column_filters._base import ColumnFilter, _cast_type_to_list
from helicast.logging import configure_logging
from helicast.utils import link_docs_to_class

configure_logging()
logger = getLogger(__name__)


__all__ = [
    "NameSelector",
    "select_columns_by_names",
    "NameRemover",
    "remove_columns_by_names",
]


@dataclass
class NameBase(ColumnFilter):
    """Abstract base class name-based filtering. Both ``NameSelector`` and
    ``NameRemover`` inherit from this class."""

    names: Annotated[Union[str, List[str]], BeforeValidator(_cast_type_to_list(str))]

    strict: bool = True

    def _check_names(self, X: pd.DataFrame) -> None:
        extras = set(self.names) - set(X.columns)
        if extras and self.strict:
            raise KeyError(
                f"Column names {extras} are not in DataFrame `X`! You can set `strict` "
                "to False if you want to ignore this."
            )

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


@dataclass
class NameSelector(NameBase):
    """Select the columns of the DataFrame specified in ``names``.

    Args:
        names: Name of the column to select as a string or list of string of columns to
         select. If it is a string, it is automatically wrapped onto a list of
         one element.
        strict: If True, ``names`` must contains strictly valid column names, that is,
         column names which will be in the DataFrame. If False, invalid names are
         quietly ingored. Defaults to True
    """

    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        self._check_names(X)
        return list(set(self.names).intersection(set(X.columns)))


@link_docs_to_class(cls=NameSelector)
def select_columns_by_names(
    X: pd.DataFrame, names: Union[str, List[str]], strict: bool = True
) -> pd.DataFrame:
    return NameSelector(names=names, strict=strict).fit_transform(X)


@dataclass
class NameRemover(NameBase):
    """Remove the columns of the DataFrame specified in ``names``.

    Args:
        names: Name of the column to remove as a string or list of string of columns to
         select. If it is a string, it is automatically wrapped onto a list of
         one element.
        strict: If True, ``names`` must contains strictly valid column names, that is,
         column names which will be in the DataFrame. If False, invalid names are
         quietly ingored. Defaults to True
    """

    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        self._check_names(X)
        return list(set(X.columns) - set(self.names))


@link_docs_to_class(cls=NameRemover)
def remove_columns_by_names(
    X: pd.DataFrame, names: Union[str, List[str]], strict: bool = True
) -> pd.DataFrame:
    return NameRemover(names=names, strict=strict).fit_transform(X)
