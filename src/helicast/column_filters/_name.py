from logging import getLogger
from typing import List, Union

import numpy as np
import pandas as pd
from pydantic import field_validator

from helicast.base import dataclass
from helicast.column_filters._base import ColumnFilter
from helicast.logging import configure_logging

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
    """Base model for name-based filtering."""

    names: Union[str, List[str]]

    strict: bool = True

    @field_validator("names")
    @classmethod
    def auto_cast_to_list(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            v = [v]
        return v

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


def select_columns_by_names(
    df: pd.DataFrame, names: Union[str, List[str]], strict: bool = True
) -> pd.DataFrame:
    return NameSelector(names=names, strict=strict).fit_transform(df)


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


def remove_columns_by_names(
    df: pd.DataFrame, names: Union[str, List[str]], strict: bool = True
) -> pd.DataFrame:
    return NameRemover(names=names, strict=strict).fit_transform(df)
