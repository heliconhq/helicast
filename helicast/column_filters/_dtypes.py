from logging import getLogger
from typing import List, Union

import numpy as np
import pandas as pd
from pydantic import field_validator

from helicast.column_filters._base import ColumnFilter
from helicast.logging import configure_logging

configure_logging()
logger = getLogger(__name__)


__all__ = [
    "DTypeSelector",
    "select_columns_by_dtype",
    "DTypeRemover",
    "remove_columns_by_dtype",
]


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
