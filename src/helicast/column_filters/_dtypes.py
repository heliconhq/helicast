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
    "DTypeSelector",
    "select_columns_by_dtype",
    "DTypeRemover",
    "remove_columns_by_dtype",
]


@dataclass
class DTypeBase(ColumnFilter):
    dtypes: Annotated[Union[str, List[str]], BeforeValidator(_cast_type_to_list(str))]

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


@link_docs_to_class(cls=DTypeSelector)
def select_columns_by_dtype(
    X: pd.DataFrame, dtypes: Union[str, List[str]]
) -> pd.DataFrame:
    return DTypeSelector(dtypes=dtypes).fit_transform(X)


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


@link_docs_to_class(cls=DTypeRemover)
def remove_columns_by_dtype(
    X: pd.DataFrame, dtypes: Union[str, List[str]]
) -> pd.DataFrame:
    return DTypeRemover(dtypes=dtypes).fit_transform(X)
